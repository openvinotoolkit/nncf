# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from abc import ABC
from abc import abstractmethod
from copy import deepcopy
from typing import Any, List, Tuple, Union

# This should be set befre any torch import
# to enable speed up for quantized model
# comipled with torch.compile.
os.environ["TORCHINDUCTOR_FREEZING"] = "1"

import argparse
import re
import subprocess
import traceback
import warnings
from pathlib import Path
from time import time

import openvino as ov
import openvino.torch  # noqa
import pandas as pd
import torch
import torch.fx
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import convert_pt2e
from torch.ao.quantization.quantize_pt2e import prepare_pt2e
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
from torch.ao.quantization.quantizer.x86_inductor_quantizer import get_default_x86_inductor_quantization_config
from torch.fx.passes.graph_drawer import FxGraphDrawer
from torch.jit import TracerWarning

import nncf
from nncf import AdvancedQuantizationParameters
from nncf.common.factory import NNCFGraphFactory
from nncf.experimental.torch.fx.quantization.backend_parameters import FXBackendParameters
from nncf.torch.dynamic_graph.patch_pytorch import disable_patching
from tests.torch.fx.performance_check.model_scope import MODEL_SCOPE
from tests.torch.fx.performance_check.model_scope import ModelConfig

warnings.filterwarnings("ignore", category=TracerWarning)
warnings.filterwarnings("ignore", category=UserWarning)

VISUALIZE_FX_INT8_GRAPH = True


class ExportInterface(ABC):
    @abstractmethod
    def __call__(self, model: Any, model_config: ModelConfig, path_to_save_model: Path) -> Any:
        """
        Converts passed torch.nn.Module to the target representation
        """

    @abstractmethod
    def name(self) -> str:
        """
        Return name of the export before quantization stage.
        """


class NoExport(ExportInterface):
    def __call__(self, model: Any, model_config: ModelConfig, path_to_save_model: Path) -> Any:
        return model

    def name(self) -> str:
        return "No export"


class CapturePreAutogradGraphExport(ExportInterface):
    def __call__(self, model: Any, model_config: ModelConfig, path_to_save_model: Path) -> torch.fx.GraphModule:
        with disable_patching():
            with torch.no_grad():
                return capture_pre_autograd_graph(model, args=model_config.model_builder.get_example_inputs())

    def name(self) -> str:
        return "capture_pre_autograd_graph"


class TorchExport(ExportInterface):
    def __call__(self, model: Any, model_config: ModelConfig, path_to_save_model: Path) -> Any:
        with disable_patching():
            with torch.no_grad():
                return torch.export.export(
                    model, args=model_config.model_builder.get_example_inputs(), strict=model_config.torch_export_strict
                ).module()

    def name(self) -> str:
        return "torch.export.export"


class OpenvinoIRExport(ExportInterface):
    def __call__(self, model: Any, model_config: ModelConfig, path_to_save_model: Path) -> Any:
        with disable_patching():
            with torch.no_grad():
                example_inputs = model_config.model_builder.get_example_inputs()
                export_inputs = example_inputs[0] if isinstance(example_inputs[0], tuple) else example_inputs
                input_sizes = model_config.model_builder.get_input_sizes()
                ex_model = torch.export.export(model, export_inputs)
                ov_model = ov.convert_model(ex_model, example_input=example_inputs[0], input=input_sizes)
                ov.serialize(ov_model, path_to_save_model)
                return ov_model

    def name(self) -> str:
        return "Export to openvino IR"


class TorchCompileExport(ExportInterface):
    def __call__(self, model: Any, model_config: ModelConfig, path_to_save_model: Path):
        return torch.compile(model)

    def name(self) -> str:
        return "torch.compile(...)"


class TorchCompileOVExport(ExportInterface):
    def __call__(self, model: Any, model_config: ModelConfig, path_to_save_model: Path):
        return torch.compile(model, backend="openvino")

    def name(self) -> str:
        return "torch.compile(..., backend='openvino')"


class CompressionInference(ABC):
    @abstractmethod
    def __call__(self, model: Any, model_config: ModelConfig, save_dir: Path):
        """
        Quantizes given model with given parameters
        """

    @abstractmethod
    def name(self) -> str:
        """
        The name of the quantization stage.
        """


class NoQuantize(CompressionInference):
    def __call__(self, model: Any, model_config: ModelConfig, save_dir: Path) -> Any:
        return model

    def name(self) -> str:
        return "No quantization"


class NNCFQuantize(CompressionInference):
    def __init__(self, compress_weights: bool, serialize_fx_int8_graph: bool = VISUALIZE_FX_INT8_GRAPH):
        self.serialize_fx_int8_graph = serialize_fx_int8_graph
        self.compress_weights = compress_weights

    def __call__(self, model: Any, model_config: ModelConfig, save_dir: Path) -> Any:
        advanced_parameters = model_config.quantization_params.get(
            "advanced_parameters", AdvancedQuantizationParameters()
        )
        advanced_parameters.backend_params[FXBackendParameters.COMPRESS_WEIGHTS] = self.compress_weights
        model_config.quantization_params["advanced_parameters"] = advanced_parameters
        with disable_patching():
            with torch.no_grad():
                example_inputs = model_config.model_builder.get_example_inputs()
                quantized_model = nncf.quantize(
                    model,
                    nncf.Dataset(example_inputs),
                    **model_config.quantization_params,
                )
        backend = ""
        if isinstance(quantized_model, ov.Model):
            backend = "OV"
            ov_int8_model_path = save_dir / "openvino_int8_model.xml"
            ov.serialize(quantized_model, ov_int8_model_path)
            print(f"Openvino quantized model saved to {ov_int8_model_path}")

        elif isinstance(quantized_model, torch.fx.GraphModule):
            backend = "FX"
            _save_int8_torch_fx_info(quantized_model, save_dir, self.serialize_fx_int8_graph, "nncf")

        int8_graph_visualization_path = str(save_dir / f"{backend}_int8_nncf_graph.dot")
        NNCFGraphFactory.create(quantized_model).visualize_graph(int8_graph_visualization_path)
        print(f"NNCFGraph visualization of int8 model is saved to {int8_graph_visualization_path}")

        return quantized_model

    def name(self) -> str:
        return f"nncf.quantize(compress_weights=={self.compress_weights})"


class TorchAOQuantize(CompressionInference):
    def __init__(self, fold_quantize: bool, serialize_fx_int8_graph: bool = VISUALIZE_FX_INT8_GRAPH) -> None:
        """
        fold_quantize == False for the torch.compile("openvino") inference
        """
        self.fold_quantize = fold_quantize
        self.serialize_fx_int8_graph = serialize_fx_int8_graph

    def __call__(self, model: Any, model_config: ModelConfig, save_dir: Path) -> torch.fx.GraphModule:
        assert isinstance(model, torch.fx.GraphModule)
        quantizer = X86InductorQuantizer()
        quantizer.set_global(get_default_x86_inductor_quantization_config())

        with disable_patching():
            example_inputs = model_config.model_builder.get_example_inputs()
            export_inputs = example_inputs[0] if isinstance(example_inputs[0], tuple) else example_inputs
            prepared_model = prepare_pt2e(model, quantizer)
            prepared_model(*export_inputs)
            quantized_model = convert_pt2e(prepared_model, fold_quantize=self.fold_quantize)
            _save_int8_torch_fx_info(quantized_model, save_dir, self.serialize_fx_int8_graph, "torch_ao")
            return quantized_model

    def name(self) -> str:
        return f"torch.ao quantization (fold_quantize=={self.fold_quantize})"


def _save_int8_torch_fx_info(
    quantized_model: torch.fx.GraphModule, save_dir: Path, serialize_fx_int8_graph: bool, q_backend: str
):
    int8_code_path = str(save_dir / f"int8_code_{q_backend}.py")
    with open(int8_code_path, "w") as f:
        f.write(quantized_model.code)
    print(f"int8 FX code is saved to {int8_code_path}")

    if serialize_fx_int8_graph:
        int8_model_visualization_path = str(save_dir / f"int8_fx_graph_q_backend_{q_backend}.svg")
        g = FxGraphDrawer(quantized_model, int8_model_visualization_path)
        g.get_dot_graph().write_svg(int8_model_visualization_path)
        print(f"Visualization of int8 model is saved to {int8_model_visualization_path}")


class BenchmarkInterface(ABC):
    @abstractmethod
    def __call__(self, model: Any, model_config: ModelConfig, model_path: Path) -> Any:
        """
        Benchmarks given model.
        """

    @abstractmethod
    def name(self) -> str:
        """
        Name of the Benchmarking stage.
        """


class LatencyBenchmark(BenchmarkInterface):
    def __call__(self, model: Any, model_config: ModelConfig, model_path: Path) -> Any:
        with disable_patching():
            with torch.no_grad():
                example_inputs = model_config.model_builder.get_example_inputs()
                if isinstance(model, ov.Model):
                    return measure_time_ov(model, example_inputs, model_config.num_iters)
                return measure_time(model, example_inputs, model_config.num_iters)

    def name(self) -> str:
        return "Latency, msec"


class BenchmarkAppFPS(BenchmarkInterface):
    def __call__(self, model: Any, model_config: ModelConfig, model_path: Path) -> Any:
        return benchmark_performance(model_path, model_config.model_builder.get_input_sizes())

    def name(self) -> str:
        return "FPS"


def measure_time(model, example_inputs, num_iters=500):
    with torch.no_grad():
        model(*example_inputs)
        total_time = 0
        for _ in range(num_iters):
            start_time = time()
            model(*example_inputs)
            total_time += time() - start_time
        average_time = (total_time / num_iters) * 1000
    return average_time


def measure_time_ov(model, example_inputs, num_iters=500):
    ie = ov.Core()
    compiled_model = ie.compile_model(model, "CPU")
    infer_request = compiled_model.create_infer_request()
    infer_request.infer(example_inputs)
    total_time = 0
    for _ in range(num_iters):
        start_time = time()
        infer_request.infer(example_inputs)
        total_time += time() - start_time
    average_time = (total_time / num_iters) * 1000
    return average_time


def benchmark_performance(model_path, input_shape) -> float:
    command = f"benchmark_app -m {model_path} -d CPU -api async -t 30"
    command += f' -shape "[{",".join(str(s) for s in input_shape)}]"'
    cmd_output = subprocess.check_output(command, shell=True)  # nosec

    match = re.search(r"Throughput\: (.+?) FPS", str(cmd_output))
    return float(match.group(1))


class BenchmarkPipeline:
    def __init__(
        self,
        export_before_q: ExportInterface,
        compress: CompressionInference,
        benchmarks: List[Tuple[ExportInterface, Union[List[BenchmarkInterface], BenchmarkInterface]]],
    ):
        self.export_before_q = export_before_q
        self.compress = compress
        self.benchmarks = benchmarks

    def run(self, model_name: torch.nn.Module, model_config: ModelConfig, save_dir: Path):
        pt_model = model_config.model_builder.build()

        exported_model = self.export_before_q(
            model=pt_model, model_config=model_config, path_to_save_model=save_dir / "ov_fp32_model.xml"
        )
        compressed_model = self.compress(exported_model, model_config, save_dir)
        prefix = [model_name, self.export_before_q.name(), self.compress.name()]

        keys, values = [], []
        for export_after, benchmarks in self.benchmarks:
            compressed_model_path = save_dir / "ov_int8_model.xml"
            exported_compressed_model = export_after(
                model=deepcopy(compressed_model), model_config=model_config, path_to_save_model=compressed_model_path
            )
            benchmarks = benchmarks if isinstance(benchmarks, list) else [benchmarks]
            for benchmark in benchmarks:
                key = tuple(prefix + [export_after.name(), benchmark.name()])
                value = benchmark(exported_compressed_model, model_config, compressed_model_path)
                print("; ".join(key) + f" : {value}")
                keys.append(key)
                values.append(value)
        return keys, values


PIPELINES = (
    BenchmarkPipeline(
        # TorchExport(),
        CapturePreAutogradGraphExport(),
        NoQuantize(),
        [
            (TorchCompileExport(), LatencyBenchmark()),
            (TorchCompileOVExport(), LatencyBenchmark()),
            (OpenvinoIRExport(), LatencyBenchmark()),
        ],
    ),
    BenchmarkPipeline(
        # TorchExport(),
        CapturePreAutogradGraphExport(),
        NNCFQuantize(compress_weights=True),
        [
            (TorchCompileOVExport(), LatencyBenchmark()),
            (OpenvinoIRExport(), LatencyBenchmark()),
        ],
    ),
    BenchmarkPipeline(
        # TorchExport(),
        CapturePreAutogradGraphExport(),
        NNCFQuantize(compress_weights=False),
        [
            (TorchCompileExport(), LatencyBenchmark()),
        ],
    ),
    BenchmarkPipeline(
        # TorchExport(),
        CapturePreAutogradGraphExport(),
        TorchAOQuantize(fold_quantize=False),
        [(TorchCompileOVExport(), LatencyBenchmark()), (TorchCompileExport(), LatencyBenchmark())],
    ),
    # BenchmarkPipeline(
    #    # TorchExport(),
    #    CapturePreAutogradGraphExport(),
    #    TorchAOQuantize(fold_quantize=True),
    #    [(TorchCompileExport(), LatencyBenchmark())],
    # ),
    BenchmarkPipeline(
        OpenvinoIRExport(),
        # CapturePreAutogradGraphExport(),
        NNCFQuantize(compress_weights=False),
        [(NoExport(), LatencyBenchmark())],
    ),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Target model name", type=str, default="all")
    parser.add_argument("--file_name", help="Output csv file_name", type=str, default="result.csv")

    args = parser.parse_args()

    target_models = []
    if args.model == "all":
        for model_name in MODEL_SCOPE:
            target_models.append(model_name)
    else:
        target_models.append(args.model)

    keys, values = [], []
    for model_name in target_models:
        print("---------------------------------------------------")
        print(f"name: {model_name}")
        try:
            model_config = MODEL_SCOPE[model_name]
            save_dir = Path(__file__).parent.resolve() / model_name
            save_dir.mkdir(exist_ok=True)
            for pipeline in PIPELINES:
                keys_, values_ = pipeline.run(model_name, model_config, save_dir)
                keys.extend(keys_)
                values.extend(values_)
        except Exception as e:
            print(f"FAILS TO CHECK PERFORMANCE FOR {model_name} MODEL:")
            err_msg = str(e)
            print(err_msg)
            traceback.print_exc()

    index = pd.MultiIndex.from_tuples(
        keys,
        names=["model", "export before int8", "compression", "export after int8", "benchmark key"],
    )
    df = pd.DataFrame(values, index=index)

    print(df)
    df.to_csv(args.file_name)


if __name__ == "__main__":
    main()
