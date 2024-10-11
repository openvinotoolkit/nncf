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
from torch._export import capture_pre_autograd_graph
from torch.fx.passes.graph_drawer import FxGraphDrawer
from torch.jit import TracerWarning

import nncf
from nncf.common.factory import NNCFGraphFactory
from nncf.torch.dynamic_graph.patch_pytorch import disable_patching
from tests.torch.fx.performance_check.model_scope import MODEL_SCOPE

warnings.filterwarnings("ignore", category=TracerWarning)
warnings.filterwarnings("ignore", category=UserWarning)

VISUALIZE_FX_INT8_GRAPH = False


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


def process_model_ov(model_name: str):
    result = {"name": model_name}
    model_config = MODEL_SCOPE[model_name]
    pt_model = model_config.model_builder.build()
    example_inputs = model_config.model_builder.get_example_inputs()
    export_inputs = example_inputs[0] if isinstance(example_inputs[0], tuple) else example_inputs
    input_sizes = model_config.model_builder.get_input_sizes()
    save_dir = Path(__file__).parent.resolve() / model_name / "OV"
    save_dir.mkdir(exist_ok=True, parents=True)

    try:
        with disable_patching():
            with torch.no_grad():
                ex_model = torch.export.export(pt_model, export_inputs)
                ov_model = ov.convert_model(ex_model, example_input=example_inputs[0], input=input_sizes)
                ov_model_path = save_dir / "openvino_model.xml"
                ov.serialize(ov_model, ov_model_path)
    except Exception as e:
        print("FAILS TO EXPORT FP32 MODEL TO OPENVINO:")
        err_msg = str(e)
        print(err_msg)
        traceback.print_exc()
        return result

    quant_ov_model = nncf.quantize(
        ov_model,
        nncf.Dataset(example_inputs),
        **model_config.quantization_params,
    )
    ov_int8_model_path = save_dir / "openvino_int8_model.xml"
    ov.serialize(quant_ov_model, ov_int8_model_path)
    latency_int8_ov = measure_time_ov(quant_ov_model, example_inputs, model_config.num_iters)
    fps_int8_ov = benchmark_performance(ov_int8_model_path, input_sizes)

    result["ov_int8_ov_latency"] = latency_int8_ov
    result["ov_int8_ov_benchmark_fps"] = fps_int8_ov
    print(f"ov int8 ov model latency: {latency_int8_ov}")
    print(f"ov int8 ov model benchmark fps: {fps_int8_ov}")
    pd.DataFrame([result]).to_csv(save_dir / "result_ov.csv")
    return result


def process_model(model_name: str):
    result = {"name": model_name}
    model_config = MODEL_SCOPE[model_name]
    pt_model = model_config.model_builder.build()
    example_inputs = model_config.model_builder.get_example_inputs()
    export_inputs = example_inputs[0] if isinstance(example_inputs[0], tuple) else example_inputs
    input_sizes = model_config.model_builder.get_input_sizes()
    save_dir = Path(__file__).parent.resolve() / model_name
    save_dir.mkdir(exist_ok=True)

    with disable_patching():
        latency_fp32 = measure_time(torch.compile(pt_model, backend="openvino"), export_inputs, model_config.num_iters)
    result["fp32_compile_latency"] = latency_fp32
    print(f"fp32 compiled model latency: {latency_fp32}")

    try:
        with disable_patching():
            with torch.no_grad():
                ex_model = torch.export.export(pt_model, export_inputs)
                ov_model = ov.convert_model(ex_model, example_input=example_inputs[0], input=input_sizes)
                ov_model_path = save_dir / "openvino_model.xml"
                ov.serialize(ov_model, ov_model_path)
        latency_fp32_ov = measure_time_ov(ov_model, example_inputs, model_config.num_iters)
        fps_fp32_ov = benchmark_performance(ov_model_path, input_sizes)
    except Exception as e:
        print("FAILS TO EXPORT FP32 MODEL TO OPENVINO:")
        err_msg = str(e)
        print(err_msg)
        traceback.print_exc()
        latency_fp32_ov = -1
        fps_fp32_ov = -1

    result["fp32_ov_latency"] = latency_fp32_ov
    result["fp32_ov_benchmark_fps"] = fps_fp32_ov
    print(f"fp32 ov model latency: {latency_fp32_ov}")
    print(f"fp32 ov model benchmark fps: {fps_fp32_ov}")

    with disable_patching():
        with torch.no_grad():
            exported_model = capture_pre_autograd_graph(pt_model, export_inputs)

    with disable_patching():
        with torch.no_grad():
            quant_fx_model = nncf.quantize(
                exported_model,
                nncf.Dataset(example_inputs),
                **model_config.quantization_params,
            )

    int8_graph_visualization_path = str(save_dir / "int8_nncf_graph.dot")
    NNCFGraphFactory.create(quant_fx_model).visualize_graph(int8_graph_visualization_path)
    print(f"NNCFGraph visualization of int8 model is saved to {int8_graph_visualization_path}")

    int8_code_path = str(save_dir / "int8_code.py")
    with open(int8_code_path, "w") as f:
        f.write(quant_fx_model.code)
    print(f"int8 FX code is saved to {int8_code_path}")

    if VISUALIZE_FX_INT8_GRAPH:
        int8_model_visualization_path = str(save_dir / "int8_fx_graph.svg")
        g = FxGraphDrawer(quant_fx_model, int8_model_visualization_path)
        g.get_dot_graph().write_svg(int8_model_visualization_path)
        print(f"Visualization of int8 model is saved to {int8_model_visualization_path}")

    quant_fx_model = torch.compile(quant_fx_model, backend="openvino")

    with disable_patching():
        latency_int8 = measure_time(quant_fx_model, export_inputs, model_config.num_iters)
    result["int8_compiled_latency"] = latency_int8
    print(f"int8 compiled model latency: {latency_int8}")

    try:
        with disable_patching():
            with torch.no_grad():
                ex_int8_model = torch.export.export(quant_fx_model, export_inputs)
                ov_int8_model = ov.convert_model(ex_int8_model, example_input=example_inputs[0], input=input_sizes)
                ov_int8_model_path = save_dir / "openvino_model_int8.xml"
                ov.serialize(ov_int8_model, ov_int8_model_path)

        latency_int8_ov = measure_time_ov(ov_int8_model, export_inputs, model_config.num_iters)
        fps_int8_ov = benchmark_performance(ov_int8_model_path, input_sizes)
    except Exception as e:
        print("FAILS TO EXPORT INT8 MODEL TO OPENVINO:")
        err_msg = str(e)
        print(err_msg)
        traceback.print_exc()
        latency_int8_ov = -1
        fps_int8_ov = -1

    result["fx_int8_ov_latency"] = latency_int8_ov
    result["fx_int8_ov_benchmark_fps"] = fps_int8_ov
    print(f"fx int8 ov model latency: {latency_int8_ov}")
    print(f"fx int8 ov model benchmark fps: {fps_int8_ov}")
    print("*" * 100)
    print(f"Torch compile latency speed up: {latency_fp32 / latency_int8}")
    print(f"Torch export + openvino latenyc speed up: {latency_fp32_ov / latency_int8_ov}")
    print(f"Openvino FPS benchmark speed up: {fps_int8_ov / fps_fp32_ov}")
    print("*" * 100)

    result["compile_latency_diff_speedup"] = latency_fp32 / latency_int8
    result["ov_latency_diff_speedup"] = latency_fp32_ov / latency_int8_ov
    result["ov_benchmark_fps_speedup"] = fps_int8_ov / fps_fp32_ov
    pd.DataFrame([result]).to_csv(save_dir / "result_fx.csv")
    del result["compile_latency_diff_speedup"]
    del result["ov_latency_diff_speedup"]
    del result["ov_benchmark_fps_speedup"]
    return result


def process_model_native_to_ov(model_name: str):
    result = {"name": model_name}
    model_config = MODEL_SCOPE[model_name]
    pt_model = model_config.model_builder.build()
    example_inputs = model_config.model_builder.get_example_inputs()
    export_inputs = example_inputs[0] if isinstance(example_inputs[0], tuple) else example_inputs
    save_dir = Path(__file__).parent.resolve() / model_name
    save_dir.mkdir(exist_ok=True)

    # Check native performance
    with disable_patching():
        latency_native = measure_time(pt_model, example_inputs, model_config.num_iters)
    result["pt_latency"] = latency_native
    print(f"pt latency: {latency_native}")

    # Check native compiled performance
    with disable_patching():
        latency_compiled_native = measure_time(torch.compile(pt_model), example_inputs, model_config.num_iters)
    result["pt_latency_compiled"] = latency_compiled_native
    print(f"pt latency compiled {latency_compiled_native}")

    # Check openvino compiled performance
    with disable_patching():
        latency_compiled_ov = measure_time(
            torch.compile(pt_model, backend="openvino"), example_inputs, model_config.num_iters
        )
    result["ov_latency_compiled"] = latency_compiled_ov
    print(f"ov_latency_compiled: {latency_compiled_ov}")

    # Check FX INT8 with openvino compiled performance
    with disable_patching():
        with torch.no_grad():
            exported_model = capture_pre_autograd_graph(pt_model, export_inputs)

    with disable_patching():
        with torch.no_grad():
            quant_fx_model = nncf.quantize(
                exported_model,
                nncf.Dataset(example_inputs),
                **model_config.quantization_params,
            )

    with disable_patching():
        latency_compiled_ov = measure_time(
            torch.compile(quant_fx_model, backend="openvino"), example_inputs, model_config.num_iters
        )
    result["int8_ov_latency_compiled"] = latency_compiled_ov
    print(f"int8_ov_latency_compiled: {latency_compiled_ov}")
    return result


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

    results_list = []
    for model_name in target_models:
        print("---------------------------------------------------")
        print(f"name: {model_name}")
        try:
            results_list.append({**process_model_native_to_ov(model_name)})
        except Exception as e:
            print(f"FAILS TO CHECK PERFORMANCE FOR {model_name} MODEL:")
            err_msg = str(e)
            print(err_msg)
            traceback.print_exc()

    df = pd.DataFrame(results_list)
    print(df)
    df.to_csv(args.file_name)


if __name__ == "__main__":
    main()
