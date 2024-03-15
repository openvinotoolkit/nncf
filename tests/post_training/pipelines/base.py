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
import datetime as dt
import os
import re
import time
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import onnx
import openvino as ov
import torch
from memory_profiler import memory_usage
from optimum.intel import OVQuantizer

import nncf
from nncf import TargetDevice
from tests.shared.command import Command

DEFAULT_VAL_THREADS = 4


class BackendType(Enum):
    FP32 = "FP32"
    TORCH = "TORCH"
    CUDA_TORCH = "CUDA_TORCH"
    ONNX = "ONNX"
    OV = "OV"
    OPTIMUM = "OPTIMUM"


NNCF_PTQ_BACKENDS = [BackendType.TORCH, BackendType.CUDA_TORCH, BackendType.ONNX, BackendType.OV]
ALL_PTQ_BACKENDS = NNCF_PTQ_BACKENDS
PT_BACKENDS = [BackendType.TORCH, BackendType.CUDA_TORCH]
OV_BACKENDS = [BackendType.OV, BackendType.OPTIMUM]

LIMIT_LENGTH_OF_STATUS = 120


class StatsFromOutput:
    """
    Contains statistics that are parsed from the stdout.
    """

    def get_stats(self) -> Dict[str, str]:
        """
        Returns statistics collected from the stdout. Usually it parses execution time from the log of the progress bar.
        """
        return {}

    def fill(self, stdout: str) -> None:
        """
        Parses standard output from the post-training conformance tests and collect statistics, for instance, the
        duration of different algorithm's stages.

        :param stdout: string containing the standard output
        """


@dataclass
class PTQTimeStats(StatsFromOutput):
    """
    Contains statistics that are parsed from the stdout of PTQ tests.
    """

    time_stat_collection: Optional[str] = None
    time_bias_correction: Optional[str] = None
    time_validation: Optional[str] = None

    STAT_NAMES = ["Stat. collection time", "Bias correction time", "Validation time"]

    def fill(self, stdout: str):
        time_stat_collection_ = None
        time_bias_correction_ = None
        for line in stdout.splitlines():
            match = re.search(r"Statistics\scollection.*•\s(.*)\s•.*", line)
            if match:
                if time_stat_collection_ is None:
                    time_stat_collection_ = dt.datetime.strptime(match.group(1), "%H:%M:%S")
                else:
                    time = dt.datetime.strptime(match.group(1), "%H:%M:%S")
                    time_stat_collection_ += dt.timedelta(hours=time.hour, minutes=time.minute, seconds=time.second)
                continue

            match = re.search(r"Applying.*correction.*\/(\d+)\s•\s(.*)\s•.*", line)
            if match:
                if time_bias_correction_ is None:
                    time_bias_correction_ = dt.datetime.strptime(match.group(2), "%H:%M:%S")
                else:
                    time_bias_correction_ += dt.datetime.strptime(match.group(2), "%H:%M:%S")
                continue

            match = re.search(r"Validation.*\/\d+\s•\s(.*)\s•.*", line)
            if match:
                self.time_validation = match.group(1)
                continue

        if time_stat_collection_:
            self.time_stat_collection = time_stat_collection_.strftime("%H:%M:%S")
        if time_bias_correction_:
            self.time_bias_correction = time_bias_correction_.strftime("%H:%M:%S")

    def get_stats(self):
        VARS = [self.time_stat_collection, self.time_bias_correction, self.time_validation]
        return dict(zip(self.STAT_NAMES, VARS))


@dataclass
class RunInfo:
    """
    Containing data about compression of the model.
    """

    model: Optional[str] = None
    backend: Optional[BackendType] = None
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    metric_diff: Optional[float] = None
    num_fq_nodes: Optional[float] = None
    compression_memory_usage: Optional[int] = None
    status: Optional[str] = None
    fps: Optional[float] = None
    time_total: Optional[float] = None
    time_compression: Optional[float] = None
    stats_from_output = StatsFromOutput()

    @staticmethod
    def format_time(time_elapsed):
        if time_elapsed is None:
            return None
        return str(timedelta(seconds=int(time_elapsed)))

    @staticmethod
    def format_memory_usage(memory):
        if memory is None:
            return None
        return int(memory)

    def get_result_dict(self):
        return {
            "Model": self.model,
            "Backend": self.backend.value if self.backend else None,
            "Metric name": self.metric_name,
            "Metric value": self.metric_value,
            "Metric diff": self.metric_diff,
            "Num FQ": self.num_fq_nodes,
            "RAM MiB": self.format_memory_usage(self.compression_memory_usage),
            "Compr. time": self.format_time(self.time_compression),
            **self.stats_from_output.get_stats(),
            "Total time": self.format_time(self.time_total),
            "FPS": self.fps,
            "Status": self.status[:LIMIT_LENGTH_OF_STATUS] if self.status is not None else None,
        }


class BaseTestPipeline(ABC):
    """
    Base class to test compression algorithms.
    """

    def __init__(
        self,
        reported_name: str,
        model_id: str,
        backend: BackendType,
        compression_params: dict,
        output_dir: Path,
        data_dir: Path,
        reference_data: dict,
        no_eval: bool,
        run_benchmark_app: bool,
        params: dict = None,
    ) -> None:
        self.reported_name = reported_name
        self.model_id = model_id
        self.backend = backend
        self.compression_params = compression_params
        self.output_dir = output_dir
        self.data_dir = data_dir
        self.reference_data = reference_data
        self.params = params or {}
        self.no_eval = no_eval
        self.run_benchmark_app = run_benchmark_app
        self.output_model_dir: Path = self.output_dir / self.reported_name / self.backend.value
        self.output_model_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = f"{self.reported_name}_{self.backend.value}"
        self.fp32_model_name = self.model_id.replace("/", "__")
        self.fp32_model_dir: Path = self.output_dir / "fp32_models" / self.fp32_model_name
        self.fp32_model_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.model_hf = None
        self.calibration_dataset = None
        self.dummy_tensor = None
        self.input_size = None

        self.run_info = RunInfo(model=reported_name, backend=self.backend)

    @abstractmethod
    def prepare_preprocessor(self) -> None:
        """Prepare preprocessor for the target model."""

    @abstractmethod
    def prepare_calibration_dataset(self) -> None:
        """Prepare calibration dataset for the target model."""

    @abstractmethod
    def prepare_model(self) -> None:
        """Prepare model."""

    @abstractmethod
    def cleanup_cache(self):
        """Helper for removing cached model representation."""

    @abstractmethod
    def collect_data_from_stdout(self, stdout: str):
        """Collects statistics from the standard output."""

    @abstractmethod
    def compress(self) -> None:
        """Run compression of the model and collect time and memory usage information."""

    @abstractmethod
    def save_compressed_model(self) -> None:
        """Save compressed model to IR."""

    @abstractmethod
    def get_num_compressed(self) -> None:
        """Get number of the compressed nodes in the compressed IR."""

    @abstractmethod
    def run_bench(self) -> None:
        """Run a benchmark to collect performance statistics."""

    @abstractmethod
    def _validate(self) -> None:
        """Validate IR."""

    def prepare(self):
        """
        Preparing model and calibration dataset for compression.
        """
        print("Preparing...")
        self.prepare_model()
        if self.model is None:
            raise nncf.ValidationError("self.model is None")
        self.prepare_preprocessor()
        self.prepare_calibration_dataset()

    def validate(self) -> None:
        """
        Validate and compare result with reference.
        """
        if self.no_eval:
            print("Validation skipped")
            return
        print("Validation...")

        self._validate()

        metric_value = self.run_info.metric_value
        metric_reference = self.reference_data.get("metric_value")
        metric_value_fp32 = self.reference_data.get("metric_value_fp32")

        if metric_value is not None and metric_value_fp32 is not None:
            self.run_info.metric_diff = round(self.run_info.metric_value - self.reference_data["metric_value_fp32"], 5)

        if (
            metric_value is not None
            and metric_reference is not None
            and not np.isclose(metric_value, metric_reference, atol=self.reference_data.get("atol", 0.001))
        ):
            if metric_value < metric_reference:
                status_msg = f"Regression: Metric value is less than reference {metric_value} < {metric_reference}"
                raise ValueError(status_msg)
            if metric_value > metric_reference:
                self.run_info.status = (
                    f"Improvement: Metric value is better than reference {metric_value} > {metric_reference}"
                )

    def run(self) -> None:
        """
        Run full pipeline of compression.
        """
        self.prepare()
        self.compress()
        self.save_compressed_model()
        self.get_num_compressed()
        self.validate()
        self.run_bench()


class PTQTestPipeline(BaseTestPipeline):
    """
    Base class to test post training quantization.
    """

    def _compress(self):
        """
        Quantize self.model
        """
        if self.backend == BackendType.OPTIMUM:
            quantizer = OVQuantizer.from_pretrained(self.model_hf)
            quantizer.quantize(calibration_dataset=self.calibration_dataset, save_directory=self.output_model_dir)
        else:
            self.compressed_model = nncf.quantize(
                model=self.model,
                target_device=TargetDevice.CPU,
                calibration_dataset=self.calibration_dataset,
                **self.compression_params,
            )

    def compress(self) -> None:
        """
        Run quantization of the model and collect time and memory usage information.
        """
        if self.backend == BackendType.FP32:
            # To validate not compressed model
            self.path_compressed_ir = self.fp32_model_dir / "model_fp32.xml"
            return

        print("Quantization...")

        if self.backend in PT_BACKENDS:
            inference_num_threads = os.environ.get("INFERENCE_NUM_THREADS")
            if inference_num_threads is not None:
                torch.set_num_threads(int(inference_num_threads))

        start_time = time.perf_counter()
        self.run_info.compression_memory_usage = memory_usage(self._compress, max_usage=True)
        self.run_info.time_compression = time.perf_counter() - start_time

    def save_compressed_model(self) -> None:
        """
        Save compressed model to IR.
        """
        print("Saving quantized model...")
        if self.backend == BackendType.OPTIMUM:
            self.path_compressed_ir = self.output_model_dir / "openvino_model.xml"
        elif self.backend in PT_BACKENDS:
            ov_model = ov.convert_model(
                self.compressed_model.cpu(), example_input=self.dummy_tensor.cpu(), input=self.input_size
            )
            self.path_compressed_ir = self.output_model_dir / "model.xml"
            ov.serialize(ov_model, self.path_compressed_ir)
        elif self.backend == BackendType.ONNX:
            onnx_path = self.output_model_dir / "model.onnx"
            onnx.save(self.compressed_model, str(onnx_path))
            ov_model = ov.convert_model(onnx_path)
            self.path_compressed_ir = self.output_model_dir / "model.xml"
            ov.serialize(ov_model, self.path_compressed_ir)
        elif self.backend in OV_BACKENDS:
            self.path_compressed_ir = self.output_model_dir / "model.xml"
            ov.serialize(self.compressed_model, str(self.path_compressed_ir))

    def get_num_compressed(self) -> None:
        """
        Get number of the FakeQuantize nodes in the compressed IR.
        """

        ie = ov.Core()
        model = ie.read_model(model=self.path_compressed_ir)

        num_fq = 0
        for node in model.get_ops():
            node_type = node.type_info.name
            if node_type == "FakeQuantize":
                num_fq += 1

        self.run_info.num_fq_nodes = num_fq

    def run_bench(self) -> None:
        """
        Run benchmark_app to collect performance statistics.
        """
        if not self.run_benchmark_app:
            return
        runner = Command(f"benchmark_app -m {self.path_compressed_ir}")
        runner.run(stdout=False)
        cmd_output = " ".join(runner.output)

        match = re.search(r"Throughput\: (.+?) FPS", cmd_output)
        if match is not None:
            fps = match.group(1)
            self.run_info.fps = float(fps)

    def cleanup_cache(self):
        """
        Helper for removing cached model representation.

        After run torch.jit.trace in convert_model, PyTorch does not clear the trace cache automatically.
        """
        torch._C._jit_clear_class_registry()
        torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
        torch.jit._state._clear_class_state()

    def collect_data_from_stdout(self, stdout: str):
        stats = PTQTimeStats()
        stats.fill(stdout)
        self.run_info.stats_from_output = stats
