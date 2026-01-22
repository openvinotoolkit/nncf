# Copyright (c) 2026 Intel Corporation
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
import gc
import os
import re
import time
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import onnx
import openvino as ov
import torch
from memory_profiler import memory_usage
from optimum.intel import OVQuantizer

import nncf
from nncf import TargetDevice
from tests.cross_fw.shared.command import Command
from tools.memory_monitor import MemoryType
from tools.memory_monitor import MemoryUnit
from tools.memory_monitor import memory_monitor_context

DEFAULT_VAL_THREADS = 4
XFAIL_SUFFIX = "_xfail_reason"


class ErrorReason(Enum):
    METRICS = "metrics"
    NUM_COMPRESSED = "num_compressed"
    EXCEPTION = "exception"


@dataclass
class ErrorReport:
    reason: ErrorReason
    msg: str


class BackendType(Enum):
    FP32 = "FP32"
    TORCH = "TORCH"
    CUDA_TORCH = "CUDA_TORCH"
    FX_TORCH = "FX_TORCH"
    CUDA_FX_TORCH = "CUDA_FX_TORCH"
    OV_QUANTIZER_NNCF = "OV_QUANTIZER_NNCF"
    OV_QUANTIZER_AO = "OV_QUANTIZER_AO"
    ONNX = "ONNX"
    OV = "OV"
    OPTIMUM = "OPTIMUM"


NNCF_PTQ_BACKENDS = [BackendType.TORCH, BackendType.CUDA_TORCH, BackendType.ONNX, BackendType.OV]
ALL_PTQ_BACKENDS = NNCF_PTQ_BACKENDS
PT_BACKENDS = [BackendType.TORCH, BackendType.CUDA_TORCH]
FX_BACKENDS = [
    BackendType.FX_TORCH,
    BackendType.CUDA_FX_TORCH,
    BackendType.OV_QUANTIZER_NNCF,
    BackendType.OV_QUANTIZER_AO,
]
OV_BACKENDS = [BackendType.OV, BackendType.OPTIMUM]

LIMIT_LENGTH_OF_STATUS = 120


class StatsFromOutput:
    """
    Contains statistics that are parsed from the stdout.
    """

    def get_stats(self) -> dict[str, str]:
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
class NumCompressNodes:
    num_int8: Optional[int] = None

    def get_data(self):
        return {"Num int8": self.num_int8}


@dataclass
class PTQNumCompressNodes(NumCompressNodes):
    num_fq_nodes: Optional[int] = None

    def get_data(self):
        data = super().get_data()
        data["Num FQ"] = self.num_fq_nodes
        return data


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
    compression_memory_usage: Optional[int] = None
    compression_memory_usage_rss: Optional[int] = None
    compression_memory_usage_system: Optional[int] = None
    status: Optional[str] = None
    fps: Optional[float] = None
    time_total: Optional[float] = None
    time_compression: Optional[float] = None
    num_compress_nodes: Optional[NumCompressNodes] = None
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

    def get_result_dict(self) -> dict[str, str]:
        """Returns a dictionary with the results of the run."""
        ram_data = {}
        if self.compression_memory_usage_system is None:
            ram_data["RAM MiB"] = self.format_memory_usage(self.compression_memory_usage)
        else:
            ram_data["RAM MiB"] = self.format_memory_usage(self.compression_memory_usage_rss)
            ram_data["RAM MiB System"] = self.format_memory_usage(self.compression_memory_usage_system)

        result = {
            "Model": self.model,
            "Backend": self.backend.value if self.backend else None,
            "Metric name": self.metric_name,
            "Metric value": self.metric_value,
            "Metric diff": self.metric_diff,
            **self.num_compress_nodes.get_data(),
            "Compr. time": self.format_time(self.time_compression),
            **self.stats_from_output.get_stats(),
            "Total time": self.format_time(self.time_total),
            "FPS": self.fps,
            **ram_data,
            "Status": self.status[:LIMIT_LENGTH_OF_STATUS] if self.status is not None else None,
            "Build url": os.environ.get("BUILD_URL", ""),
        }
        return result


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
        batch_size: int = 1,
        memory_monitor: bool = False,
    ) -> None:
        self.reported_name = reported_name
        self.model_id = model_id
        self.backend = backend
        self.compression_params = compression_params
        self.output_dir = output_dir
        self.data_dir = data_dir
        self.reference_data = reference_data
        self.params = params or {}
        self.batch_size = batch_size
        self.memory_monitor = memory_monitor
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

        self.run_info = RunInfo(model=reported_name, backend=self.backend, num_compress_nodes=NumCompressNodes())

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

    def _validate(self) -> None:
        """
        Validates some test criteria.
        returns:
            A list of error reports generated during validation.
        """

    def prepare(self):
        """
        Preparing model and calibration dataset for compression.
        """
        print("Preparing...")
        self.prepare_model()
        if self.model is None:
            msg = "self.model is None"
            raise nncf.ValidationError(msg)
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

    def collect_errors(self) -> list[ErrorReport]:
        """
        Collects errors based on the pipeline's run information.

        :param pipeline: The pipeline object containing run information.
        :return: List of error reports.
        """
        errors = []

        run_info = self.run_info
        reference_data = self.reference_data

        metric_value = run_info.metric_value
        metric_reference = reference_data.get("metric_value")
        metric_value_fp32 = reference_data.get("metric_value_fp32")

        if metric_value is not None and metric_value_fp32 is not None:
            run_info.metric_diff = round(metric_value - metric_value_fp32, 5)

        if metric_value is not None and metric_reference is not None:
            atol = reference_data.get("atol", 0.001)
            if not np.isclose(metric_value, metric_reference, atol=atol):
                status_msg = (
                    f"Regression: Metric value is less than reference {metric_value} < {metric_reference}"
                    if metric_value < metric_reference
                    else f"Improvement: Metric value is better than reference {metric_value} > {metric_reference}"
                )
                errors.append(ErrorReport(ErrorReason.METRICS, status_msg))

        return errors

    def update_status(self, error_reports: list[ErrorReport]) -> list[str]:
        """
        Updates status of the pipeline based on the errors encountered during the run.

        :param pipeline: The pipeline object containing run information.
        :param error_reports: List of errors encountered during the run.
        :return: List of unexpected errors.
        """
        self.run_info.status = ""  # Successful status
        xfails, unexpected_errors = [], []

        for report in error_reports:
            xfail_reason = report.reason.value + XFAIL_SUFFIX
            if _is_error_xfailed(report, xfail_reason, self.reference_data):
                xfails.append(_get_xfail_message(report, xfail_reason, self.reference_data))
            else:
                unexpected_errors.append(report.msg)

        if xfails:
            self.run_info.status = "\n".join(xfails)
        if unexpected_errors:
            self.run_info.status = "\n".join(unexpected_errors)
        return unexpected_errors


class PTQTestPipeline(BaseTestPipeline):
    """
    Base class to test post training quantization.
    """

    def __init__(
        self,
        reported_name,
        model_id,
        backend,
        compression_params,
        output_dir,
        data_dir,
        reference_data,
        no_eval,
        run_benchmark_app,
        params=None,
        batch_size=1,
        memory_monitor=False,
    ):
        super().__init__(
            reported_name,
            model_id,
            backend,
            compression_params,
            output_dir,
            data_dir,
            reference_data,
            no_eval,
            run_benchmark_app,
            params,
            batch_size,
            memory_monitor,
        )
        self.run_info = RunInfo(model=reported_name, backend=self.backend, num_compress_nodes=PTQNumCompressNodes())

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
        if self.memory_monitor:
            self.run_info.compression_memory_usage_rss = -1
            self.run_info.compression_memory_usage_system = -1
            gc.collect()
            with memory_monitor_context(
                interval=0.1,
                memory_unit=MemoryUnit.MiB,
                return_max_value=True,
                save_dir=self.output_model_dir / "ptq_memory_logs",
            ) as mmc:
                self._compress()
            self.run_info.compression_memory_usage_rss = mmc.memory_data[MemoryType.RSS]
            self.run_info.compression_memory_usage_system = mmc.memory_data[MemoryType.SYSTEM]
        else:
            self.run_info.compression_memory_usage = memory_usage(self._compress, max_usage=True)
        self.run_info.time_compression = time.perf_counter() - start_time

    def save_compressed_model(self) -> None:
        """
        Save compressed model to IR.
        """
        print("Saving quantized model...")
        self.path_compressed_ir = self.output_model_dir / "model.xml"
        if self.backend == BackendType.OPTIMUM:
            self.path_compressed_ir = self.output_model_dir / "openvino_model.xml"
        elif self.backend in PT_BACKENDS:
            ov_model = ov.convert_model(
                self.compressed_model.cpu(), example_input=self.dummy_tensor.cpu(), input=self.input_size
            )
            ov.serialize(ov_model, self.path_compressed_ir)
        elif self.backend in FX_BACKENDS:
            exported_model = torch.export.export(self.compressed_model.cpu(), (self.dummy_tensor.cpu(),), strict=True)
            # Torch export is used to save the model because ov.convert_model does not fully claim support for
            # Converting ExportedProgram
            torch.export.save(exported_model, self.output_model_dir / "model.pt2")
            # torch.compile is used to cache the OV model because this is the default user journey with Torch FX
            # backend. This is also neccesary because PT FE translation in OV for convert_model
            # and the translations used for torch.compile sometimes differ. This method can help
            # ensure that the correct OV graph is being verified.
            mod = torch.compile(
                exported_model.module(),
                backend="openvino",
                options={"aot_autograd": True, "model_caching": True, "cache_dir": str(self.output_model_dir)},
            )
            mod(self.dummy_tensor)

            # Get the OV *.xml files in torch compile cache directory
            cached_ov_model_files = list(Path(self.output_model_dir / "model").glob("*.xml"))
            if len(cached_ov_model_files) > 1:
                msg = "Graph break encountered in torch compile!"
                raise nncf.InternalError(msg)
            if len(cached_ov_model_files) == 0:
                msg = "Openvino Model Files Not Found!"
                raise FileNotFoundError(msg)
            self.path_compressed_ir = cached_ov_model_files[0]

            if self.backend == BackendType.CUDA_FX_TORCH:
                self.model = self.model.cuda()
                self.dummy_tensor = self.dummy_tensor.cuda()

        elif self.backend == BackendType.ONNX:
            onnx_path = self.output_model_dir / "model.onnx"
            onnx.save(self.compressed_model, str(onnx_path))
            ov_model = ov.convert_model(onnx_path)
            ov.serialize(ov_model, self.path_compressed_ir)
        elif self.backend in OV_BACKENDS:
            from openvino._offline_transformations import apply_moc_transformations

            apply_moc_transformations(self.compressed_model, cf=True)
            ov.serialize(self.compressed_model, str(self.path_compressed_ir))

    def run_bench(self) -> None:
        """
        Run benchmark_app to collect performance statistics.
        """
        if not self.run_benchmark_app:
            return

        try:
            runner = Command(f"benchmark_app -m {self.path_compressed_ir}")
            runner.run(stdout=False)
            cmd_output = " ".join(runner.output)

            match = re.search(r"Throughput\: (.+?) FPS", cmd_output)
            if match is not None:
                fps = match.group(1)
                self.run_info.fps = float(fps)
        except Exception as e:
            print(e)

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

    def get_num_compressed(self) -> None:
        ie = ov.Core()
        model = ie.read_model(model=self.path_compressed_ir)
        num_fq, _, num_int8 = get_num_fq_int4_int8(model)
        self.run_info.num_compress_nodes.num_int8 = num_int8
        self.run_info.num_compress_nodes.num_fq_nodes = num_fq


def get_num_fq_int4_int8(model: ov.Model) -> tuple[int, int, int]:
    num_fq = 0
    num_int8 = 0
    num_int4 = 0
    for node in model.get_ops():
        node_type = node.type_info.name
        if node_type == "FakeQuantize":
            num_fq += 1

        for i in range(node.get_output_size()):
            if node.get_output_element_type(i).get_type_name() in ["i8", "u8"]:
                num_int8 += 1
            if node.get_output_element_type(i).get_type_name() in ["i4", "u4", "nf4"]:
                num_int4 += 1

    return num_fq, num_int4, num_int8


def _are_exceptions_matched(report: ErrorReport, reference_exception: dict[str, str]) -> bool:
    return (
        reference_exception["error_message"] == report.msg.split(" | ")[1]
        and reference_exception["type"] == report.msg.split(" | ")[0]
    )


def _is_error_xfailed(report: ErrorReport, xfail_reason: str, reference_data: dict[str, dict[str, str]]) -> bool:
    if xfail_reason not in reference_data:
        return False

    if report.reason == ErrorReason.EXCEPTION:
        return _are_exceptions_matched(report, reference_data[xfail_reason])
    return True


def _get_xfail_message(report: ErrorReport, xfail_reason: str, reference_data: dict[str, dict[str, str]]) -> str:
    if report.reason == ErrorReason.EXCEPTION:
        return f"XFAIL: {reference_data[xfail_reason]['message']} - {report.msg}"
    return f"XFAIL: {xfail_reason} - {report.msg}"
