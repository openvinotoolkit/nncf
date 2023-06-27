# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
import openvino.runtime as ov
import torch
import torch.nn as nn
from memory_profiler import memory_usage
from optimum.intel import OVQuantizer

import nncf
from nncf import TargetDevice
from nncf.experimental.torch.quantization.quantize_model import quantize_impl as pt_impl_experimental
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from tests.shared.command import Command

DEFAULT_VAL_THREADS = 4


class BackendType(Enum):
    FP32 = "FP32"
    LEGACY_TORCH = "Legacy_Torch"
    TORCH = "Torch"
    ONNX = "ONNX"
    OV = "OV"
    POT = "POT"
    OPTIMUM = "OPTIMUM"


ALL_NNCF_PTQ_BACKENDS = [BackendType.LEGACY_TORCH, BackendType.TORCH, BackendType.ONNX, BackendType.OV, BackendType.POT]
PT_BACKENDS = [BackendType.TORCH, BackendType.LEGACY_TORCH]
OV_BACKENDS = [BackendType.OV, BackendType.POT]


@dataclass
class RunInfo:
    """
    Containing data about quantization of the model.
    """

    model: Optional[str] = None
    backend: Optional[BackendType] = None
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    metric_diff: Optional[float] = None
    num_fq_nodes: Optional[float] = None
    quant_memory_usage: Optional[int] = None
    time_full: Optional[float] = None
    time_quantization: Optional[float] = None
    error_message: Optional[str] = None

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
            "Quant. RAM MiB": self.format_memory_usage(self.quant_memory_usage),
            "Quant. time": self.format_time(self.time_quantization),
            "Full time": self.format_time(self.time_full),
            "Error": self.error_message,
        }


def export_to_onnx(model: nn.Module, save_path: str, data_sample: torch.Tensor) -> None:
    """
    Export Torch model to ONNX format.
    """
    torch.onnx.export(model, data_sample, save_path, export_params=True, opset_version=13, do_constant_folding=False)


def export_to_ir(model_path: str, save_path: str, model_name: str) -> None:
    """
    Export ONNX model to OpenVINO format.

    :param model_path: Path to ONNX model.
    :param save_path: Path directory to save OpenVINO IR model.
    :param model_name: Model name.
    """
    runner = Command(f"mo -m {model_path} -o {save_path} -n {model_name}")
    runner.run()


class BaseTestPipeline(ABC):
    """
    Base class to test post training quantization.
    """

    def __init__(
        self,
        reported_name: str,
        model_id: str,
        backend: BackendType,
        ptq_params: dict,
        params: dict,
        output_dir: Path,
        data_dir: Path,
        mode: str,
        reference_data: dict,
    ) -> None:
        self.reported_name = reported_name
        self.model_id = model_id
        self.backend = backend
        self.ptq_params = ptq_params
        self.params = params
        self.output_dir = Path(output_dir)
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.reference_data = reference_data

        self.output_model_dir = self.output_dir / self.reported_name / self.backend.value
        self.output_model_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = f"{self.reported_name}_{self.backend.value}"
        self.model = None

        self.run_info = RunInfo(model=reported_name, backend=self.backend)

        self.post_init()

    def post_init(self):
        pass

    @abstractmethod
    def prepare_preprocessor(self) -> None:
        """Prepare preprocessor for the target model."""
        pass

    @abstractmethod
    def prepare_calibration_dataset(self) -> None:
        """Prepare calibration dataset for the target model."""
        pass

    @abstractmethod
    def prepare_model(self) -> None:
        """Prepare model"""
        pass

    def prepare(self):
        """
        Preparing model and calibration dataset for quantization.
        """
        print("Preparing...")
        self.prepare_model()
        if self.model is None:
            raise RuntimeError("self.model is None")
        self.prepare_preprocessor()
        self.prepare_calibration_dataset()

    def _quantize(self):
        """
        Quantize self.model
        """
        if self.backend == BackendType.OPTIMUM:
            quantizer = OVQuantizer.from_pretrained(self.model_hf)
            quantizer.quantize(calibration_dataset=self.calibration_dataset, save_directory=self.output_model_dir)
        else:
            quantize_fn = nncf.quantize
            if self.backend == BackendType.TORCH:
                # Use experimental torch api
                quantize_fn = pt_impl_experimental
                if "preset" not in self.ptq_params:
                    self.ptq_params["preset"] = nncf.QuantizationPreset.PERFORMANCE
                if "subset_size" not in self.ptq_params:
                    self.ptq_params["subset_size"] = 300

            if self.backend == BackendType.POT:
                self.ptq_params["advanced_parameters"] = AdvancedQuantizationParameters(
                    backend_params={"use_pot": True}
                )

            self.quantized_model = quantize_fn(
                model=self.model,
                target_device=TargetDevice.CPU,
                fast_bias_correction=True,
                calibration_dataset=self.calibration_dataset,
                **self.ptq_params,
            )

    def quantize(self) -> None:
        """
        Run quantization of the model and collect time and memory usage information.
        """
        print("Quantization...")
        start_time = time.perf_counter()
        self.run_info.quant_memory_usage = memory_usage(self._quantize, max_usage=True)
        self.run_info.time_quantization = time.perf_counter() - start_time

    def post_quantize(self) -> None:
        """
        Save quantized model to IR.
        """
        print("Post quantization...")
        if self.backend == BackendType.OPTIMUM:
            self.path_quantized_ir = self.output_model_dir / "openvino_model.xml"
        elif self.backend in PT_BACKENDS:
            onnx_path = self.output_model_dir / "model.onnx"
            export_to_onnx(self.quantized_model, str(onnx_path), self.dummy_tensor)
            export_to_ir(onnx_path, self.output_model_dir, model_name="model")
            self.path_quantized_ir = self.output_model_dir / "model.xml"
        elif self.backend == BackendType.ONNX:
            onnx_path = self.output_model_dir / "model.onnx"
            onnx.save(self.quantized_model, str(onnx_path))
            export_to_ir(onnx_path, str(self.output_model_dir), model_name="model")
            self.path_quantized_ir = self.output_model_dir / "model.xml"
        elif self.backend in OV_BACKENDS:
            self.path_quantized_ir = self.output_model_dir / "model.xml"
            ov.serialize(self.quantized_model, str(self.path_quantized_ir))

    def get_num_fq(self) -> None:
        """
        Get number of the FakeQuantize nodes in the quantized IR.
        """
        num_fq = 0
        with open(self.path_quantized_ir, "r", encoding="UTF-8") as stream:
            for line in stream.readlines():
                if 'type="FakeQuantize"' in line:
                    num_fq += 1
        self.run_info.num_fq_nodes = num_fq

    @abstractmethod
    def _validate(self) -> None:
        pass

    def validate(self) -> None:
        print("Validation...")
        self._validate()

        metric_value = self.run_info.metric_value
        metric_reference = self.reference_data.get("metric_value")
        metric_value_fp32 = self.reference_data.get("metric_value_fp32")

        if metric_value is not None and metric_value_fp32 is not None:
            self.run_info.metric_diff = self.reference_data["metric_value_fp32"] - self.run_info.metric_value

        if metric_value is not None and metric_reference is not None:
            print(f"{np.isclose(metric_value, metric_reference)}")
            if not np.isclose(metric_value, metric_reference):
                raise ValueError(f"Metric value {metric_value} is not close to reference {metric_reference}")

    def run(self) -> None:
        start_time = time.perf_counter()
        self.prepare()
        self.quantize()
        self.post_quantize()
        self.get_num_fq()
        self.validate()
        self.run_info.time_full = time.perf_counter() - start_time

    def get_run_info(self) -> RunInfo:
        return self.run_info


class BaseHFTestPipeline(BaseTestPipeline):
    def prepare_model(self) -> None:
        if self.backend in [BackendType.TORCH, BackendType.LEGACY_TORCH]:
            self.model_hf = self.params["pt_model_class"].from_pretrained(self.model_id)
            self.model = self.model_hf

        if self.backend in [BackendType.OV, BackendType.POT, BackendType.OPTIMUM]:
            self.model_hf = self.params["ov_model_class"].from_pretrained(self.model_id, export=True, compile=False)
            self.model = self.model_hf.model

        if self.backend in [BackendType.ONNX]:
            self.model_hf = self.params["onnx_model_class"].from_pretrained(self.model_id, export=True)
            self.model = onnx.load(self.model_hf.model_path)
