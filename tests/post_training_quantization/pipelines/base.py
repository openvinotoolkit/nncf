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
from enum import Enum
from pathlib import Path

import onnx
import openvino.runtime as ov
from memory_profiler import memory_usage
from optimum.intel import OVQuantizer

import nncf
from nncf import TargetDevice
from nncf.experimental.torch.quantization.quantize_model import quantize_impl as pt_impl_experimental


class BackendType(Enum):
    # FP32 = "FP32"
    LEGACY_TORCH = "Legacy_Torch"
    TORCH = "Torch"
    ONNX = "ONNX"
    OV = "OV"
    POT = "POT"
    OPTIMUM = "OPTIMUM"


ALL_NNCF_PTQ_BACKENDS = [BackendType.TORCH, BackendType.ONNX, BackendType.OV]
PT_BACKENDS = [BackendType.TORCH, BackendType.LEGACY_TORCH]
OV_BACKENDS = [BackendType.OV, BackendType.POT]


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
        num_samples: int,
        params: dict,
        output_dir: Path,
        cache_dir: Path,
    ) -> None:
        self.reported_name: str = reported_name
        self.model_id: str = model_id
        self.backend: BackendType = backend
        self.ptq_params: dict = ptq_params
        self.num_samples: int = num_samples
        self.params: dict = params
        self.output_dir: Path = output_dir
        self.cache_dir: Path = cache_dir

        self.output_model_dir = output_dir / self.reported_name / self.backend.value
        self.model_hf = None
        self.model = None
        self.times = {}
        self.mem_profile = {}

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
        print("Preparing...")
        start_time = time.perf_counter()
        self.prepare_preprocessor()
        self.prepare_model()
        self.prepare_calibration_dataset()

        self.times["prepare"] = time.perf_counter() - start_time

    def _quantize(self):
        if self.backend == BackendType.OPTIMUM:
            quantizer = OVQuantizer.from_pretrained(self.model_hf)
            quantizer.quantize(calibration_dataset=self.calibration_dataset, save_directory=self.output_model_dir)

            model_xml_path = self.output_model_dir / "openvino_model.xml"
            core = ov.Core()
            self.quantized_model = core.read_model(model_xml_path)
        else:
            quantize_fn = nncf.quantize
            if self.backend == BackendType.TORCH:
                # Use experimental torch api
                quantize_fn = pt_impl_experimental

            self.quantized_model = quantize_fn(
                model=self.model,
                target_device=TargetDevice.CPU,
                subset_size=1,
                fast_bias_correction=True,
                calibration_dataset=self.calibration_dataset,
                **self.ptq_params
            )

    def quantize(self) -> None:
        print("Quantization...")
        start_time = time.perf_counter()
        self.mem_profile["quantize"] = memory_usage(self._quantize, interval=1, include_children=True, max_usage=True)
        self.times["quantize"] = time.perf_counter() - start_time

    @abstractmethod
    def _validate(self) -> None:
        pass

    def validate(self):
        print("Validate...")
        start_time = time.perf_counter()
        self.mem_profile["validate"] = memory_usage(self._validate, interval=1, include_children=True, max_usage=True)
        self.times["validate"] = time.perf_counter() - start_time

    def get_result_dict(self):
        result = {
            "reported_name": self.reported_name,
            "model_id": self.model_id,
            "backend": self.backend.value,
            "metric name": "todo",
            "metric value": None,
            "memory used in quantitation": self.mem_profile.get("quantize"),
            "quantization time": self.times.get("quantize"),
        }
        return result


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
