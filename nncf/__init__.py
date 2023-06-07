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
"""
Neural Network Compression Framework (NNCF) for enhanced OpenVINO™ inference.
"""
from nncf.common.logging import nncf_logger
from nncf.common.logging.logger import disable_logging
from nncf.common.logging.logger import set_log_level
from nncf.config import NNCFConfig
from nncf.data import Dataset
from nncf.parameters import DropType
from nncf.parameters import ModelType
from nncf.parameters import TargetDevice
from nncf.quantization import QuantizationPreset
from nncf.quantization import quantize
from nncf.quantization import quantize_with_accuracy_control
from nncf.scopes import IgnoredScope
from nncf.version import __version__

_LOADED_FRAMEWORKS = {"torch": True, "tensorflow": True, "onnx": True, "openvino": True}  # fmt: off

try:
    import torch
except ImportError:
    _LOADED_FRAMEWORKS["torch"] = False

try:
    import tensorflow as tf
except ImportError:
    _LOADED_FRAMEWORKS["tensorflow"] = False

try:
    import onnx
except ImportError:
    _LOADED_FRAMEWORKS["onnx"] = False

try:
    import openvino.runtime as ov_runtime
except ImportError:
    _LOADED_FRAMEWORKS["openvino"] = False


if not any(_LOADED_FRAMEWORKS.values()):
    nncf_logger.error(
        "Neither PyTorch, TensorFlow, ONNX or OpenVINO Python packages have been found in your Python "
        "environment.\n"
        "Please install one of the supported frameworks above in order to use NNCF on top of it.\n"
        "See the installation guide at https://github.com/openvinotoolkit/nncf#installation for help."
    )
else:
    nncf_logger.info(
        f"NNCF initialized successfully. Supported frameworks detected: "
        f"{', '.join([name for name, loaded in _LOADED_FRAMEWORKS.items() if loaded])}"
    )
