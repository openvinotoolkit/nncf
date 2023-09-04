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
from nncf.common.strip import strip
from nncf.config import NNCFConfig
from nncf.data import Dataset
from nncf.parameters import DropType
from nncf.parameters import ModelType
from nncf.parameters import TargetDevice
from nncf.quantization import QuantizationPreset
from nncf.quantization import compress_weights
from nncf.quantization import quantize
from nncf.quantization import quantize_with_accuracy_control
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from nncf.scopes import IgnoredScope
from nncf.version import __version__

_SUPPORTED_FRAMEWORKS = ["torch", "tensorflow", "onnx", "openvino"]


from importlib.util import find_spec as _find_spec  # pylint:disable=wrong-import-position
from pathlib import Path as _Path  # pylint:disable=wrong-import-position

_AVAILABLE_FRAMEWORKS = {}

for fw_name in _SUPPORTED_FRAMEWORKS:
    spec = _find_spec(fw_name)
    framework_present = False
    if spec is not None and spec.origin is not None:
        origin_path = _Path(spec.origin)
        here = _Path(__file__)
        if origin_path not in here.parents:
            # if the framework is not present, spec may still be not None because
            # it found our nncf.*backend_name* subpackage, and spec.origin will point to a folder in NNCF code
            framework_present = True
    _AVAILABLE_FRAMEWORKS[fw_name] = framework_present

if not any(_AVAILABLE_FRAMEWORKS.values()):
    nncf_logger.error(
        "Neither PyTorch, TensorFlow, ONNX or OpenVINO Python packages have been found in your Python "
        "environment.\n"
        "Please install one of the supported frameworks above in order to use NNCF on top of it.\n"
        "See the installation guide at https://github.com/openvinotoolkit/nncf#installation-guide for help."
    )
else:
    nncf_logger.info(
        f"NNCF initialized successfully. Supported frameworks detected: "
        f"{', '.join([name for name, loaded in _AVAILABLE_FRAMEWORKS.items() if loaded])}"
    )
