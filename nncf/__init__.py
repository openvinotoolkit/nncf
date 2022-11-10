"""
 Copyright (c) 2019-2022 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from nncf.version import __version__

from nncf.config import NNCFConfig
from nncf.data import Dataset
from nncf.parameters import IgnoredScope
from nncf.parameters import ModelType
from nncf.parameters import TargetDevice
from nncf.quantization import QuantizationPreset
from nncf.quantization import quantize

try:
    import torch
except ImportError:
    torch = None

try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    import onnx
except ImportError:
    onnx = None

try:
    import openvino.runtime as ov_runtime
    import openvino.tools.pot as ov_pot
except ImportError:
    ov_runtime = None
    ov_pot = None

if torch is None and tf is None and onnx is None and ov_runtime is None and ov_pot is None:
    import warnings
    warnings.warn("None of PyTorch, TensorFlow, ONNX, OpenVINO have been found. Please, install one of the frameworks")
