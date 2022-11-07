"""
 Copyright (c) 2022 Intel Corporation
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

from typing import Optional

from nncf.api.compression import ModelType
from nncf.common.quantization.structs import QuantizationPreset
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.data import Dataset
from nncf.parameters import IgnoredScope
from nncf.parameters import TargetDevice


def quantize(model: ModelType,
             calibration_dataset: Dataset,
             preset: QuantizationPreset = QuantizationPreset.PERFORMANCE,
             target_device: TargetDevice = TargetDevice.ANY,
             subset_size: int = 300,
             fast_error_correction: bool = True,
             model_type: Optional[str] = None,
             ignored_scope: Optional[IgnoredScope] = None) -> ModelType:
    """
    Applies post-training quantization algorithm to provided model.

    :param model: A model to be quantized.
    :param calibration_dataset: A representative dataset for the
        calibration process.
    :param preset: A preset that controls the quantization mode
        (symmetric and asymmetric). It can take the following values:
        - `performance`: Symmetric quantization of weights and activations.
        - `mixed`: Symmetric quantization of weights and asymmetric
          quantization of activations.
    :param target_device: A target device the specificity of which will be taken
        into account while compressing in order to obtain the best performance
        for this type of device.
    :param subset_size: Size of a subset to calculate activations
        statistics used for quantization.
    :param fast_error_correction: Setting this option to `False` enables a different
        bias correction method which is more accurate, in general, and takes
        more time but requires less memory.
    :param model_type: Model type is needed to specify additional patterns
        in the model. Supported only `transformer` now.
    :param ignored_scope: An ignored scope that defined the list of model control
        flow graph nodes to be ignored during quantization.
    :return: The quantized model.
    """
    backend = get_backend(model)
    if backend == BackendType.OPENVINO:
        from nncf.openvino.quantization.helpers import quantize_impl
        return quantize_impl(model, calibration_dataset, preset, target_device, subset_size,
                             fast_error_correction, model_type, ignored_scope)

    raise RuntimeError(f'Unsupported type of backend: {backend}')
