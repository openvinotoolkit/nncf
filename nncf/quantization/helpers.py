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

from typing import Callable
from typing import Optional

from nncf.api.compression import ModelType
from nncf.common.utils.backend import infer_backend_from_model
from nncf.common.utils.backend import BackendType
from nncf.data import DataLoader


def quantize(model: ModelType,
             calibration_dataset: DataLoader,
             preset: str = 'performance',
             target_device: str = 'ANY',
             subset_size: int = 300,
             fast_error_correction: bool = True,
             model_type: Optional[str] = None) -> ModelType:
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
    :param target_device: A target device the specificity of which will
        be taken into account during optimization It can the take
        following values: `ANY`, `CPU`, `GPU`, `GNA`.
    :param subset_size: Size of a subset to calculate activations
        statistics used for quantization.
    :param fast_error_correction: Setting this option to `False` enables a different
        bias correction method which is more accurate, in general, and takes
        more time but requires less memory.
    :param model_type: Model type is needed to specify additional patterns
        in the model. Supported only `transformer` now.
    :return: The quantized model.
    """
    backend = infer_backend_from_model(model)
    if backend == BackendType.OPENVINO:
        from nncf.openvino.quantization.helpers import quantize_impl
        return quantize_impl(model, calibration_dataset, preset, target_device, subset_size,
                             fast_error_correction, model_type)

    raise RuntimeError(f'Unsupported type of backend: {backend}')
