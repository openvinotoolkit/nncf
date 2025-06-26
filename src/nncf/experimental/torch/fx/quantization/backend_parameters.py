# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters


class FXBackendParameters:
    COMPRESS_WEIGHTS = "compress_weights"


def is_weight_compression_needed(advanced_parameters: Optional[AdvancedQuantizationParameters]) -> bool:
    """
    Determines whether weight compression is needed based on the provided
    advanced quantization parameters.

    :param advanced_parameters: Advanced quantization parameters.
    :return: True if weight compression is needed, False otherwise.
    """
    if advanced_parameters is not None and advanced_parameters.backend_params is not None:
        return advanced_parameters.backend_params.get(FXBackendParameters.COMPRESS_WEIGHTS, True)
    return True
