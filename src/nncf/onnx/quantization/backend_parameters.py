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

from typing import Optional, Union

from nncf.quantization.advanced_parameters import AdvancedCompressionParameters
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters


class BackendParameters:
    """
    :param EXTERNAL_DATA_DIR: An absolute path to the directory where the external data
        files are stored. All external data files must be located in the same folder.
    :param COMPRESS_WEIGHTS: TODO
    """

    COMPRESS_WEIGHTS = "compress_weights"
    EXTERNAL_DATA_DIR = "external_data_dir"


def is_weight_compression_needed(advanced_parameters: Optional[AdvancedQuantizationParameters]) -> bool:
    """
    Determines whether weight compression is needed based on the provided
    advanced quantization parameters.

    :param advanced_parameters: Advanced quantization parameters.
    :return: True if weight compression is needed, False otherwise.
    """
    if advanced_parameters is not None and advanced_parameters.backend_params is not None:
        return advanced_parameters.backend_params.get(BackendParameters.COMPRESS_WEIGHTS, True)
    return True


def get_external_data_dir(
    advanced_parameters: Optional[Union[AdvancedQuantizationParameters, AdvancedCompressionParameters]],
) -> Optional[str]:
    """
    Returns the value associated with the `BackendParameters.EXTERNAL_DATA_DIR` key from the backend parameters.

    :param advanced_parameters: Advanced parameters that may contain backend parameters.
    :return: A string representing the external data directory if found; otherwise, `None`.
    """
    if advanced_parameters is not None and advanced_parameters.backend_params is not None:
        return advanced_parameters.backend_params.get(BackendParameters.EXTERNAL_DATA_DIR, None)
    return None
