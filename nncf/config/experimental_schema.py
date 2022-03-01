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

import copy

from nncf.config.schema import QUANTIZATION_SCHEMA


EXPERIMENTAL_QUANTIZATION_ALGO_NAME_IN_CONFIG = 'experimental_quantization'
EXPERIMENTAL_QUANTIZATION_SCHEMA = copy.deepcopy(QUANTIZATION_SCHEMA)
EXPERIMENTAL_QUANTIZATION_SCHEMA['properties']['algorithm']['const'] = EXPERIMENTAL_QUANTIZATION_ALGO_NAME_IN_CONFIG


EXPERIMENTAL_REF_VS_ALGO_SCHEMA = {
    EXPERIMENTAL_QUANTIZATION_ALGO_NAME_IN_CONFIG: EXPERIMENTAL_QUANTIZATION_SCHEMA,
}
