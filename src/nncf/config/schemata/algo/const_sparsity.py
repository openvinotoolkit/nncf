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
from nncf.config.definitions import CONST_SPARSITY_ALGO_NAME_IN_CONFIG
from nncf.config.schemata.common.compression import BASIC_COMPRESSION_ALGO_SCHEMA
from nncf.config.schemata.common.targeting import SCOPING_PROPERTIES

CONST_SPARSITY_SCHEMA = {
    **BASIC_COMPRESSION_ALGO_SCHEMA,
    "properties": {
        "algorithm": {"const": CONST_SPARSITY_ALGO_NAME_IN_CONFIG},
        **SCOPING_PROPERTIES,
    },
    "additionalProperties": False,
    "description": "This algorithm takes no additional parameters and is used when you want to load "
    "a checkpoint trained with another sparsity algorithm and do other compression without "
    "changing the sparsity mask.",
}
