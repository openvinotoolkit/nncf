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
from nncf.config.schemata.common.compression import BASIC_COMPRESSION_ALGO_SCHEMA
from nncf.config.schemata.common.compression import COMPRESSION_LR_MULTIPLIER_PROPERTY
from nncf.config.schemata.basic import NUMBER
from nncf.config.schemata.basic import with_attributes
from nncf.config.schemata.common.sparsity import COMMON_SPARSITY_PARAM_PROPERTIES
from nncf.config.schemata.common.targeting import SCOPING_PROPERTIES

RB_SPARSITY_ALGO_NAME_IN_CONFIG = "rb_sparsity"
RB_SPARSITY_SCHEMA = {
    **BASIC_COMPRESSION_ALGO_SCHEMA,
    "properties": {
        "algorithm": {
            "const": RB_SPARSITY_ALGO_NAME_IN_CONFIG
        },
        **COMPRESSION_LR_MULTIPLIER_PROPERTY,
        "sparsity_init": with_attributes(NUMBER,
                                         description="Initial value of the sparsity level applied to the "
                                                     "model"),
        "params":
            {
                "type": "object",
                "properties": COMMON_SPARSITY_PARAM_PROPERTIES,
                "additionalProperties": False
            },
        **SCOPING_PROPERTIES
    },
    "additionalProperties": False
}
