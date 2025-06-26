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
from nncf.config.definitions import MAGNITUDE_SPARSITY_ALGO_NAME_IN_CONFIG
from nncf.config.definitions import ONLINE_DOCS_ROOT
from nncf.config.schemata.basic import NUMBER
from nncf.config.schemata.basic import STRING
from nncf.config.schemata.basic import with_attributes
from nncf.config.schemata.common.compression import BASIC_COMPRESSION_ALGO_SCHEMA
from nncf.config.schemata.common.sparsity import COMMON_SPARSITY_PARAM_PROPERTIES
from nncf.config.schemata.common.targeting import GENERIC_INITIALIZER_SCHEMA
from nncf.config.schemata.common.targeting import SCOPING_PROPERTIES
from nncf.config.schemata.defaults import SPARSITY_INIT

WEIGHT_IMPORTANCE_OPTIONS = ["abs", "normed_abs"]

MAGNITUDE_SPARSITY_SCHEMA = {
    **BASIC_COMPRESSION_ALGO_SCHEMA,
    "description": f"Applies sparsity on top of the current model. Each weight tensor value will be either kept as-is, "
    f"or set to 0 based on its magnitude. For large sparsity levels, this will improve performance on "
    f"hardware that can profit from it. "
    f"See [Sparsity.md]"
    f"({ONLINE_DOCS_ROOT}"
    f"/docs/compression_algorithms/Sparsity.md#magnitude-sparsity) and the rest of this schema for "
    f"more details and parameters.",
    "properties": {
        "algorithm": {"const": MAGNITUDE_SPARSITY_ALGO_NAME_IN_CONFIG},
        "sparsity_init": with_attributes(
            NUMBER, description="Initial value of the sparsity level applied to the model.", default=SPARSITY_INIT
        ),
        "initializer": GENERIC_INITIALIZER_SCHEMA,
        "params": {
            "type": "object",
            "properties": {
                **COMMON_SPARSITY_PARAM_PROPERTIES,
                "weight_importance": with_attributes(
                    STRING,
                    description="Determines the way in which the weight values "
                    "will be sorted after being aggregated in order "
                    "to determine the sparsity threshold "
                    "corresponding to a specific sparsity level.",
                    enum=WEIGHT_IMPORTANCE_OPTIONS,
                    default="normed_abs",
                ),
            },
            "additionalProperties": False,
        },
        **SCOPING_PROPERTIES,
    },
    "additionalProperties": False,
}
