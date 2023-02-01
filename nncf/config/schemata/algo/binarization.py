"""
 Copyright (c) 2023 Intel Corporation
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
from nncf.config.definitions import BINARIZATION_ALGO_NAME_IN_CONFIG
from nncf.config.definitions import ONLINE_DOCS_ROOT
from nncf.config.schemata.common.compression import BASIC_COMPRESSION_ALGO_SCHEMA
from nncf.config.schemata.common.compression import COMPRESSION_LR_MULTIPLIER_PROPERTY
from nncf.config.schemata.algo.quantization import QUANTIZATION_INITIALIZER_SCHEMA
from nncf.config.schemata.algo.quantization import STAGED_QUANTIZATION_PARAMS
from nncf.config.schemata.basic import STRING
from nncf.config.schemata.basic import with_attributes
from nncf.config.schemata.common.targeting import SCOPING_PROPERTIES
from nncf.config.schemata.defaults import BINARIZATION_MODE

BINARIZATION_MODE_OPTIONS = ['xnor', 'dorefa']
BINARIZATION_SCHEMA = {
    **BASIC_COMPRESSION_ALGO_SCHEMA,
    "description": f"Binarization is a specific particular case of the more general quantization algorithm."
                   f"\nSee [Binarization.md]"
                   f"({ONLINE_DOCS_ROOT}"
                   f"/docs/compression_algorithms/Binarization.md) and the rest of this schema for "
                   f"more details and parameters.",
    "properties": {
        "algorithm": {
            "const": BINARIZATION_ALGO_NAME_IN_CONFIG
        },
        "mode": with_attributes(STRING,
                                description="Selects the mode of binarization - either 'xnor' for XNOR binarization,"
                                            "or 'dorefa' for DoReFa binarization.",
                                enum=BINARIZATION_MODE_OPTIONS,
                                default=BINARIZATION_MODE),
        "initializer": QUANTIZATION_INITIALIZER_SCHEMA,
        **STAGED_QUANTIZATION_PARAMS,
        **SCOPING_PROPERTIES,
        **COMPRESSION_LR_MULTIPLIER_PROPERTY,
    },
    "additionalProperties": False
}
