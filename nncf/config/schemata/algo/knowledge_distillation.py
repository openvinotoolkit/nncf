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
from nncf.config.schemata.basic import NUMBER
from nncf.config.schemata.basic import with_attributes

KNOWLEDGE_DISTILLATION_TYPE_SCHEMA = {
    "type": "string",
    "enum": ["mse", "softmax"]
}
KNOWLEDGE_DISTILLATION_ALGO_NAME_IN_CONFIG = 'knowledge_distillation'
KNOWLEDGE_DISTILLATION_SCHEMA = {
    **BASIC_COMPRESSION_ALGO_SCHEMA,
    "properties": {
        "algorithm": {
            "const": KNOWLEDGE_DISTILLATION_ALGO_NAME_IN_CONFIG
        },
        "type": with_attributes(KNOWLEDGE_DISTILLATION_TYPE_SCHEMA,
                                description="Type of Knowledge Distillation Loss (mse/softmax)"),
        "scale": with_attributes(NUMBER, description="Knowledge Distillation loss value multiplier", default=1),
        "temperature": with_attributes(NUMBER, description="Temperature for logits softening "
                                                           "(works only with softmax disitllation)", default=1)
    },
    "additionalProperties": False,
    "required": ["type"]
}
