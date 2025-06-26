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
from nncf.config.definitions import KNOWLEDGE_DISTILLATION_ALGO_NAME_IN_CONFIG
from nncf.config.definitions import ONLINE_DOCS_ROOT
from nncf.config.schemata.basic import NUMBER
from nncf.config.schemata.basic import STRING
from nncf.config.schemata.basic import with_attributes
from nncf.config.schemata.common.compression import BASIC_COMPRESSION_ALGO_SCHEMA
from nncf.config.schemata.defaults import KNOWLEDGE_DISTILLATION_SCALE
from nncf.config.schemata.defaults import KNOWLEDGE_DISTILLATION_TEMPERATURE

KNOWLEDGE_DISTILLATION_TYPE_OPTIONS = ["mse", "softmax"]

KNOWLEDGE_DISTILLATION_SCHEMA = {
    **BASIC_COMPRESSION_ALGO_SCHEMA,
    "description": f"This algorithm is only useful in combination with other compression algorithms and improves the"
    f"end accuracy result of the corresponding algorithm by calculating knowledge distillation loss "
    f"between the compressed model currently in training and its original, uncompressed counterpart. "
    f"See [KnowledgeDistillation.md]"
    f"({ONLINE_DOCS_ROOT}"
    f"/docs/compression_algorithms/KnowledgeDistillation.md) and the rest of this schema for "
    f"more details and parameters.",
    "properties": {
        "algorithm": {"const": KNOWLEDGE_DISTILLATION_ALGO_NAME_IN_CONFIG},
        "type": with_attributes(
            STRING, description="Type of Knowledge Distillation Loss.", enum=KNOWLEDGE_DISTILLATION_TYPE_OPTIONS
        ),
        "scale": with_attributes(
            NUMBER, description="Knowledge Distillation loss value multiplier", default=KNOWLEDGE_DISTILLATION_SCALE
        ),
        "temperature": with_attributes(
            NUMBER,
            description="`softmax` type only - Temperature for logits softening.",
            default=KNOWLEDGE_DISTILLATION_TEMPERATURE,
        ),
    },
    "additionalProperties": False,
    "required": ["type"],
}
