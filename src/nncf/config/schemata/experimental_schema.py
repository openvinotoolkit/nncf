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

import copy

from nncf.config.definitions import EXPERIMENTAL_QUANTIZATION_ALGO_NAME_IN_CONFIG
from nncf.config.schemata.algo.quantization import QUANTIZATION_SCHEMA
from nncf.config.schemata.basic import ARRAY_OF_NUMBERS
from nncf.config.schemata.basic import BOOLEAN
from nncf.config.schemata.basic import NUMBER
from nncf.config.schemata.basic import STRING
from nncf.config.schemata.basic import make_string_or_array_of_strings_schema
from nncf.config.schemata.basic import with_attributes

########################################################################################################################
# Experimental Quantization
########################################################################################################################
EXPERIMENTAL_QUANTIZATION_SCHEMA = copy.deepcopy(QUANTIZATION_SCHEMA)
EXPERIMENTAL_QUANTIZATION_SCHEMA["properties"]["algorithm"]["const"] = EXPERIMENTAL_QUANTIZATION_ALGO_NAME_IN_CONFIG  # type: ignore[index]

########################################################################################################################
# Movement Sparsity
########################################################################################################################

MOVEMENT_SPARSE_STRUCTURE_MODE = ["fine", "block", "per_dim"]
MOVEMENT_POWER = 3.0
MOVEMENT_FINAL_IMPORTANCE_THRESHOLD = 0.0
MOVEMENT_ENABLE_STRUCTURED_MASKING = True

MOVEMENT_SPARSE_STRUCTURE_BY_SCOPES_SCHEMA = {
    "type": "object",
    "properties": {
        "mode": with_attributes(
            STRING,
            description="Defines in which mode a supported layer will be sparsified.",
            enum=MOVEMENT_SPARSE_STRUCTURE_MODE,
        ),
        "sparse_factors": with_attributes(
            ARRAY_OF_NUMBERS,
            description='The block shape for weights to sparsify. Required when `mode`="block".',
        ),
        "axis": with_attributes(
            NUMBER,
            description='The dimension for weights to sparsify. Required when `mode`="per_dim".',
        ),
        "target_scopes": with_attributes(
            make_string_or_array_of_strings_schema(),
            description="Model control flow graph node scopes to be considered in this mode.",
        ),
    },
    "additionalProperties": False,
    "required": ["mode", "target_scopes"],
}

MOVEMENT_SCHEDULER_PARAMS_SCHEMA = {
    "type": "object",
    "properties": {
        "warmup_start_epoch": with_attributes(
            NUMBER,
            description="Index of the starting epoch (include) for warmup stage.",
        ),
        "warmup_end_epoch": with_attributes(NUMBER, description="Index of the end epoch (exclude) for warmup stage."),
        "importance_regularization_factor": with_attributes(
            NUMBER,
            description="The regularization factor on weight importance scores. With a larger "
            "positive value, more model weights will be regarded as less important "
            "and thus be sparsified.",
        ),
        "enable_structured_masking": with_attributes(
            BOOLEAN,
            description="Whether to do structured mask resolution after warmup stage. Only "
            "supports structured masking on multi-head self-attention blocks and "
            "feed-forward networks now.",
            default=MOVEMENT_ENABLE_STRUCTURED_MASKING,
        ),
        "power": with_attributes(
            NUMBER,
            description="The power value of polynomial decay for threshold and "
            "regularization factor update during warmup stage.",
            default=MOVEMENT_POWER,
        ),
        "init_importance_threshold": with_attributes(
            NUMBER,
            description="The initial value of importance threshold during warmup stage. If not "
            "specified, this will be automatically decided during training so that "
            "the model is with about 0.1% linear layer sparsity on involved layers at "
            "the beginning of warmup stage.",
        ),
        "final_importance_threshold": with_attributes(
            NUMBER,
            description="The final value of importance threshold during warmup stage.",
            default=MOVEMENT_FINAL_IMPORTANCE_THRESHOLD,
        ),
        "steps_per_epoch": with_attributes(
            NUMBER,
            description="Number of training steps in one epoch, used for proper threshold and "
            "regularization factor updates. Optional if warmup_start_epoch >=1 since "
            "this can be counted in the 1st epoch. Otherwise users have to specify it.",
        ),
    },
    "additionalProperties": False,
    "required": [
        "warmup_start_epoch",
        "warmup_end_epoch",
        "importance_regularization_factor",
    ],
}

########################################################################################################################
# All experimental schemas
########################################################################################################################

EXPERIMENTAL_REF_VS_ALGO_SCHEMA = {
    EXPERIMENTAL_QUANTIZATION_ALGO_NAME_IN_CONFIG: EXPERIMENTAL_QUANTIZATION_SCHEMA,
}
