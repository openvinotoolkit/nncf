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
from nncf.config.definitions import MOVEMENT_SPARSITY_ALGO_NAME_IN_CONFIG
from nncf.config.schemata.basic import ARRAY_OF_NUMBERS
from nncf.config.schemata.common.compression import BASIC_COMPRESSION_ALGO_SCHEMA
from nncf.config.schemata.common.compression import COMPRESSION_LR_MULTIPLIER_PROPERTY
from nncf.config.schemata.basic import STRING, NUMBER, BOOLEAN
from nncf.config.schemata.basic import with_attributes
from nncf.config.schemata.common.targeting import SCOPING_PROPERTIES

NULL = {'type': 'null'}

SPARSE_STRUCTURE_MODE = ['fine', 'block', 'per_dim']

SPARSE_STRUCTURE_BY_SCOPES_SCHEMA = {
    "type": "object",
    "properties": {
        "mode": with_attributes(STRING,
                                description="TBD",
                                enum=SPARSE_STRUCTURE_MODE),
        "sparse_factors": with_attributes(ARRAY_OF_NUMBERS,
                                          description="TBD"),
        "axis": with_attributes(NUMBER,
                                description="TBD"),
        "target_scopes": with_attributes(STRING,
                                         description="TBD")
    },
    "additionalProperties": False,
}

SCHEDULER_PARAMS_SCHEMA = {
    "type": "object",
    "properties": {
        "power": with_attributes(NUMBER,
                                 description="For polynomial scheduler - determines the corresponding power value."),
        "init_importance_threshold": with_attributes(NUMBER,
                                                     description="importance masking threshold @ warmup_start_epoch"),
        "warmup_start_epoch": with_attributes(NUMBER,
                                              description="Index of the starting epoch for importance masking threshold"
                                                          "warmup at the value of init_importance_threshold"),
        "final_importance_threshold": with_attributes(NUMBER,
                                                      description="importance masking threshold @ warmup_end_epoch"),
        "warmup_end_epoch": with_attributes(NUMBER,
                                            description="Index of the ending epoch of the importance masking threshold"
                                            "warmup at the value of final_importance_threshold"),
        "importance_regularization_factor": with_attributes(NUMBER,
                                                            description="regularization final lambda"),
        "enable_structured_masking": with_attributes(BOOLEAN,
                                                     default=True,
                                                     description="Whether to enable structured masking"
                                                     " after warmup stage."),
        "steps_per_epoch": with_attributes({"oneOf": [NUMBER, NULL]},
                                           description="Number of optimizer steps in one epoch. "
                                           "Required to start proper scheduling in the first training epoch if "
                                           "'update_per_optimizer_step' is true"),
    },
    "additionalProperties": False
}


MOVEMENT_SPARSITY_SCHEMA = {
    **BASIC_COMPRESSION_ALGO_SCHEMA,
    # TODO: fill in description
    "description": "to-do."
                   "placeholder. ",
    "properties": {
        "algorithm": {
            "const": MOVEMENT_SPARSITY_ALGO_NAME_IN_CONFIG
        },
        "params": SCHEDULER_PARAMS_SCHEMA,
        "sparse_structure_by_scopes": {
            "type": "array",
            "items": SPARSE_STRUCTURE_BY_SCOPES_SCHEMA,
            "description": "TBD"
        },
        **SCOPING_PROPERTIES,
        **COMPRESSION_LR_MULTIPLIER_PROPERTY
    },
    "additionalProperties": False
}
