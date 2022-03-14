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

from nncf.config.schema import ARRAY_OF_NUMBERS
from nncf.config.schema import ARRAY_OF_STRINGS
from nncf.config.schema import BATCHNORM_ADAPTATION_SCHEMA
from nncf.config.schema import BOOLEAN
from nncf.config.schema import IGNORED_SCOPES_DESCRIPTION
from nncf.config.schema import NUMBER
from nncf.config.schema import QUANTIZATION_SCHEMA
from nncf.config.schema import STRING
from nncf.config.schema import TARGET_SCOPES_DESCRIPTION
from nncf.config.schema import make_string_or_array_of_strings_schema
from nncf.config.schema import with_attributes

########################################################################################################################
# Experimental Quantization
########################################################################################################################
EXPERIMENTAL_QUANTIZATION_ALGO_NAME_IN_CONFIG = 'experimental_quantization'
EXPERIMENTAL_QUANTIZATION_SCHEMA = copy.deepcopy(QUANTIZATION_SCHEMA)
EXPERIMENTAL_QUANTIZATION_SCHEMA['properties']['algorithm']['const'] = EXPERIMENTAL_QUANTIZATION_ALGO_NAME_IN_CONFIG

########################################################################################################################
# BootstrapNAS
########################################################################################################################
BOOTSTRAP_NAS_ALGO_NAME_IN_CONFIG = 'bootstrapNAS'

TRAINING_ALGORITHMS_SCHEMA = {
    "type": "string",
    "enum": ["progressive_shrinking"],
}

ELASTIC_DEPTH_MODE_SCHEMA = {
    "type": "string",
    "enum": ["manual", "auto"],
}

ELASTIC_DEPTH_SCHEMA = {
    "type": "object",
    "properties": {
        "skipped_blocks": {
            "type": "array",
            "items": ARRAY_OF_STRINGS,
            "description": "List of building blocks to be skipped. "
                           "The block is defined by names of start and end nodes.",
            "examples": [
                [
                    ["start_op_1", "end_op_1"],
                    ["start_op_2", "end_op_2"]
                ]
            ],
        },
        "mode": with_attributes(ELASTIC_DEPTH_MODE_SCHEMA,
                                description="The way of elastic depth configuration - how skipped blocks are "
                                            "defined. Two modes are supported: manual and auto. The first "
                                            "refers to explicit setting coordinates of blocks in the config. "
                                            "The latter assumes no input from the user - blocks to skip are "
                                            "found automatically"),
        "min_block_size": with_attributes(NUMBER,
                                          description="Defines minimal number of operations in the skipping block. "
                                                      "Option is available for the auto mode only. "
                                                      "Default value is 6"),
        "max_block_size": with_attributes(NUMBER,
                                          description="Defines minimal number of operations in the block. "
                                                      "Option is available for the auto mode only. "
                                                      "Default value is 50"),
        "allow_nested_blocks": with_attributes(BOOLEAN,
                                               description="If true, automatic block search will consider nested "
                                                           "blocks: the ones that are part of bigger block. By "
                                                           "default, nested blocks are declined during the search and "
                                                           "bigger blocks are found only. False, by default"),
        "allow_linear_combination": with_attributes(BOOLEAN,
                                                    description="If False, automatic block search will decline blocks "
                                                                "that are a combination of other blocks, "
                                                                "in another words, that consist entirely of operations "
                                                                "of other blocks. False, by default"),
    },
    "additionalProperties": False
}

ELASTIC_WIDTH_SCHEMA = {
    "type": "object",
    "properties": {
        "min_width": with_attributes(NUMBER,
                                     description="Minimal number of output channels that can be activated for "
                                                 "each layers with elastic width. Default value is 32."),
        "max_num_widths": with_attributes(NUMBER,
                                          description="Restricts total number of different elastic width values for "
                                                      "each layer. The default value is -1 means that there's no "
                                                      "restrictions."),
        "width_step": with_attributes(NUMBER,
                                      description="Defines a step size for a generation of the elastic width search "
                                                  "space - the list of all possible width values for each layer. The "
                                                  "generation starts from the number of output channels in the "
                                                  "original model and stops when it reaches whether a "
                                                  "`min_width` width value or number of generated width values "
                                                  "equal to `max_num_widths`"),
        "width_multipliers": with_attributes(ARRAY_OF_NUMBERS,
                                             description="Defines elastic width search space via a list of "
                                                         "multipliers. All possible width values are obtained by "
                                                         "multiplying the original width value with the values in the "
                                                         "given list."),
        "filter_importance": with_attributes(STRING,
                                             description="The type of filter importance metric. Can be"
                                                         " one of `L1`, `L2`, `geometric_median`."
                                                         " `L2` by default.")
    },
    "additionalProperties": False
}

ELASTIC_KERNEL_SCHEMA = {
    "type": "object",
    "properties": {
        "max_num_kernels": with_attributes(NUMBER,
                                           description="Restricts total number of different elastic kernel values for "
                                                       "each layer. The default value is -1 means that there's no "
                                                       "restrictions."),
    },
    "additionalProperties": False
}

ELASTICITY_SCHEMA = {
    "type": "object",
    "properties": {
        "depth": ELASTIC_DEPTH_SCHEMA,
        "width": ELASTIC_WIDTH_SCHEMA,
        "kernel": ELASTIC_KERNEL_SCHEMA,
        "available_elasticity_dims": with_attributes(ARRAY_OF_STRINGS,
                                                     description="Defines the available elasticity dimension for "
                                                                 "sampling subnets. By default, all elastic dimensions "
                                                                 "are available - [width, depth, kernel]"),
        "ignored_scopes": with_attributes(make_string_or_array_of_strings_schema(),
                                          description=IGNORED_SCOPES_DESCRIPTION),
        "target_scopes": with_attributes(make_string_or_array_of_strings_schema(),
                                         description=TARGET_SCOPES_DESCRIPTION),
    },
    "additionalProperties": False
}

STAGE_DESCRIPTOR_SCHEMA = {
    "type": "object",
    "properties": {
        "train_dims": with_attributes(ARRAY_OF_STRINGS,
                                      description="Elasticity dimensions that are enabled for subnet sampling,"
                                                  "the rest elastic dimensions are disabled"),
        "epochs": with_attributes(NUMBER,
                                  description="Duration of the training stage in epochs"),
        "depth_indicator": with_attributes(NUMBER,
                                           description="Restricts the maximum number of blocks in each "
                                                       "independent group that can be skipped. For example, Resnet50 "
                                                       "has 4 four independent groups, each group consists of a "
                                                       "specific number of Bottleneck layers [3,4,6,3], that "
                                                       "potentially can be skipped. If depth indicator equals to 1,"
                                                       " only the last Bottleneck can be skipped in each group, if it "
                                                       "equals 2 - the last two and etc. This allows to implement "
                                                       "progressive shrinking logic from `Once for all` paper. Default "
                                                       "value is 1."),
        "width_indicator": with_attributes(NUMBER,
                                           description="Restricts the maximum number of width values in each elastic "
                                                       "layer. For example, some conv2d with elastic width can vary "
                                                       "number of output channels from the following list: [8, 16, 32] "
                                                       "If width indicator is equal to 1, it can only activate the "
                                                       "maximum number of channels - 32. If it equals 2, then the last "
                                                       " two can be selected - 16 or 32, or both of them."),
        "reorg_weights": with_attributes(BOOLEAN,
                                         description="if True, triggers reorganization of weights in order to have "
                                                     "filters sorted by importance (e.g. by l2 norm) in the "
                                                     "beginning of the stage"),
        "bn_adapt": with_attributes(BOOLEAN,
                                    description="if True, triggers batchnorm adaptation in the beginning of the stage"),
    },
    "description": "Defines a supernet training stage: how many epochs it takes, which elasticities with which "
                   "settings are enabled, whether some operation should happen in the beginning",
    "additionalProperties": False

}
NAS_SCHEDULE_SCHEMA = {
    "type": "object",
    "properties": {
        "list_stage_descriptions": {
            "type": "array",
            "items": STAGE_DESCRIPTOR_SCHEMA,
            "description": "List of parameters per each supernet training stage"
        }
    },
    "additionalProperties": False
}
BOOTSTRAP_NAS_TRAINING_SCHEMA = {
    "type": "object",
    "properties": {
        "algorithm": with_attributes(TRAINING_ALGORITHMS_SCHEMA,
                                     description="Defines training strategy for tuning supernet. By default, "
                                                 "progressive shrinking"),
        "progressivity_of_elasticity": with_attributes(ARRAY_OF_STRINGS,
                                                       description="Defines the order of adding a new elasticity "
                                                                   "dimension from stage to stage",
                                                       examples=[["width", "depth", "kernel"]]),
        "batchnorm_adaptation": BATCHNORM_ADAPTATION_SCHEMA,
        "schedule": NAS_SCHEDULE_SCHEMA,
        "elasticity": ELASTICITY_SCHEMA
    },
    "additionalProperties": False
}

BOOTSTRAP_NAS_SCHEMA = {
    "type": "object",
    "properties": {
        "training": BOOTSTRAP_NAS_TRAINING_SCHEMA
    },
    "additionalProperties": False
}

########################################################################################################################
# All experimental schemas
########################################################################################################################

EXPERIMENTAL_REF_VS_ALGO_SCHEMA = {
    EXPERIMENTAL_QUANTIZATION_ALGO_NAME_IN_CONFIG: EXPERIMENTAL_QUANTIZATION_SCHEMA,
    BOOTSTRAP_NAS_ALGO_NAME_IN_CONFIG: BOOTSTRAP_NAS_SCHEMA
}
