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

from nncf.config.definitions import BOOTSTRAP_NAS_ALGO_NAME_IN_CONFIG
from nncf.config.definitions import EXPERIMENTAL_QUANTIZATION_ALGO_NAME_IN_CONFIG
from nncf.config.definitions import KNOWLEDGE_DISTILLATION_ALGO_NAME_IN_CONFIG
from nncf.config.definitions import MOVEMENT_SPARSITY_ALGO_NAME_IN_CONFIG
from nncf.config.schemata.algo.quantization import QUANTIZATION_SCHEMA
from nncf.config.schemata.basic import ARRAY_OF_NUMBERS
from nncf.config.schemata.basic import ARRAY_OF_STRINGS
from nncf.config.schemata.basic import BOOLEAN
from nncf.config.schemata.basic import NUMBER
from nncf.config.schemata.basic import STRING
from nncf.config.schemata.basic import make_object_or_array_of_objects_schema
from nncf.config.schemata.basic import make_string_or_array_of_strings_schema
from nncf.config.schemata.basic import with_attributes
from nncf.config.schemata.common.compression import BASIC_COMPRESSION_ALGO_SCHEMA
from nncf.config.schemata.common.compression import COMPRESSION_LR_MULTIPLIER_PROPERTY
from nncf.config.schemata.common.initialization import BATCHNORM_ADAPTATION_SCHEMA
from nncf.config.schemata.common.targeting import IGNORED_SCOPES_DESCRIPTION
from nncf.config.schemata.common.targeting import SCOPING_PROPERTIES
from nncf.config.schemata.common.targeting import TARGET_SCOPES_DESCRIPTION

########################################################################################################################
# Experimental Quantization
########################################################################################################################
EXPERIMENTAL_QUANTIZATION_SCHEMA = copy.deepcopy(QUANTIZATION_SCHEMA)
EXPERIMENTAL_QUANTIZATION_SCHEMA["properties"]["algorithm"]["const"] = EXPERIMENTAL_QUANTIZATION_ALGO_NAME_IN_CONFIG  # type: ignore[index]

########################################################################################################################
# BootstrapNAS
########################################################################################################################

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
            "description": "List of building blocks to be skipped. The block is defined by names of start and end "
            "nodes. The end node is skipped. In contrast, the start node is executed. It produces a "
            "tensor that is bypassed through the skipping nodes until the one after end node. ",
            "examples": [[["start_op_1", "end_op_1"], ["start_op_2", "end_op_2"]]],
        },
        "min_block_size": with_attributes(
            NUMBER,
            description="Defines minimal number of operations in the skipping block. "
            "Option is available for the auto mode only. "
            "Default value is 5",
        ),
        "max_block_size": with_attributes(
            NUMBER,
            description="Defines maximal number of operations in the block. "
            "Option is available for the auto mode only. "
            "Default value is 50",
        ),
        "hw_fused_ops": with_attributes(
            BOOLEAN,
            description="If True, automatic block search will not relate operations, "
            "which are fused on inference, into different blocks for skipping. "
            "True, by default",
        ),
    },
    "additionalProperties": False,
}

ELASTIC_WIDTH_SCHEMA = {
    "type": "object",
    "properties": {
        "min_width": with_attributes(
            NUMBER,
            description="Minimal number of output channels that can be activated for "
            "each layers with elastic width. Default value is 32.",
        ),
        "max_num_widths": with_attributes(
            NUMBER,
            description="Restricts total number of different elastic width values for "
            "each layer. The default value is -1 means that there's no "
            "restrictions.",
        ),
        "width_step": with_attributes(
            NUMBER,
            description="Defines a step size for a generation of the elastic width search "
            "space - the list of all possible width values for each layer. The "
            "generation starts from the number of output channels in the "
            "original model and stops when it reaches whether a "
            "`min_width` width value or number of generated width values "
            "equal to `max_num_widths`",
        ),
        "width_multipliers": with_attributes(
            ARRAY_OF_NUMBERS,
            description="Defines elastic width search space via a list of "
            "multipliers. All possible width values are obtained by "
            "multiplying the original width value with the values in the "
            "given list.",
        ),
        "filter_importance": with_attributes(
            STRING,
            description="The type of filter importance metric. Can be"
            " one of `L1`, `L2`, `geometric_median`, `external`."
            " `L2` by default.",
        ),
        "external_importance_path": with_attributes(
            STRING,
            description="Path to the custom external weight importance (PyTorch tensor) per node "
            "that needs to weight reorder. Valid only when filter_importance "
            "is `external`. The file should be loaded via the torch interface "
            "torch.load(), represented as a dictionary. It maps NNCF node name "
            "to importance tensor with the same shape as the weights in the node "
            "module. For example, node `Model/NNCFLinear[fc1]/linear_0` has a "
            "3x1 linear module with weight [0.2, 0.3, 0.9], and in the dict"
            "{'Model/NNCFLinear[fc1]/linear_0': tensor([0.4, 0.01, 0.2])} represents "
            "the corresponding weight importance.",
        ),
    },
    "additionalProperties": False,
}

ELASTIC_KERNEL_SCHEMA = {
    "type": "object",
    "properties": {
        "max_num_kernels": with_attributes(
            NUMBER,
            description="Restricts the total number of different elastic kernel values for "
            "each layer. The default value is -1 means that there's no "
            "restrictions.",
        ),
    },
    "additionalProperties": False,
}

ELASTICITY_SCHEMA = {
    "type": "object",
    "properties": {
        "depth": ELASTIC_DEPTH_SCHEMA,
        "width": ELASTIC_WIDTH_SCHEMA,
        "kernel": ELASTIC_KERNEL_SCHEMA,
        "available_elasticity_dims": with_attributes(
            ARRAY_OF_STRINGS,
            description="Defines the available elasticity dimension for "
            "sampling subnets. By default, all elastic dimensions "
            "are available - [width, depth, kernel]",
        ),
        "ignored_scopes": with_attributes(
            make_string_or_array_of_strings_schema(),
            description=IGNORED_SCOPES_DESCRIPTION,
        ),
        "target_scopes": with_attributes(
            make_string_or_array_of_strings_schema(),
            description=TARGET_SCOPES_DESCRIPTION,
        ),
    },
    "additionalProperties": False,
}

STAGE_DESCRIPTOR_SCHEMA = {
    "type": "object",
    "properties": {
        "train_dims": with_attributes(
            ARRAY_OF_STRINGS,
            description="Elasticity dimensions that are enabled for subnet sampling,"
            "the rest elastic dimensions are disabled",
        ),
        "epochs": with_attributes(NUMBER, description="Duration of the training stage in epochs"),
        "depth_indicator": with_attributes(
            NUMBER,
            description="Restricts the maximum number of blocks in each "
            "independent group that can be skipped. For example, Resnet50 "
            "has 4 four independent groups, each group consists of a "
            "specific number of Bottleneck layers [3,4,6,3], that "
            "potentially can be skipped. If depth indicator equals to 1,"
            " only the last Bottleneck can be skipped in each group, if it "
            "equals 2 - the last two and etc. This allows to implement "
            "progressive shrinking logic from `Once for all` paper. Default "
            "value is 1.",
        ),
        "width_indicator": with_attributes(
            NUMBER,
            description="Restricts the maximum number of width values in each elastic "
            "layer. For example, some conv2d with elastic width can vary "
            "number of output channels from the following list: [8, 16, 32] "
            "If width indicator is equal to 1, it can only activate the "
            "maximum number of channels - 32. If it equals 2, then the last "
            " two can be selected - 16 or 32, or both of them.",
        ),
        "reorg_weights": with_attributes(
            BOOLEAN,
            description="if True, triggers reorganization of weights in order to have "
            "filters sorted by importance (e.g. by l2 norm) in the "
            "beginning of the stage",
        ),
        "bn_adapt": with_attributes(
            BOOLEAN,
            description="if True, triggers batchnorm adaptation in the beginning of the stage",
        ),
        "init_lr": with_attributes(
            NUMBER,
            description="Initial learning rate for a stage. If specified in the stage "
            " descriptor, it will trigger a reset of the learning rate at "
            "the beginning of the stage.",
        ),
        "epochs_lr": with_attributes(
            NUMBER,
            description="Number of epochs to compute the adjustment of the learning rate.",
        ),
        "sample_rate": with_attributes(
            NUMBER,
            description="Number of iterations to activate the random subnet. Default value is 1.",
        ),
    },
    "description": "Defines a supernet training stage: how many epochs it takes, which elasticities with which "
    "settings are enabled, whether some operation should happen in the beginning",
    "additionalProperties": False,
}
NAS_SCHEDULE_SCHEMA = {
    "type": "object",
    "properties": {
        "list_stage_descriptions": {
            "type": "array",
            "items": STAGE_DESCRIPTOR_SCHEMA,
            "description": "List of parameters per each supernet training stage",
        }
    },
    "additionalProperties": False,
}

LR_SCHEDULE_SCHEMA = {
    "type": "object",
    "properties": {
        "params": {
            "type": "object",
            "properties": {
                "base_lr": with_attributes(
                    NUMBER,
                    description="Defines a global learning rate scheduler."
                    "If these parameters are not set, a stage learning rate scheduler will be used.",
                ),
            },
            "additionalProperties": False,
        }
    },
    "additionalProperties": False,
}

BOOTSTRAP_NAS_TRAINING_SCHEMA = {
    "type": "object",
    "properties": {
        "algorithm": with_attributes(
            TRAINING_ALGORITHMS_SCHEMA,
            description="Defines training strategy for tuning supernet. By default, progressive shrinking",
        ),
        "progressivity_of_elasticity": with_attributes(
            ARRAY_OF_STRINGS,
            description="Defines the order of adding a new elasticity dimension from stage to stage",
            examples=[["width", "depth", "kernel"]],
        ),
        "batchnorm_adaptation": BATCHNORM_ADAPTATION_SCHEMA,
        "schedule": NAS_SCHEDULE_SCHEMA,
        "elasticity": ELASTICITY_SCHEMA,
        "lr_schedule": LR_SCHEDULE_SCHEMA,
        "train_steps": with_attributes(
            NUMBER,
            description="Defines the number of samples used for each training epoch.",
        ),
    },
    "additionalProperties": False,
}

SEARCH_ALGORITHMS_SCHEMA = {
    "type": "string",
    "enum": ["NSGA2", "RNSGA2"],
}

BOOTSTRAP_NAS_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "algorithm": with_attributes(
            SEARCH_ALGORITHMS_SCHEMA,
            description="Defines the search algorithm. Default algorithm is NSGA-II.",
        ),
        "batchnorm_adaptation": BATCHNORM_ADAPTATION_SCHEMA,
        "num_evals": with_attributes(
            NUMBER,
            description="Defines the number of evaluations that will be used by the search algorithm.",
        ),
        "num_constraints": with_attributes(NUMBER, description="Number of constraints in search problem."),
        "population": with_attributes(
            NUMBER,
            description="Defines the population size when using an evolutionary search algorithm.",
        ),
        "crossover_prob": with_attributes(NUMBER, description="Crossover probability used by a genetic algorithm."),
        "crossover_eta": with_attributes(NUMBER, description="Crossover eta."),
        "mutation_eta": with_attributes(NUMBER, description="Mutation eta for genetic algorithm."),
        "mutation_prob": with_attributes(NUMBER, description="Mutation probability for genetic algorithm."),
        "acc_delta": with_attributes(
            NUMBER,
            description="Defines the absolute difference in accuracy that is tolerated "
            "when looking for a subnetwork.",
        ),
        "ref_acc": with_attributes(
            NUMBER,
            description="Defines the reference accuracy from the pre-trained model used "
            "to generate the super-network.",
        ),
        "aspiration_points": with_attributes(
            ARRAY_OF_NUMBERS, description="Information to indicate the preferred parts of the Pareto front"
        ),
        "epsilon": with_attributes(NUMBER, description="epsilon distance of surviving solutions for RNSGA-II."),
        "weights": with_attributes(NUMBER, description="weights used by RNSGA-II."),
        "extreme_points_as_ref_points": with_attributes(
            BOOLEAN, description="Find extreme points and use them as aspiration points."
        ),
        "compression": make_object_or_array_of_objects_schema(
            {"oneOf": [{"$ref": f"#/$defs/{KNOWLEDGE_DISTILLATION_ALGO_NAME_IN_CONFIG}"}]}
        ),
    },
    "additionalProperties": False,
}


BOOTSTRAP_NAS_SCHEMA = {
    "type": "object",
    "properties": {
        "training": BOOTSTRAP_NAS_TRAINING_SCHEMA,
        "search": BOOTSTRAP_NAS_SEARCH_SCHEMA,
    },
    "additionalProperties": False,
}

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


MOVEMENT_SPARSITY_SCHEMA = {
    **BASIC_COMPRESSION_ALGO_SCHEMA,
    "properties": {
        "algorithm": {"const": MOVEMENT_SPARSITY_ALGO_NAME_IN_CONFIG},
        "params": MOVEMENT_SCHEDULER_PARAMS_SCHEMA,
        "sparse_structure_by_scopes": {
            "type": "array",
            "items": MOVEMENT_SPARSE_STRUCTURE_BY_SCOPES_SCHEMA,
            "description": "Describes how each supported layer will be sparsified.",
        },
        **SCOPING_PROPERTIES,
        **COMPRESSION_LR_MULTIPLIER_PROPERTY,
    },
    "additionalProperties": False,
}

########################################################################################################################
# All experimental schemas
########################################################################################################################

EXPERIMENTAL_REF_VS_ALGO_SCHEMA = {
    EXPERIMENTAL_QUANTIZATION_ALGO_NAME_IN_CONFIG: EXPERIMENTAL_QUANTIZATION_SCHEMA,
    BOOTSTRAP_NAS_ALGO_NAME_IN_CONFIG: BOOTSTRAP_NAS_SCHEMA,
    MOVEMENT_SPARSITY_ALGO_NAME_IN_CONFIG: MOVEMENT_SPARSITY_SCHEMA,
}
