"""
 Copyright (c) 2020 Intel Corporation
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
import logging
from typing import Dict

import jsonschema

logger = logging.getLogger('nncf')


def make_string_or_array_of_strings_schema(addtl_dict_entries: Dict = None) -> Dict:
    if addtl_dict_entries is None:
        addtl_dict_entries = {}
    retval = {
        "type": ["array", "string"],
        "items": {
            "type": "string"
        }
    }
    retval.update(addtl_dict_entries)
    return retval


def make_object_or_array_of_objects_schema(single_object_schema: Dict = None) -> Dict:
    retval = {
        "oneOf": [
            {
                "type": "array",
                "items": single_object_schema
            },
            single_object_schema
        ]
    }
    return retval


def with_attributes(schema: Dict, **kwargs) -> Dict:
    retval = {**schema, **kwargs}
    return retval


_NUMBER = {
    "type": "number"
}

_STRING = {
    "type": "string"
}

_BOOLEAN = {
    "type": "boolean"
}

_ARRAY_OF_NUMBERS = {
    "type": "array",
    "items": _NUMBER
}

_ARRAY_OF_STRINGS = {
    "type": "array",
    "items": _STRING
}

SINGLE_INPUT_INFO_SCHEMA = {
    "type": "object",
    "properties": {
        "sample_size": with_attributes(_ARRAY_OF_NUMBERS,
                                       description="Shape of the tensor expected as input to the model.",
                                       examples=[[1, 3, 224, 224]]),
        "type": with_attributes(_STRING,
                                description="Data type of the model input tensor."),
        "filler": with_attributes(_STRING,
                                  description="Determines what the tensor will be filled with when passed to the model"
                                              " during tracing and exporting."),
        "keyword": with_attributes(_STRING,
                                   description="Keyword to be used when passing the tensor to the model's "
                                               "'forward' method.")
    },
    "additionalProperties": False
}

QUANTIZER_CONFIG_PROPERTIES = {
    "mode": with_attributes(_STRING,
                            description="Mode of quantization"),
    "bits": with_attributes(_NUMBER,
                            description="Bitwidth to quantize to. It is intended for manual bitwidth setting. Can be "
                                        "overridden by the `bits` parameter from the `precision` initializer section. "
                                        "An error happens if it doesn't match a corresponding bitwidth constraints "
                                        "from the hardware configuration."),
    "signed": with_attributes(_BOOLEAN,
                              description="Whether to use signed or unsigned input/output values for quantization."
                                          " If specified as unsigned and the input values during initialization have "
                                          "differing signs, will reset to performing signed quantization instead."),
    "logarithm_scale": with_attributes(_BOOLEAN,
                                       description="Whether to use log of scale as optimized parameter"
                                                   " instead of scale itself."),
    "per_channel": with_attributes(_BOOLEAN,
                                   description="Whether to quantize inputs per channel (i.e. per 0-th dimension for "
                                               "weight quantization, and per 1-st dimension for activation "
                                               "quantization)")
}

IGNORED_SCOPES_DESCRIPTION = "A list of model control flow graph node scopes to be ignored for this " \
                             "operation - functions as a 'allowlist'. Optional."
TARGET_SCOPES_DESCRIPTION = "A list of model control flow graph node scopes to be considered for this operation" \
                            " - functions as a 'denylist'. Optional."

QUANTIZER_GROUP_PROPERTIES = {
    **QUANTIZER_CONFIG_PROPERTIES,
    "ignored_scopes": with_attributes(make_object_or_array_of_objects_schema(_STRING),
                                      description=IGNORED_SCOPES_DESCRIPTION),
    "target_scopes": with_attributes(make_object_or_array_of_objects_schema(_STRING),
                                     description=TARGET_SCOPES_DESCRIPTION)
}

WEIGHTS_GROUP_SCHEMA = {
    "type": "object",
    "properties": {
        **QUANTIZER_GROUP_PROPERTIES,
    },
    "additionalProperties": False
}

LINKED_ACTIVATION_SCOPES_SPECIFIER_SCHEMA = {
    "type": "array",
    "items": _ARRAY_OF_STRINGS
}

ACTIVATIONS_GROUP_SCHEMA = {
    "type": "object",
    "properties": {
        **QUANTIZER_GROUP_PROPERTIES,
        "linked_quantizer_scopes": with_attributes(LINKED_ACTIVATION_SCOPES_SPECIFIER_SCHEMA,
                                                   description="Specifies points in the model which will share the "
                                                               "same quantizer module for activations. This is helpful "
                                                               "in case one and the same quantizer scale is required "
                                                               "for inputs to the same operation. Each sub-array will"
                                                               "define a group of activation quantizer insertion "
                                                               "points that have to share a single actual "
                                                               "quantization module, each entry in this subarray "
                                                               "should correspond to exactly one node in the NNCF "
                                                               "graph and the groups should not overlap. The final"
                                                               "quantizer for each sub-array will be associated with "
                                                               "the first element of this sub-array.")
    },
    "additionalProperties": False
}

GENERIC_INITIALIZER_SCHEMA = {
    "type": "object",
    "properties": {
        "batchnorm_adaptation":
            {
                "type": "object",
                "properties": {
                    "num_bn_adaptation_samples": with_attributes(_NUMBER,
                                                                 description="Number of samples from the training "
                                                                             "dataset to use for model inference "
                                                                             "during the BatchNorm statistics "
                                                                             "adaptation procedure for the compressed "
                                                                             "model. The actual number of samples will "
                                                                             "be a closest multiple of the batch "
                                                                             "size."),
                    "num_bn_forget_samples": with_attributes(_NUMBER,
                                                             description="Number of samples from the training "
                                                                         "dataset to use for model inference during "
                                                                         "the BatchNorm statistics adaptation "
                                                                         "in the initial statistics forgetting step. "
                                                                         "The actual number of samples will be a "
                                                                         "closest multiple of the batch size."),
                },
                "additionalProperties": False,
            },
    },
    "additionalProperties": False,
}

BASIC_RANGE_INIT_CONFIG_PROPERTIES = {
    "type": "object",
    "properties": {
        "num_init_samples": with_attributes(_NUMBER,
                                            description="Number of samples from the training dataset to "
                                                        "consume as sample model inputs for purposes of "
                                                        "setting initial minimum and maximum quantization "
                                                        "ranges"),
        "type": with_attributes(_STRING, description="Type of the initializer - determines which "
                                                     "statistics gathered during initialization will be "
                                                     "used to initialize the quantization ranges"),
        "params": {
            "type": "object",
            "properties": {
                "min_percentile": with_attributes(_NUMBER,
                                                  description="For 'percentile' type - specify the percentile of "
                                                              "input value histograms to be set as the initial "
                                                              "value for minimum quantizer input"),
                "max_percentile": with_attributes(_NUMBER,
                                                  description="For 'percentile' type - specify the percentile of "
                                                              "input value histograms to be set as the initial "
                                                              "value for maximum quantizer input"),
            }
        }
    },
    "additionalProperties": False,
}
PER_LAYER_RANGE_INIT_CONFIG_PROPERTIES = {
    "type": "object",
    "properties": {
        **BASIC_RANGE_INIT_CONFIG_PROPERTIES["properties"],
        "target_scopes": with_attributes(make_string_or_array_of_strings_schema(),
                                         description=TARGET_SCOPES_DESCRIPTION),
        "ignored_scopes": with_attributes(make_string_or_array_of_strings_schema(),
                                          description=IGNORED_SCOPES_DESCRIPTION),
        "target_quantizer_qroup": with_attributes(_STRING, description="The target group of quantizers for which "
                                                                       "specified type of range initialization will "
                                                                       "be applied. It can take 'activations' or "
                                                                       "'weights'. By default specified type of range "
                                                                       "initialization will be applied to all group of"
                                                                       "quantizers. Optional.")
    }
}
RANGE_INIT_CONFIG_PROPERTIES = {
    "initializer": {
        "type": "object",
        "properties": {
            "range": {
                "oneOf": [
                    {
                        "type": "array",
                        "items": PER_LAYER_RANGE_INIT_CONFIG_PROPERTIES
                    },
                    BASIC_RANGE_INIT_CONFIG_PROPERTIES
                ],
            },
        },
        "additionalProperties": False,
    },
}

BITWIDTH_ASSIGNMENT_MODE_SCHEMA = {
    "type": "string",
    "enum": ['strict', 'liberal'],
    "default": "liberal",
    "description": "The mode for assignment bitwidth to activation quantizers. A group of quantizers between modules "
                   "with quantizable inputs have the same bitwidth in the strict mode. Liberal one allows different "
                   "precisions within the group. Bitwidth is assigned based on hardware constraints. If multiple "
                   "variants are possible the minimal compatible bitwidth is chosen."
}

QUANTIZATION_INITIALIZER_SCHEMA = {
    "type": "object",
    "properties": {
        "batchnorm_adaptation":
            {
                "type": "object",
                "properties": {
                    "num_bn_adaptation_samples": with_attributes(_NUMBER,
                                                                 description="Number of samples from the training "
                                                                             "dataset to use for model inference "
                                                                             "durung the BatchNorm statistics "
                                                                             "adaptation procedure for the compressed "
                                                                             "model"),
                    "num_bn_forget_samples": with_attributes(_NUMBER,
                                                             description="Number of samples from the training "
                                                                         "dataset to use for model inference during "
                                                                         "the BatchNorm statistics adaptation "
                                                                         "in the initial statistics forgetting step"),
                },
                "additionalProperties": False,
            },
        **RANGE_INIT_CONFIG_PROPERTIES["initializer"]["properties"],
        "precision":
            {
                "type": "object",
                "properties": {
                    "type": with_attributes(_STRING,
                                            description="Type of precision initialization."),
                    "bits": with_attributes(_ARRAY_OF_NUMBERS,
                                            description="A list of bitwidth to choose from when performing precision "
                                                        "initialization. Overrides bits constraints specified in "
                                                        "`weight` and `activation` sections",
                                            examples=[[4, 8]]),
                    "num_data_points": with_attributes(_NUMBER,
                                                       description="Number of data points to iteratively estimate "
                                                                   "Hessian trace, 1000 by default."),
                    "iter_number": with_attributes(_NUMBER,
                                                   description="Maximum number of iterations of Hutchinson algorithm "
                                                               "to Estimate Hessian trace, 500 by default"),
                    "tolerance": with_attributes(_NUMBER,
                                                 description="Minimum relative tolerance for stopping the Hutchinson "
                                                             "algorithm. It's calculated  between mean average trace "
                                                             "from previous iteration and current one. 1e-5 by default"
                                                             "bitwidth_per_scope"),
                    "compression_ratio": with_attributes(_NUMBER,
                                                         description="The desired ratio between bits complexity of "
                                                                     "fully INT8 model and mixed-precision lower-bit "
                                                                     "one. On precision initialization stage the HAWQ "
                                                                     "algorithm chooses the most accurate "
                                                                     "mixed-precision configuration with ratio no less "
                                                                     "than the specified. Bit complexity of the model "
                                                                     "is a sum of bit complexities for each quantized "
                                                                     "layer, which are a multiplication of FLOPS for "
                                                                     "the layer by number of bits for its "
                                                                     "quantization.",
                                                         default=1.5),
                    "eval_subset_ratio": with_attributes(_NUMBER,
                                                         description="The desired ratio of dataloader to be iterated "
                                                                     "during each search iteration of AutoQ precision "
                                                                     "initialization. Specifically, this ratio applies "
                                                                     "to the registered autoq_eval_loader via "
                                                                     "register_default_init_args.",
                                                         default=1.0),
                    "warmup_iter_number": with_attributes(_NUMBER,
                                                         description="The number of random policy at the beginning of "
                                                                     "of AutoQ precision initialization to populate "
                                                                     "replay buffer with experiences. This key is "
                                                                     "meant for internal testing use. Users need not "
                                                                     "to configure.",
                                                         default=20),
                    "bitwidth_per_scope": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items":
                                [
                                    _NUMBER,
                                    _STRING
                                ],
                            "description": "A tuple of a bitwidth and a scope of the quantizer to assign the "
                                           "bitwidth to."
                        },
                        "description": "Manual settings for the quantizer bitwidths. Scopes are used to identify "
                                       "the quantizers."
                    },
                    "traces_per_layer_path": with_attributes(_STRING,
                                                             description="Path to serialized PyTorch Tensor with "
                                                                         "average Hessian traces per quantized modules."
                                                                         " It can be used to accelerate mixed precision"
                                                                         "initialization by using average Hessian "
                                                                         "traces from previous run of HAWQ algorithm."),
                    "dump_init_precision_data": with_attributes(_BOOLEAN,
                                                      description="Whether to dump data related to Precision "
                                                                  "Initialization algorithm. HAWQ dump includes "
                                                                  "bitwidth graph, average traces and different "
                                                                  "plots. AutoQ dump includes DDPG agent "
                                                                  "learning trajectory in tensorboard and "
                                                                  "mixed-precision environment metadata.",
                                                      default=True),
                    "bitwidth_assignment_mode": BITWIDTH_ASSIGNMENT_MODE_SCHEMA,
                },
                "additionalProperties": False,
            }
    },
    "additionalProperties": False,
}

COMMON_COMPRESSION_ALGORITHM_PROPERTIES = {
    "ignored_scopes": with_attributes(make_string_or_array_of_strings_schema(),
                                      description=IGNORED_SCOPES_DESCRIPTION),
    "target_scopes": with_attributes(make_string_or_array_of_strings_schema(),
                                     description=TARGET_SCOPES_DESCRIPTION),
}

BASIC_COMPRESSION_ALGO_SCHEMA = {
    "type": "object",
    "required": ["algorithm"]
}

STAGED_QUANTIZATION_PARAMS = {
    "params": {
        "type": "object",
        "properties": {
            "batch_multiplier": with_attributes(_NUMBER,
                                                description="Gradients will be accumulated for this number of "
                                                            "batches before doing a 'backward' call. Increasing "
                                                            "this may improve training quality, since binarized "
                                                            "networks exhibit noisy gradients requiring larger "
                                                            "batch sizes than could be accomodated by GPUs"),
            "activations_quant_start_epoch": with_attributes(_NUMBER,
                                                             description="Epoch to start binarizing activations"),
            "weights_quant_start_epoch": with_attributes(_NUMBER,
                                                         description="Epoch to start binarizing weights"),
            "lr_poly_drop_start_epoch": with_attributes(_NUMBER,
                                                        description="Epoch to start dropping the learning rate"),
            "lr_poly_drop_duration_epochs": with_attributes(_NUMBER,
                                                            description="Duration, in epochs, of the learning "
                                                                        "rate dropping process."),
            "disable_wd_start_epoch": with_attributes(_NUMBER,
                                                      description="Epoch to disable weight decay in the optimizer"),
            "base_lr": with_attributes(_NUMBER, description="Initial value of learning rate"),
            "base_wd": with_attributes(_NUMBER, description="Initial value of weight decay"),
        },
        "additionalProperties": False
    }
}

QUANTIZATION_ALGO_NAME_IN_CONFIG = "quantization"

QUANTIZATION_SCHEMA = {
    **BASIC_COMPRESSION_ALGO_SCHEMA,
    "properties": {
        "algorithm": {
            "const": QUANTIZATION_ALGO_NAME_IN_CONFIG
        },
        "initializer": QUANTIZATION_INITIALIZER_SCHEMA,
        "weights": with_attributes(WEIGHTS_GROUP_SCHEMA,
                                   description="Constraints to be applied to model weights quantization only."),
        "activations": with_attributes(ACTIVATIONS_GROUP_SCHEMA,
                                       description="Constraints to be applied to model activations quantization only."),
        "quantize_inputs": with_attributes(_BOOLEAN,
                                           description="Whether the model inputs should be immediately quantized prior "
                                                       "to any other model operations.",
                                           default=True),
        "quantize_outputs": with_attributes(_BOOLEAN,
                                            description="Whether the model outputs should be additionally quantized.",
                                            default=False),

        "quantizable_subgraph_patterns": {
            "type": "array",
            "items": make_string_or_array_of_strings_schema(),
            "description": "Each sub-list in this list will correspond to a sequence of operations in the "
                           "model control flow graph that will have a quantizer appended at the end of the "
                           "sequence",
            "examples": [["cat", "batch_norm"], "h_swish"]
        },
        "scope_overrides": {
            "type": "object",
            "patternProperties": {
                ".*": {
                    "type": "object",
                    "properties": {
                        **QUANTIZER_CONFIG_PROPERTIES,
                        **RANGE_INIT_CONFIG_PROPERTIES,
                    },
                    "additionalProperties": False
                },
            },
            "description": "This option is used to specify overriding quantization constraints for specific scope,"
                           "e.g. in case you need to quantize a single operation differently than the rest of the "
                           "model."
        },
        "export_to_onnx_standard_ops": with_attributes(_BOOLEAN,
                                                       description="Determines how should the additional quantization "
                                                                   "operations be exported into the ONNX format. Set "
                                                                   "this to false for export to OpenVINO-supported "
                                                                   "FakeQuantize ONNX, or to true for export to ONNX "
                                                                   "standard QuantizeLinear-DequantizeLinear "
                                                                   "node pairs (8-bit quantization only in the latter "
                                                                   "case). Default: false"),
        **STAGED_QUANTIZATION_PARAMS,
        **COMMON_COMPRESSION_ALGORITHM_PROPERTIES,
    },
    "additionalProperties": False
}

BINARIZATION_ALGO_NAME_IN_CONFIG = "binarization"
BINARIZATION_SCHEMA = {
    **BASIC_COMPRESSION_ALGO_SCHEMA,
    "properties": {
        "algorithm": {
            "const": BINARIZATION_ALGO_NAME_IN_CONFIG
        },
        "initializer": QUANTIZATION_INITIALIZER_SCHEMA,
        "mode": with_attributes(_STRING,
                                description="Selects the mode of binarization - either 'xnor' for XNOR binarization,"
                                            "or 'dorefa' for DoReFa binarization"),
        **STAGED_QUANTIZATION_PARAMS,
        **COMMON_COMPRESSION_ALGORITHM_PROPERTIES
    },
    "additionalProperties": False
}

CONST_SPARSITY_ALGO_NAME_IN_CONFIG = "const_sparsity"
CONST_SPARSITY_SCHEMA = {
    **BASIC_COMPRESSION_ALGO_SCHEMA,
    "properties": {
        "algorithm": {
            "const": CONST_SPARSITY_ALGO_NAME_IN_CONFIG
        },
        **COMMON_COMPRESSION_ALGORITHM_PROPERTIES,
    },
    "additionalProperties": False,
    "description": "This algorithm takes no additional parameters and is used when you want to load "
                   "a checkpoint trained with another sparsity algorithm and do other compression without "
                   "changing the sparsity mask."
}

COMMON_SPARSITY_PARAM_PROPERTIES = {
    "schedule": with_attributes(_STRING,
                                description="The type of scheduling to use for adjusting the target"
                                            "sparsity level"),
    "patience": with_attributes(_NUMBER,
                                description="A regular patience parameter for the scheduler, "
                                            "as for any other standard scheduler. Specified in units "
                                            "of scheduler steps."),
    "power": with_attributes(_NUMBER,
                             description="For polynomial scheduler - determines the corresponding power value."),
    "concave": with_attributes(_BOOLEAN, description="For polynomial scheduler - if True, then the target sparsity "
                                                     "level will be approached in concave manner, and in convex "
                                                     "manner otherwise."),
    "sparsity_target": with_attributes(_NUMBER,
                                       description="Target value of the sparsity level for the model"),
    "sparsity_target_epoch": with_attributes(_NUMBER,
                                             description="Index of the epoch from which the sparsity level "
                                                         "of the model will be equal to spatsity_target value"),
    "sparsity_freeze_epoch": with_attributes(_NUMBER,
                                             description="Index of the epoch from which the sparsity mask will "
                                                         "be frozen and no longer trained"),
    "update_per_optimizer_step": with_attributes(_BOOLEAN,
                                                 description="Whether the function-based sparsity level schedulers "
                                                             "should update the sparsity level after each optimizer "
                                                             "step instead of each epoch step."),
    "steps_per_epoch": with_attributes(_NUMBER,
                                       description="Number of optimizer steps in one epoch. Required to start proper "
                                                   " scheduling in the first training epoch if "
                                                   "'update_per_optimizer_step' is true"),
    "multistep_steps": with_attributes(_ARRAY_OF_NUMBERS,
                                       description="A list of scheduler steps at which to transition "
                                                   "to the next scheduled sparsity level (multistep "
                                                   "scheduler only)."),
    "multistep_sparsity_levels": with_attributes(_ARRAY_OF_NUMBERS,
                                                 description="Levels of sparsity to use at each step "
                                                             "of the scheduler as specified in the "
                                                             "'multistep_steps' attribute. The first"
                                                             "sparsity level will be applied "
                                                             "immediately, so the length of this list "
                                                             "should be larger than the length of the "
                                                             "'steps' by one."),
    "sparsity_level_setting_mode":with_attributes(_STRING,
                                                  description="The mode of sparsity level setting( "
                                                              "'global' - one sparsity level is set for all layer, "
                                                              "'local' - sparsity level is set per-layer.)"),
}

MAGNITUDE_SPARSITY_ALGO_NAME_IN_CONFIG = "magnitude_sparsity"
MAGNITUDE_SPARSITY_SCHEMA = {
    **BASIC_COMPRESSION_ALGO_SCHEMA,
    "properties": {
        "algorithm": {
            "const": MAGNITUDE_SPARSITY_ALGO_NAME_IN_CONFIG
        },
        "sparsity_init": with_attributes(_NUMBER,
                                         description="Initial value of the sparsity level applied to the "
                                                     "model"),
        "initializer": GENERIC_INITIALIZER_SCHEMA,
        "params":
            {
                "type": "object",
                "properties": {
                    **COMMON_SPARSITY_PARAM_PROPERTIES,
                    "weight_importance": with_attributes(_STRING,
                                                         description="Determines the way in which the weight values "
                                                                     "will be sorted after being aggregated in order "
                                                                     "to determine the sparsity threshold "
                                                                     "corresponding to a specific sparsity level. "
                                                                     "Either 'abs' or 'normed_abs'.",
                                                         default="normed_abs")
                },
                "additionalProperties": False
            },
        **COMMON_COMPRESSION_ALGORITHM_PROPERTIES
    },
    "additionalProperties": False
}

RB_SPARSITY_ALGO_NAME_IN_CONFIG = "rb_sparsity"
RB_SPARSITY_SCHEMA = {
    **BASIC_COMPRESSION_ALGO_SCHEMA,
    "properties": {
        "algorithm": {
            "const": RB_SPARSITY_ALGO_NAME_IN_CONFIG
        },
        "sparsity_init": with_attributes(_NUMBER,
                                         description="Initial value of the sparsity level applied to the "
                                                     "model"),
        "params":
            {
                "type": "object",
                "properties": COMMON_SPARSITY_PARAM_PROPERTIES,
                "additionalProperties": False
            },
        **COMMON_COMPRESSION_ALGORITHM_PROPERTIES
    },
    "additionalProperties": False
}

FILTER_PRUNING_ALGO_NAME_IN_CONFIG = 'filter_pruning'
FILTER_PRUNING_SCHEMA = {
    **BASIC_COMPRESSION_ALGO_SCHEMA,
    "properties": {
        "algorithm": {
            "const": FILTER_PRUNING_ALGO_NAME_IN_CONFIG
        },
        "initializer": GENERIC_INITIALIZER_SCHEMA,
        "pruning_init": with_attributes(_NUMBER,
                                        description="Initial value of the pruning level applied to the"
                                                    " model. 0.0 by default."),
        "params":
            {
                "type": "object",
                "properties": {
                    "schedule": with_attributes(_STRING,
                                                description="The type of scheduling to use for adjusting the target"
                                                            " pruning level. Either `exponential`, `exponential_with"
                                                            "_bias`,  or `baseline`, by default it is `baseline`"),
                    "pruning_target": with_attributes(_NUMBER,
                                                      description="Target value of the pruning level for the model."
                                                                  " 0.5 by default."),
                    "pruning_flops_target": with_attributes(_NUMBER,
                                                            description="Target value of the pruning level for model"
                                                                        " FLOPs."),
                    "num_init_steps": with_attributes(_NUMBER,
                                                      description="Number of epochs for model pretraining before"
                                                                  " starting filter pruning. 0 by default."),
                    "pruning_steps": with_attributes(_NUMBER,
                                                     description="Number of epochs during which the pruning rate is"
                                                                 " increased from `pruning_init` to `pruning_target`"
                                                                 " value."),
                    "weight_importance": with_attributes(_STRING,
                                                         description="The type of filter importance metric. Can be"
                                                                     " one of `L1`, `L2`, `geometric_median`."
                                                                     " `L2` by default."),
                    "all_weights": with_attributes(_BOOLEAN,
                                                   description="Whether to prune layers independently (choose filters"
                                                               " with the smallest importance in each layer separately)"
                                                               " or not. `False` by default.",
                                                   default=False),
                    "prune_first_conv": with_attributes(_BOOLEAN,
                                                        description="Whether to prune first Convolutional layers or"
                                                                    " not. First means that it is a convolutional layer"
                                                                    " such that there is a path from model input to "
                                                                    "this layer such that there are no other "
                                                                    "convolution operations on it. `False` by default.",
                                                        default=False
                                                        ),
                    "prune_last_conv": with_attributes(_BOOLEAN,
                                                       description="whether to prune last Convolutional layers or not."
                                                                   "  Last means that it is a Convolutional layer such"
                                                                   " that there is a path from this layer to the model"
                                                                   " output such that there are no other convolution"
                                                                   " operations on it. `False` by default. ",
                                                       default=False
                                                       ),
                    "prune_downsample_convs": with_attributes(_BOOLEAN,
                                                              description="whether to prune downsample Convolutional"
                                                                          " layers (with stride > 1) or not. `False`"
                                                                          " by default.",
                                                              default=False
                                                              ),
                    "prune_batch_norms": with_attributes(_BOOLEAN,
                                                         description="whether to nullifies parameters of Batch Norm"
                                                                     " layer corresponds to zeroed filters of"
                                                                     " convolution corresponding to this Batch Norm."
                                                                     " `False` by default.",
                                                         default=False
                                                         ),
                    "zero_grad": with_attributes(_BOOLEAN,
                                                 description="Whether to setting gradients corresponding to zeroed"
                                                             " filters to zero during training, `True` by default.",
                                                 default=True),

                },
                "additionalProperties": False,
            },
        **COMMON_COMPRESSION_ALGORITHM_PROPERTIES
    },
    "additionalProperties": False
}

ALL_SUPPORTED_ALGO_SCHEMA = [BINARIZATION_SCHEMA,
                             QUANTIZATION_SCHEMA,
                             CONST_SPARSITY_SCHEMA,
                             MAGNITUDE_SPARSITY_SCHEMA,
                             RB_SPARSITY_SCHEMA,
                             FILTER_PRUNING_SCHEMA]

REF_VS_ALGO_SCHEMA = {BINARIZATION_ALGO_NAME_IN_CONFIG: BINARIZATION_SCHEMA,
                      QUANTIZATION_ALGO_NAME_IN_CONFIG: QUANTIZATION_SCHEMA,
                      CONST_SPARSITY_ALGO_NAME_IN_CONFIG: CONST_SPARSITY_SCHEMA,
                      MAGNITUDE_SPARSITY_ALGO_NAME_IN_CONFIG: MAGNITUDE_SPARSITY_SCHEMA,
                      RB_SPARSITY_ALGO_NAME_IN_CONFIG: RB_SPARSITY_SCHEMA,
                      FILTER_PRUNING_ALGO_NAME_IN_CONFIG: FILTER_PRUNING_SCHEMA}

TARGET_DEVICE_SCHEMA = {
    "type": "string",
    "enum": ["ANY", "CPU", "GPU", "VPU", "TRIAL"]
}

ROOT_NNCF_CONFIG_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema",
    "type": "object",
    "properties": {
        "input_info": with_attributes(make_object_or_array_of_objects_schema(SINGLE_INPUT_INFO_SCHEMA),
                                      description="Required - describe the specifics of your model inputs here."
                                                  "This information is used to build the internal graph representation"
                                                  "that is leveraged for proper compression functioning, and for "
                                                  "exporting the compressed model to ONNX - a dummy tensor with a "
                                                  "corresponding shape and filler will be generated for each entry"
                                                  "and passed as a corresponding argument into the model's forward"
                                                  "method. Keywords can be specified for each entry - if left "
                                                  "unspecified, the dummy tensor will be passed as a positional arg."),
        "disable_shape_matching": with_attributes(_BOOLEAN,
                                                  description="Whether to enable strict input tensor"
                                                              "shape matching when building the internal graph"
                                                              "representation of the model. Set this to false if your"
                                                              "model inputs have any variable dimension other than "
                                                              "the 0-th (batch) dimension, or if any non-batch "
                                                              "dimension of the intermediate tensors in your model "
                                                              "execution flow depends on the input dimension,"
                                                              "otherwise the compression will most likely fail."),
        # Validation of each separate compression description schema occurs in a separate step.
        # This is required for better user feedback, since holistic schema validation is uninformative
        # if there is an error in one of the compression configs.
        "compression": make_object_or_array_of_objects_schema(BASIC_COMPRESSION_ALGO_SCHEMA),
        "target_device": with_attributes(TARGET_DEVICE_SCHEMA,
                                         description="The target device, the specificity of which will be taken into "
                                                     "account while compressing in order to obtain the best "
                                                     "performance for this type of device. The default 'ANY' means "
                                                     "compatible quantization supported by any HW. The parameter takes "
                                                     "values from the set ('CPU', 'GPU', 'VPU', 'ANY', 'TRIAL'). Set "
                                                     "this value to 'TRIAL' if you are going to use a custom "
                                                     "quantization schema. Optional."),
        "log_dir": with_attributes(_STRING,
                                   description="Log directory for NNCF-specific logging outputs"),
        "quantizer_setup_type": with_attributes(_STRING,
                                                description="Selects the mode of placement quantizers - either "
                                                            "'pattern_based' or 'propagation_based'. "
                                                            "In 'pattern_based' mode, the quantizers are placed "
                                                            "according to the accepted patterns (Each pattern is "
                                                            "a layer or a set of layers to be quantized). "
                                                            "In 'propagation_based' initially quantizers are placed "
                                                            "on all possible quantized layers and then the algorithm "
                                                            "their propagation is run from the bottom up. Also in "
                                                            "this mode it is possible to use hw config."),
    },
    "required": ["input_info"],
    "definitions": REF_VS_ALGO_SCHEMA,
    "dependencies": {
        "target_device": {
            "properties": {
                "quantizer_setup_type": {"const": "propagation_based"}}
        }
    }
}


def validate_single_compression_algo_schema(single_compression_algo_dict: Dict):
    """single_compression_algo_dict must conform to BASIC_COMPRESSION_ALGO_SCHEMA (and possibly has other
    algo-specific properties"""
    algo_name = single_compression_algo_dict["algorithm"]
    if algo_name not in REF_VS_ALGO_SCHEMA:
        raise jsonschema.ValidationError(
            "Incorrect algorithm name - must be one of ({})".format(", ".join(REF_VS_ALGO_SCHEMA.keys())))
    try:
        jsonschema.validate(single_compression_algo_dict, schema=REF_VS_ALGO_SCHEMA[algo_name])
    except Exception as e:
        import sys
        raise type(e)("For algorithm: '{}'\n".format(algo_name) + str(e)).with_traceback(sys.exc_info()[2])
