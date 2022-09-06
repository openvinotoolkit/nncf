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
from nncf.config.schemata.common.targeting import BATCHNORM_ADAPTATION_SCHEMA
from nncf.config.schemata.common.targeting import IGNORED_SCOPES_DESCRIPTION
from nncf.config.schemata.common.targeting import SCOPING_PROPERTIES
from nncf.config.schemata.common.targeting import TARGET_SCOPES_DESCRIPTION

QUANTIZER_CONFIG_PROPERTIES = {
    "mode": with_attributes(STRING,
                            description="Mode of quantization"),
    "bits": with_attributes(NUMBER,
                            description="Bitwidth to quantize to. It is intended for manual bitwidth setting. Can be "
                                        "overridden by the `bits` parameter from the `precision` initializer section. "
                                        "An error happens if it doesn't match a corresponding bitwidth constraints "
                                        "from the hardware configuration."),
    "signed": with_attributes(BOOLEAN,
                              description="Whether to use signed or unsigned input/output values for quantization."
                                          " If specified as unsigned and the input values during initialization have "
                                          "differing signs, will reset to performing signed quantization instead."),
    "per_channel": with_attributes(BOOLEAN,
                                   description="Whether to quantize inputs per channel (i.e. per 0-th dimension for "
                                               "weight quantization, and per 1-st dimension for activation "
                                               "quantization)")
}

UNIFIED_SCALE_OP_SCOPES_SPECIFIER_SCHEMA = {
    "type": "array",
    "items": ARRAY_OF_STRINGS
}

BASIC_RANGE_INIT_CONFIG_PROPERTIES = {
    "type": "object",
    "properties": {
        "num_init_samples": with_attributes(NUMBER,
                                            description="Number of samples from the training dataset to "
                                                        "consume as sample model inputs for purposes of "
                                                        "setting initial minimum and maximum quantization "
                                                        "ranges"),
        "type": with_attributes(STRING, description="Type of the initializer - determines which "
                                                    "statistics gathered during initialization will be "
                                                    "used to initialize the quantization ranges"),
        "params": {
            "type": "object",
            "properties": {
                "min_percentile": with_attributes(NUMBER,
                                                  description="For 'percentile' type - specify the percentile of "
                                                              "input value histograms to be set as the initial "
                                                              "value for minimum quantizer input"),
                "max_percentile": with_attributes(NUMBER,
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
        "target_quantizer_group": with_attributes(STRING, description="The target group of quantizers for which "
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
        "batchnorm_adaptation": BATCHNORM_ADAPTATION_SCHEMA,
        **RANGE_INIT_CONFIG_PROPERTIES["initializer"]["properties"],
        "precision":
            {
                "type": "object",
                "properties": {
                    "type": with_attributes(STRING,
                                            description="Type of precision initialization."),
                    "bits": with_attributes(ARRAY_OF_NUMBERS,
                                            description="A list of bitwidth to choose from when performing precision "
                                                        "initialization. Overrides bits constraints specified in "
                                                        "`weight` and `activation` sections",
                                            examples=[[4, 8]]),
                    "num_data_points": with_attributes(NUMBER,
                                                       description="Number of data points to iteratively estimate "
                                                                   "Hessian trace, 1000 by default."),
                    "iter_number": with_attributes(NUMBER,
                                                   description="Maximum number of iterations of Hutchinson algorithm "
                                                               "to Estimate Hessian trace, 500 by default"),
                    "tolerance": with_attributes(NUMBER,
                                                 description="Minimum relative tolerance for stopping the Hutchinson "
                                                             "algorithm. It's calculated  between mean average trace "
                                                             "from previous iteration and current one. 1e-5 by default"
                                                             "bitwidth_per_scope"),
                    "compression_ratio": with_attributes(NUMBER,
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
                    "eval_subset_ratio": with_attributes(NUMBER,
                                                         description="The desired ratio of dataloader to be iterated "
                                                                     "during each search iteration of AutoQ precision "
                                                                     "initialization. Specifically, this ratio applies "
                                                                     "to the registered autoq_eval_loader via "
                                                                     "register_default_init_args.",
                                                         default=1.0),
                    "warmup_iter_number": with_attributes(NUMBER,
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
                                    NUMBER,
                                    STRING
                                ],
                            "description": "A tuple of a bitwidth and a scope of the quantizer to assign the "
                                           "bitwidth to."
                        },
                        "description": "Manual settings for the quantizer bitwidths. Scopes are used to identify "
                                       "the quantizers."
                    },
                    "traces_per_layer_path": with_attributes(STRING,
                                                             description="Path to serialized PyTorch Tensor with "
                                                                         "average Hessian traces per quantized modules."
                                                                         " It can be used to accelerate mixed precision"
                                                                         "initialization by using average Hessian "
                                                                         "traces from previous run of HAWQ algorithm."),
                    "dump_init_precision_data": with_attributes(BOOLEAN,
                                                                description="Whether to dump data related to Precision "
                                                                            "Initialization algorithm. "
                                                                            "HAWQ dump includes bitwidth graph,"
                                                                            " average traces and different plots. "
                                                                            "AutoQ dump includes DDPG agent "
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
STAGED_QUANTIZATION_PARAMS = {
    "params": {
        "type": "object",
        "properties": {
            "batch_multiplier": with_attributes(NUMBER,
                                                description="Gradients will be accumulated for this number of "
                                                            "batches before doing a 'backward' call. Increasing "
                                                            "this may improve training quality, since binarized "
                                                            "networks exhibit noisy gradients requiring larger "
                                                            "batch sizes than could be accomodated by GPUs"),
            "activations_quant_start_epoch": with_attributes(NUMBER,
                                                             description="Epoch to start binarizing activations"),
            "weights_quant_start_epoch": with_attributes(NUMBER,
                                                         description="Epoch to start binarizing weights"),
            "lr_poly_drop_start_epoch": with_attributes(NUMBER,
                                                        description="Epoch to start dropping the learning rate"),
            "lr_poly_drop_duration_epochs": with_attributes(NUMBER,
                                                            description="Duration, in epochs, of the learning "
                                                                        "rate dropping process."),
            "disable_wd_start_epoch": with_attributes(NUMBER,
                                                      description="Epoch to disable weight decay in the optimizer"),
            "base_lr": with_attributes(NUMBER, description="Initial value of learning rate"),
            "base_wd": with_attributes(NUMBER, description="Initial value of weight decay"),
        },
        "additionalProperties": False
    }
}
QUANTIZATION_ALGO_NAME_IN_CONFIG = "quantization"
QUANTIZATION_PRESETS_SCHEMA = {
    "type": "string",
    "enum": ["performance", "mixed"]
}

QUANTIZER_GROUP_PROPERTIES = {
    **QUANTIZER_CONFIG_PROPERTIES,
    "ignored_scopes": with_attributes(make_object_or_array_of_objects_schema(STRING),
                                      description=IGNORED_SCOPES_DESCRIPTION),
    "target_scopes": with_attributes(make_object_or_array_of_objects_schema(STRING),
                                     description=TARGET_SCOPES_DESCRIPTION),
    "logarithm_scale": with_attributes(BOOLEAN,
                                       description="Whether to use log of scale as optimized parameter"
                                                   " instead of scale itself."),
}

WEIGHTS_GROUP_SCHEMA = {
    "type": "object",
    "properties": {
        **QUANTIZER_GROUP_PROPERTIES,
    },
    "additionalProperties": False
}
ACTIVATIONS_GROUP_SCHEMA = {
    "type": "object",
    "properties": {
        **QUANTIZER_GROUP_PROPERTIES,
        "unified_scale_ops": with_attributes(UNIFIED_SCALE_OP_SCOPES_SPECIFIER_SCHEMA,
                                             description="Specifies operations in the model which will share "
                                                         "the same quantizer module for activations. This "
                                                         "is helpful in case one and the same quantizer scale "
                                                         "is required for inputs to the same operation. Each "
                                                         "sub-array will define a group of model operation "
                                                         "inputs that have to share a single actual "
                                                         "quantization module, each entry in this subarray "
                                                         "should correspond to exactly one node in the NNCF "
                                                         "graph and the groups should not overlap. The final "
                                                         "quantizer for each sub-array will be associated "
                                                         "with the first element of this sub-array.")
    },
    "additionalProperties": False
}
QUANTIZATION_SCHEMA = {
    **BASIC_COMPRESSION_ALGO_SCHEMA,
    "properties": {
        "algorithm": {
            "const": QUANTIZATION_ALGO_NAME_IN_CONFIG
        },
        **COMPRESSION_LR_MULTIPLIER_PROPERTY,
        "initializer": QUANTIZATION_INITIALIZER_SCHEMA,
        "weights": with_attributes(WEIGHTS_GROUP_SCHEMA,
                                   description="Constraints to be applied to model weights quantization only."),
        "activations": with_attributes(ACTIVATIONS_GROUP_SCHEMA,
                                       description="Constraints to be applied to model activations quantization only."),
        "quantize_inputs": with_attributes(BOOLEAN,
                                           description="Whether the model inputs should be immediately quantized prior "
                                                       "to any other model operations.",
                                           default=True),
        "quantize_outputs": with_attributes(BOOLEAN,
                                            description="Whether the model outputs should be additionally quantized.",
                                            default=False),
        "preset": with_attributes(QUANTIZATION_PRESETS_SCHEMA,
                                  description="The preset defines the quantization schema for weights and activations. "
                                              "The parameter takes values 'performance' or 'mixed'. The mode "
                                              "'performance' defines symmetric weights and activations. The mode "
                                              "'mixed' defines symmetric 'weights' and asymmetric activations."),
        "scope_overrides": {
            "type": "object",
            "properties": {
                "weights": {
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
                },
                "activations": {
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
                }
            },
            "additionalProperties": False,
            "description": "This option is used to specify overriding quantization constraints for specific scope,"
                           "e.g. in case you need to quantize a single operation differently than the rest of the "
                           "model."
        },
        "export_to_onnx_standard_ops": with_attributes(BOOLEAN,
                                                       description="Determines how should the additional quantization "
                                                                   "operations be exported into the ONNX format. Set "
                                                                   "this to false for export to OpenVINO-supported "
                                                                   "FakeQuantize ONNX, or to true for export to ONNX "
                                                                   "standard QuantizeLinear-DequantizeLinear "
                                                                   "node pairs (8-bit quantization only in the latter "
                                                                   "case). Default: false"),
        "overflow_fix": with_attributes(STRING,
                                        description="Option controls whether to apply the overflow "
                                                    "issue fix for the appropriate NNCF config or not. "
                                                    "If set to 'disable', the fix will not be applied. "
                                                    "If set to 'enable' or 'first_layer_only', "
                                                    "while appropriate target_devices are chosen, "
                                                    "the fix will be applied to the all layers or to the first"
                                                    "convolutional layer respectively."),
        **STAGED_QUANTIZATION_PARAMS,
        **SCOPING_PROPERTIES,
    },
    "additionalProperties": False
}
