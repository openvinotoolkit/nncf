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
from nncf.config.definitions import ONLINE_DOCS_ROOT
from nncf.config.definitions import QUANTIZATION_ALGO_NAME_IN_CONFIG
from nncf.config.schemata.basic import ARRAY_OF_NUMBERS
from nncf.config.schemata.basic import ARRAY_OF_STRINGS
from nncf.config.schemata.basic import BOOLEAN
from nncf.config.schemata.basic import NUMBER
from nncf.config.schemata.basic import STRING
from nncf.config.schemata.basic import annotated_enum
from nncf.config.schemata.basic import with_attributes
from nncf.config.schemata.common.compression import BASIC_COMPRESSION_ALGO_SCHEMA
from nncf.config.schemata.common.compression import COMPRESSION_LR_MULTIPLIER_PROPERTY
from nncf.config.schemata.common.initialization import BATCHNORM_ADAPTATION_SCHEMA
from nncf.config.schemata.common.targeting import SCOPING_PROPERTIES
from nncf.config.schemata.defaults import ACTIVATIONS_QUANT_START_EPOCH
from nncf.config.schemata.defaults import AUTOQ_EVAL_SUBSET_RATIO
from nncf.config.schemata.defaults import AUTOQ_WARMUP_ITER_NUMBER
from nncf.config.schemata.defaults import HAWQ_DUMP_INIT_PRECISION_DATA
from nncf.config.schemata.defaults import HAWQ_ITER_NUMBER
from nncf.config.schemata.defaults import HAWQ_NUM_DATA_POINTS
from nncf.config.schemata.defaults import HAWQ_TOLERANCE
from nncf.config.schemata.defaults import LR_POLY_DURATION_EPOCHS
from nncf.config.schemata.defaults import MAX_PERCENTILE
from nncf.config.schemata.defaults import MIN_PERCENTILE
from nncf.config.schemata.defaults import NUM_INIT_SAMPLES
from nncf.config.schemata.defaults import PRECISION_INIT_BITWIDTHS
from nncf.config.schemata.defaults import QUANTIZATION_BITS
from nncf.config.schemata.defaults import QUANTIZATION_EXPORT_TO_ONNX_STANDARD_OPS
from nncf.config.schemata.defaults import QUANTIZATION_LOGARITHM_SCALE
from nncf.config.schemata.defaults import QUANTIZATION_OVERFLOW_FIX
from nncf.config.schemata.defaults import QUANTIZATION_PER_CHANNEL
from nncf.config.schemata.defaults import QUANTIZATION_PRESET
from nncf.config.schemata.defaults import QUANTIZE_INPUTS
from nncf.config.schemata.defaults import QUANTIZE_OUTPUTS
from nncf.config.schemata.defaults import STAGED_QUANTIZATION_BASE_LR
from nncf.config.schemata.defaults import STAGED_QUANTIZATION_BASE_WD
from nncf.config.schemata.defaults import WEIGHTS_QUANT_START_EPOCH

QUANTIZER_MODES = ["symmetric", "asymmetric"]

QUANTIZER_CONFIG_PROPERTIES = {
    "mode": with_attributes(
        STRING,
        description=f"Mode of quantization. "
        f"See [Quantization.md]"
        f"({ONLINE_DOCS_ROOT}"
        f"/docs/compression_algorithms/Quantization.md#symmetric-quantization) for "
        f"more details.",
        enum=QUANTIZER_MODES,
    ),
    "bits": with_attributes(
        NUMBER,
        description="Bitwidth to quantize to. It is intended for manual bitwidth setting. Can be "
        "overridden by the `bits` parameter from the `precision` initializer section. "
        "An error occurs if it doesn't match a corresponding bitwidth constraints "
        "from the hardware configuration.",
        default=QUANTIZATION_BITS,
    ),
    "signed": with_attributes(
        BOOLEAN,
        description="Whether to use signed or unsigned input/output values for quantization. "
        "`true` will force the quantization to support signed values, `false` will "
        "force the quantization to only support input values with one and the same "
        "sign, and leaving this value unspecified (default) means relying "
        "on the initialization statistics to determine best approach. \n"
        "Note: If set to `false`, but the input "
        "values have differing signs during initialization, signed quantization "
        "will be performed instead.",
    ),
    "per_channel": with_attributes(
        BOOLEAN,
        description="Whether to quantize inputs of this quantizer "
        "per each channel of input tensor (per 0-th dimension for "
        "weight quantization, and per 1-st dimension for activation "
        "quantization).",
        default=QUANTIZATION_PER_CHANNEL,
    ),
}

UNIFIED_SCALE_OP_SCOPES_SPECIFIER_SCHEMA = {"type": "array", "items": ARRAY_OF_STRINGS}

RANGE_INIT_TYPES_VS_DESCRIPTIONS = {
    "mixed_min_max": "Minimum quantizer range initialized using minima of per-channel minima of "
    "the tensor to be quantized, maximum quantizer range initialized using "
    "maxima of per-channel maxima of the tensor to be quantized. Offline.",
    "min_max": "Minimum quantizer range initialized using global minimum of values in "
    "the tensor to be quantized, maximum quantizer range initialized using global maxima of the same"
    "values. Online.",
    "mean_min_max": "Minimum quantizer range initialized using averages (across every single initialization sample) "
    "of minima of values in "
    "the tensor to be quantized, maximum quantizer range initialized using maxima respectively. "
    "Offline.",
    "threesigma": "Quantizer minimum and maximum ranges set to be equal to +- 3 median absolute deviation from the "
    "median of the observed values in the tensor to be quantized. Offline.",
    "percentile": "Quantizer minimum and maximum ranges set to be equal to specified percentiles of the the observed "
    "values (across the entire initialization sample set) in the tensor to be quantized. Offline.",
    "mean_percentile": "Quantizer minimum and maximum ranges set to be equal to averaged (across every single "
    "initialization sample) specified percentiles of the the observed values in the tensor to be "
    "quantized. Offline.",
}

BASIC_RANGE_INIT_CONFIG_PROPERTIES = {
    "type": "object",
    "title": "global_range_init_configuration",
    "properties": {
        "num_init_samples": with_attributes(
            NUMBER,
            description="Number of samples from the training dataset to "
            "consume as sample model inputs for purposes of "
            "setting initial minimum and maximum quantization "
            "ranges.",
            default=NUM_INIT_SAMPLES,
        ),
        "type": with_attributes(
            annotated_enum(RANGE_INIT_TYPES_VS_DESCRIPTIONS),
            description="Type of the initializer - determines which "
            "statistics gathered during initialization will be "
            "used to initialize the quantization ranges.\n"
            "'Online' initializers do not have to "
            "store intermediate statistics in memory, while 'offline' do. Increasing the number "
            "of initialization samples for 'offline' initialization types will increase RAM "
            "overhead of applying NNCF to the model.\n"
            "Depending on whether the quantizer is configured to be per-tensor or per-channel, the "
            "statistics will be collected either on the basis of the set of the entire tensor values, or "
            "these will be collected and applied separately for each channel value subset.",
            default="mixed_min_max",
        ),
        "params": {
            "type": "object",
            "description": "Type-specific parameters of the initializer.",
            "properties": {
                "min_percentile": with_attributes(
                    NUMBER,
                    description="For 'percentile' and 'mean_percentile' types - specify the percentile "
                    "of input value histograms to be set as the initial "
                    "value for the quantizer input minimum.",
                    default=MIN_PERCENTILE,
                ),
                "max_percentile": with_attributes(
                    NUMBER,
                    description="For 'percentile' and 'mean_percentile' types - specify the percentile "
                    "of input value histograms to be set as the initial "
                    "value for the quantizer input maximum.",
                    default=MAX_PERCENTILE,
                ),
            },
        },
    },
    "additionalProperties": False,
}
PER_LAYER_RANGE_INIT_CONFIG_PROPERTIES = {
    "type": "object",
    "properties": {
        **BASIC_RANGE_INIT_CONFIG_PROPERTIES["properties"],  # type: ignore[dict-item]
        **SCOPING_PROPERTIES,
        "target_quantizer_group": with_attributes(
            STRING,
            description="The target group of quantizers for which "
            "the specified type of range initialization will "
            "be applied. If unspecified, then the range "
            "initialization of the given type will be applied to all quantizers.",
            enum=["activations", "weights"],
        ),
    },
}
RANGE_INIT_CONFIG_PROPERTIES = {
    "initializer": {
        "type": "object",
        "properties": {
            "range": {
                "description": "This initializer performs forward runs of the model to be quantized using "
                "samples from a user-supplied data loader to gather activation and weight "
                "tensor statistics within the network and use these to set up initial range parameters "
                "for quantizers.",
                "oneOf": [
                    BASIC_RANGE_INIT_CONFIG_PROPERTIES,
                    {
                        "type": "array",
                        "title": "per_layer_range_init_configuration",
                        "items": PER_LAYER_RANGE_INIT_CONFIG_PROPERTIES,
                    },
                ],
            },
        },
        "additionalProperties": False,
    },
}
BITWIDTH_ASSIGNMENT_MODE_SCHEMA = {
    "type": "string",
    "enum": ["strict", "liberal"],
    "default": "liberal",
    "description": "The mode for assignment bitwidth to activation quantizers. In the 'strict' mode,"
    "a group of quantizers that feed their output to one and more "
    "same modules as input (weight quantizers count as well) will have the same bitwidth in the "
    "'liberal' mode allows different precisions within the group.\n"
    "Bitwidth is assigned based on hardware constraints. If multiple "
    "variants are possible, the minimal compatible bitwidth is chosen.",
}

PRECISION_INIT_TYPES_VS_DESCRIPTION = {
    "hawq": f"Applies HAWQ algorithm to determine best bitwidths for each quantizer using a Hessian"
    f"calculation approach. For more details see "
    f"[Quantization.md]({ONLINE_DOCS_ROOT}/docs/compression_algorithms/Quantization.md#hawq)",
    "autoq": f"Applies AutoQ algorithm to determine best bitwidths for each quantizer using reinforcement learning. "
    f"For more details see "
    f"[Quantization.md]({ONLINE_DOCS_ROOT}/docs/compression_algorithms/Quantization.md#autoq)",
    "manual": "Allows to manually specify via following config options the exact bitwidth "
    "for each quantizer location. ",
}


PRECISION_INITIALIZER_SCHEMA = {
    "type": "object",
    "description": "This initializer performs advanced selection of bitwidth per each quantizer "
    "location, trying to achieve the best tradeoff between performance and "
    "quality of the resulting model.",
    "properties": {
        "required": ["type"],
        "type": with_attributes(
            annotated_enum(PRECISION_INIT_TYPES_VS_DESCRIPTION), description="Type of precision initialization."
        ),
        "bits": with_attributes(
            ARRAY_OF_NUMBERS,
            description="A list of bitwidth to choose from when performing precision "
            "initialization. Overrides bits constraints specified in "
            "`weight` and `activation` sections.",
            examples=[[4, 8]],
            default=PRECISION_INIT_BITWIDTHS,
        ),
        "num_data_points": with_attributes(
            NUMBER,
            description="Number of data points to iteratively estimate Hessian trace.",
            default=HAWQ_NUM_DATA_POINTS,
        ),
        "iter_number": with_attributes(
            NUMBER,
            description="Maximum number of iterations of Hutchinson algorithm to Estimate Hessian trace.",
            default=HAWQ_ITER_NUMBER,
        ),
        "tolerance": with_attributes(
            NUMBER,
            description="Minimum relative tolerance for stopping the Hutchinson "
            "algorithm. It's calculated  between mean average trace "
            "from the previous iteration and the current one.",
            default=HAWQ_TOLERANCE,
        ),
        "compression_ratio": with_attributes(
            NUMBER,
            description="For the `hawq` mode:\n"
            "The desired ratio between bit complexity of "
            "a fully INT8 model and a mixed-precision lower-bit "
            "one. On precision initialization stage the HAWQ "
            "algorithm chooses the most accurate "
            "mixed-precision configuration with a ratio no less "
            "than the specified. Bit complexity of the model "
            "is a sum of bit complexities for each quantized "
            "layer, which are a multiplication of FLOPS for "
            "the layer by the number of bits for its "
            "quantization.\n"
            "For the `autoq` mode:\n"
            "The target model size after quantization, relative to total "
            "parameters size in FP32. E.g. a uniform INT8-quantized model "
            "would have a `compression_ratio` equal to 0.25,"
            "and a uniform INT4-quantized model would have "
            "`compression_ratio` equal to 0.125.",
        ),
        "eval_subset_ratio": with_attributes(
            NUMBER,
            description="The desired ratio of dataloader to be iterated "
            "during each search iteration of AutoQ precision "
            "initialization. Specifically, this ratio applies "
            "to the registered autoq_eval_loader via "
            "register_default_init_args.",
            default=AUTOQ_EVAL_SUBSET_RATIO,
        ),
        "warmup_iter_number": with_attributes(
            NUMBER,
            description="The number of random policy at the beginning of "
            "of AutoQ precision initialization to populate "
            "replay buffer with experiences. This key is "
            "meant for internal testing use. Users need not "
            "to configure.",
            default=AUTOQ_WARMUP_ITER_NUMBER,
        ),
        "bitwidth_per_scope": {
            "type": "array",
            "items": {
                "type": "array",
                "items": [NUMBER, STRING],
                "description": "A tuple of a bitwidth and a scope of the quantizer to assign the bitwidth to.",
            },
            "description": "Manual settings for the quantizer bitwidths. Scopes are used to identify "
            "the quantizers.",
            "examples": [
                [
                    [2, "ResNet/NNCFConv2d[conv1]/conv2d_0|WEIGHT"],
                    [8, "ResNet/ReLU[relu]/relu__0|OUTPUT"],
                ]
            ],
        },
        "traces_per_layer_path": with_attributes(
            STRING,
            description="Path to serialized PyTorch Tensor with "
            "average Hessian traces per quantized modules. "
            "It can be used to accelerate mixed precision "
            "initialization by using average Hessian "
            "traces from previous run of HAWQ algorithm.",
        ),
        "dump_init_precision_data": with_attributes(
            BOOLEAN,
            description="Whether to dump data related to Precision "
            "Initialization algorithm. "
            "HAWQ dump includes bitwidth graph,"
            " average traces and different plots. "
            "AutoQ dump includes DDPG agent "
            "learning trajectory in tensorboard and "
            "mixed-precision environment metadata.",
            default=HAWQ_DUMP_INIT_PRECISION_DATA,
        ),
        "bitwidth_assignment_mode": BITWIDTH_ASSIGNMENT_MODE_SCHEMA,
    },
    "additionalProperties": False,
}

QUANTIZATION_INITIALIZER_SCHEMA = {
    "type": "object",
    "description": "Specifies the kind of pre-training initialization used for the quantization algorithm.\n"
    "Some kind of initialization is usually required so that the trainable quantization "
    "parameters have a better chance to get fine-tuned to values that result in good accuracy.",
    "properties": {
        "batchnorm_adaptation": BATCHNORM_ADAPTATION_SCHEMA,
        **RANGE_INIT_CONFIG_PROPERTIES["initializer"]["properties"],  # type: ignore[dict-item]
        "precision": PRECISION_INITIALIZER_SCHEMA,
    },
    "additionalProperties": False,
}
STAGED_QUANTIZATION_PARAMS = {
    "params": {
        "type": "object",
        "description": "Configures the staged quantization compression scheduler for the quantization algorithm. "
        "The quantizers will not be applied until a given epoch count is reached.",
        "properties": {
            "batch_multiplier": with_attributes(
                NUMBER,
                description="Gradients will be accumulated for this number of "
                "batches before doing a 'backward' call. Increasing "
                "this may improve training quality, since binarized "
                "networks exhibit noisy gradients and their training "
                "requires larger batch sizes than could be accommodated "
                "by GPUs.",
            ),
            "activations_quant_start_epoch": with_attributes(
                NUMBER,
                description="A zero-based index of the epoch, upon reaching which "
                "the activations will start to be quantized.",
                default=ACTIVATIONS_QUANT_START_EPOCH,
            ),
            "weights_quant_start_epoch": with_attributes(
                NUMBER,
                description="Epoch index upon which the weights will start to be quantized.",
                default=WEIGHTS_QUANT_START_EPOCH,
            ),
            "lr_poly_drop_start_epoch": with_attributes(
                NUMBER,
                description="Epoch index upon which the learning rate will "
                "start to be dropped. If unspecified, "
                "learning rate will not be dropped.",
            ),
            "lr_poly_drop_duration_epochs": with_attributes(
                NUMBER,
                description="Duration, in epochs, of the learning rate dropping process.",
                default=LR_POLY_DURATION_EPOCHS,
            ),
            "disable_wd_start_epoch": with_attributes(
                NUMBER,
                description="Epoch to disable weight decay in the optimizer. "
                "If unspecified, weight decay will not be disabled.",
            ),
            "base_lr": with_attributes(
                NUMBER, description="Initial value of learning rate.", default=STAGED_QUANTIZATION_BASE_LR
            ),
            "base_wd": with_attributes(
                NUMBER, description="Initial value of weight decay.", default=STAGED_QUANTIZATION_BASE_WD
            ),
        },
        "additionalProperties": False,
    }
}

QUANTIZATION_PRESETS_SCHEMA = {"type": "string", "enum": ["performance", "mixed"]}

QUANTIZER_GROUP_PROPERTIES = {
    **QUANTIZER_CONFIG_PROPERTIES,
    **SCOPING_PROPERTIES,
    "logarithm_scale": with_attributes(
        BOOLEAN,
        description="Whether to use log of scale as the optimization parameter "
        "instead of the scale itself. This serves as an optional "
        "regularization opportunity for training quantizer scales.",
        default=QUANTIZATION_LOGARITHM_SCALE,
    ),
}

WEIGHTS_GROUP_SCHEMA = {
    "type": "object",
    "properties": {
        **QUANTIZER_GROUP_PROPERTIES,
    },
    "additionalProperties": False,
}
ACTIVATIONS_GROUP_SCHEMA = {
    "type": "object",
    "properties": {
        **QUANTIZER_GROUP_PROPERTIES,
        "unified_scale_ops": with_attributes(
            UNIFIED_SCALE_OP_SCOPES_SPECIFIER_SCHEMA,
            description="Specifies operations in the model which will share "
            "the same quantizer module for activations. This "
            "is helpful in case one and the same quantizer scale "
            "is required for each input of this operation. Each "
            "sub-array will define a group of model operation "
            "inputs that have to share a single actual "
            "quantization module, each entry in this subarray "
            "should correspond to exactly one node in the NNCF "
            "graph and the groups should not overlap. The final "
            "quantizer for each sub-array will be associated "
            "with the first element of this sub-array.",
        ),
    },
    "additionalProperties": False,
}

OVERFLOW_FIX_OPTIONS = ["enable", "disable", "first_layer_only"]

QUANTIZATION_SCHEMA = {
    **BASIC_COMPRESSION_ALGO_SCHEMA,
    "description": f"Applies quantization on top of the input model, simulating future low-precision execution "
    f"specifics, and selects the quantization layout and parameters to strive for the best possible "
    f"quantized model accuracy and performance. "
    f"\nSee [Quantization.md]"
    f"({ONLINE_DOCS_ROOT}"
    f"/docs/compression_algorithms/Quantization.md) and the rest of this schema for "
    f"more details and parameters.",
    "properties": {
        "algorithm": {"const": QUANTIZATION_ALGO_NAME_IN_CONFIG},
        "initializer": QUANTIZATION_INITIALIZER_SCHEMA,
        "preset": with_attributes(
            QUANTIZATION_PRESETS_SCHEMA,
            description="The preset defines the quantization schema for weights and activations. "
            "The 'performance' mode sets up symmetric weight and activation "
            "quantizers. The 'mixed' mode utilizes symmetric weight quantization and "
            "asymmetric activation quantization.",
            default=QUANTIZATION_PRESET,
        ),
        "quantize_inputs": with_attributes(
            BOOLEAN,
            description="Whether the model inputs should be immediately quantized prior "
            "to any other model operations.",
            default=QUANTIZE_INPUTS,
        ),
        "quantize_outputs": with_attributes(
            BOOLEAN, description="Whether the model outputs should be additionally quantized.", default=QUANTIZE_OUTPUTS
        ),
        "weights": with_attributes(
            WEIGHTS_GROUP_SCHEMA, description="Constraints to be applied to model weights quantization only."
        ),
        "activations": with_attributes(
            ACTIVATIONS_GROUP_SCHEMA, description="Constraints to be applied to model activations quantization only."
        ),
        "scope_overrides": {
            "type": "object",
            "description": "This option is used to specify overriding quantization constraints for specific scope,"
            "e.g. in case you need to quantize a single operation differently than the rest of the "
            "model. Any other automatic or group-wise settings will be overridden.",
            "examples": [
                {
                    "weights": {
                        "QuantizeOutputsTestModel/NNCFConv2d[conv5]/conv2d_0": {
                            "mode": "asymmetric",
                        },
                        "activations": {
                            "{re}.*conv_first.*": {"mode": "asymmetric"},
                            "{re}.*conv_second.*": {"mode": "symmetric"},
                        },
                    }
                }
            ],
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
                            "additionalProperties": False,
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
                            "additionalProperties": False,
                        },
                    },
                },
            },
            "additionalProperties": False,
        },
        "export_to_onnx_standard_ops": with_attributes(
            BOOLEAN,
            description="[Deprecated] Determines how should the additional quantization "
            "operations be exported into the ONNX format. Set "
            "this to true to export to ONNX "
            "standard QuantizeLinear-DequantizeLinear "
            "node pairs (8-bit quantization only) or to false "
            "to export to OpenVINO-supported FakeQuantize ONNX"
            "(all quantization settings supported).",
            default=QUANTIZATION_EXPORT_TO_ONNX_STANDARD_OPS,
        ),
        "overflow_fix": with_attributes(
            STRING,
            description="This option controls whether to apply the overflow "
            "issue fix for the appropriate NNCF config or not. "
            "If set to `disable`, the fix will not be applied. "
            "If set to `enable` or `first_layer_only`, "
            "while appropriate target_devices are chosen, "
            "the fix will be applied to all layers or to the first "
            "convolutional layer respectively.",
            enum=OVERFLOW_FIX_OPTIONS,
            default=QUANTIZATION_OVERFLOW_FIX,
        ),
        **STAGED_QUANTIZATION_PARAMS,
        **SCOPING_PROPERTIES,
        **COMPRESSION_LR_MULTIPLIER_PROPERTY,
    },
    "additionalProperties": False,
}
