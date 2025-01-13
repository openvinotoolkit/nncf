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

import logging
from typing import Any, Dict

import jsonschema

from nncf.config.definitions import ALGO_NAME_VS_README_URL
from nncf.config.definitions import CONST_SPARSITY_ALGO_NAME_IN_CONFIG
from nncf.config.definitions import FILTER_PRUNING_ALGO_NAME_IN_CONFIG
from nncf.config.definitions import KNOWLEDGE_DISTILLATION_ALGO_NAME_IN_CONFIG
from nncf.config.definitions import MAGNITUDE_SPARSITY_ALGO_NAME_IN_CONFIG
from nncf.config.definitions import QUANTIZATION_ALGO_NAME_IN_CONFIG
from nncf.config.definitions import RB_SPARSITY_ALGO_NAME_IN_CONFIG
from nncf.config.definitions import SCHEMA_VISUALIZATION_URL
from nncf.config.schemata.accuracy_aware import ACCURACY_AWARE_MODES_VS_SCHEMA
from nncf.config.schemata.accuracy_aware import ACCURACY_AWARE_TRAINING_SCHEMA
from nncf.config.schemata.algo.const_sparsity import CONST_SPARSITY_SCHEMA
from nncf.config.schemata.algo.filter_pruning import FILTER_PRUNING_SCHEMA
from nncf.config.schemata.algo.knowledge_distillation import KNOWLEDGE_DISTILLATION_SCHEMA
from nncf.config.schemata.algo.magnitude_sparsity import MAGNITUDE_SPARSITY_SCHEMA
from nncf.config.schemata.algo.quantization import QUANTIZATION_SCHEMA
from nncf.config.schemata.algo.rb_sparsity import RB_SPARSITY_SCHEMA
from nncf.config.schemata.basic import ARRAY_OF_NUMBERS
from nncf.config.schemata.basic import BOOLEAN
from nncf.config.schemata.basic import STRING
from nncf.config.schemata.basic import make_object_or_array_of_objects_schema
from nncf.config.schemata.basic import with_attributes
from nncf.config.schemata.common.compression import COMPRESSION_LR_MULTIPLIER_PROPERTY
from nncf.config.schemata.defaults import TARGET_DEVICE
from nncf.config.schemata.experimental_schema import EXPERIMENTAL_REF_VS_ALGO_SCHEMA

logger = logging.getLogger("nncf")

REF_VS_ALGO_SCHEMA = {
    QUANTIZATION_ALGO_NAME_IN_CONFIG: QUANTIZATION_SCHEMA,
    FILTER_PRUNING_ALGO_NAME_IN_CONFIG: FILTER_PRUNING_SCHEMA,
    MAGNITUDE_SPARSITY_ALGO_NAME_IN_CONFIG: MAGNITUDE_SPARSITY_SCHEMA,
    RB_SPARSITY_ALGO_NAME_IN_CONFIG: RB_SPARSITY_SCHEMA,
    KNOWLEDGE_DISTILLATION_ALGO_NAME_IN_CONFIG: KNOWLEDGE_DISTILLATION_SCHEMA,
    CONST_SPARSITY_ALGO_NAME_IN_CONFIG: CONST_SPARSITY_SCHEMA,
    **EXPERIMENTAL_REF_VS_ALGO_SCHEMA,
}


SINGLE_INPUT_INFO_SCHEMA = {
    "type": "object",
    "properties": {
        "sample_size": with_attributes(
            ARRAY_OF_NUMBERS,
            description="Shape of the tensor expected as input to the model.",
            examples=[[1, 3, 224, 224]],
        ),
        "type": with_attributes(STRING, description="Data type of the model input tensor."),
        "filler": with_attributes(
            STRING,
            description="Determines what the tensor will be filled with when passed to the model"
            " during tracing and exporting.",
        ),
        "keyword": with_attributes(
            STRING,
            description="Keyword to be used when passing the tensor to the model's 'forward' method - "
            "leave unspecified to pass the corresponding argument as a positional arg.",
        ),
    },
    "additionalProperties": False,
}

TARGET_DEVICE_SCHEMA = {"type": "string", "enum": ["ANY", "CPU", "GPU", "NPU", "TRIAL", "CPU_SPR"]}


NNCF_CONFIG_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema",
    "title": "NNCF configuration file schema",
    "description": "The NNCF configuration file follows the JSON format and is the primary way to configure "
    "the result of NNCF application to a given user model. This configuration file is "
    "loaded into the `NNCFConfig` object by the user at runtime, after which the `NNCFConfig` "
    "is passed to the NNCF functions that perform actual compression or "
    "preparations for compression-aware training. \n\n"
    "The NNCF JSON configuration file is usually set up on a per-model, per-compression use case "
    "basis to contain:\n"
    "- a description of one or more compression algorithms to be applied to the model\n"
    "- the configuration parameters for each of the chosen algorithms\n"
    "- additional settings depending on the NNCF use case or integration scenario, e.g. specifying "
    "parameters for accuracy-aware training, or specifying model input shape for frameworks "
    "that do not have this data encapsulated in the model object in general such as PyTorch)\n"
    "and other parameters, the list of which may extend with the ongoing development of NNCF.\n\n"
    "This schema serves as a reference for users to write correct NNCF configuration files and each "
    "loaded NNCF configuration file into an `NNCFConfig` object is validated against it.",
    "type": "object",
    "properties": {
        "input_info": with_attributes(
            make_object_or_array_of_objects_schema(SINGLE_INPUT_INFO_SCHEMA),
            description="Describe the specifics of your model inputs here. "
            "This information is used to build the internal graph representation "
            "that is leveraged for proper compression functioning, and for "
            "exporting the compressed model to an executable format.\n"
            "If this field is unspecified, NNCF will try to deduce the input shapes and tensor types for the graph "
            "building purposes based on dataloader objects that are passed to compression algorithms by the user.",
        ),
        "target_device": with_attributes(
            TARGET_DEVICE_SCHEMA,
            description="The target device, the specificity of which will be taken into "
            "account while compressing in order to obtain the best "
            "performance for this type of device. The default 'ANY' means "
            "compatible quantization supported by any HW. Set "
            "this value to 'TRIAL' if you are going to use a custom "
            "quantization schema.",
            default=TARGET_DEVICE,
        ),
        "compression": make_object_or_array_of_objects_schema(
            {"oneOf": [{"$ref": f"#/$defs/{algo_name}"} for algo_name in REF_VS_ALGO_SCHEMA]}
        ),
        "accuracy_aware_training": with_attributes(
            ACCURACY_AWARE_TRAINING_SCHEMA,
            description="Options for the execution of the NNCF-powered "
            "'Accuracy Aware' training pipeline. The 'mode' "
            "property determines the mode of the accuracy-aware "
            "training execution and further available parameters.",
        ),
        # Validation of each separate compression description schema occurs in a separate step.
        # This is required for better user feedback, since holistic schema validation is uninformative
        # if there is an error in one of the compression configs.
        **COMPRESSION_LR_MULTIPLIER_PROPERTY,
        "disable_shape_matching": with_attributes(
            BOOLEAN,
            description="[Deprecated] Whether to enable strict input tensor "
            "shape matching when building the internal graph "
            "representation of the model. Set this to false if your "
            "model inputs have any variable dimension other than "
            "the 0-th (batch) dimension, or if any non-batch "
            "dimension of the intermediate tensors in your model "
            "execution flow depends on the input dimension, "
            "otherwise the compression will most likely fail.",
        ),
        "log_dir": with_attributes(STRING, description="Log directory for NNCF-specific logging outputs."),
    },
    "$defs": REF_VS_ALGO_SCHEMA,
}


def validate_single_compression_algo_schema(
    single_compression_algo_dict: Dict[str, Any], ref_vs_algo_schema: Dict[str, Any]
) -> None:
    """single_compression_algo_dict must conform to BASIC_COMPRESSION_ALGO_SCHEMA (and possibly has other
    algo-specific properties"""
    algo_name = single_compression_algo_dict["algorithm"]
    if algo_name not in ref_vs_algo_schema:
        raise jsonschema.ValidationError(
            f"Incorrect algorithm name - must be one of {str(list(ref_vs_algo_schema.keys()))}"
        )
    try:
        jsonschema.validate(single_compression_algo_dict, schema=ref_vs_algo_schema[algo_name])
    except jsonschema.ValidationError as e:
        e.message = (
            f"While validating the config for algorithm '{algo_name}' , got:\n"
            + e.message
            + f"\nRefer to the algorithm subschema definition at {SCHEMA_VISUALIZATION_URL}\n"
        )
        if algo_name in ALGO_NAME_VS_README_URL:
            e.message += (
                f"or to the algorithm documentation for examples of the configs: "
                f"{ALGO_NAME_VS_README_URL[algo_name]}"
            )
        raise e


def validate_accuracy_aware_training_schema(single_compression_algo_dict: Dict[str, Any]) -> None:
    """
    Checks accuracy_aware_training section.
    """
    jsonschema.validate(single_compression_algo_dict, schema=ACCURACY_AWARE_TRAINING_SCHEMA)
    accuracy_aware_mode = single_compression_algo_dict.get("mode")
    if accuracy_aware_mode not in ACCURACY_AWARE_MODES_VS_SCHEMA:
        raise jsonschema.ValidationError(
            "Incorrect Accuracy Aware mode - must be one of ({})".format(
                ", ".join(ACCURACY_AWARE_MODES_VS_SCHEMA.keys())
            )
        )
    try:
        jsonschema.validate(single_compression_algo_dict, schema=ACCURACY_AWARE_MODES_VS_SCHEMA[accuracy_aware_mode])
    except Exception as e:
        raise e
