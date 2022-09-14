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
from nncf.config.definitions import ONLINE_DOCS_ROOT
from nncf.config.schemata.basic import NUMBER
from nncf.config.schemata.basic import with_attributes

COMMON_AA_PROPERTIES = {
    "maximal_relative_accuracy_degradation":
        with_attributes(NUMBER,
                        description="Maximally allowed accuracy degradation of the model "
                                    "in percent relative to the original model accuracy."),
    "maximal_absolute_accuracy_degradation":
        with_attributes(NUMBER,
                        description="Maximally allowed accuracy degradation of the model "
                                    "in units of absolute metric of the original model."),
}

ADAPTIVE_COMPRESSION_LEVEL_TRAINING_MODE_NAME_IN_CONFIG = "adaptive_compression_level"
ADAPTIVE_COMPRESSION_LEVEL_TRAINING_SCHEMA = {
    "type": "object",
    "title": ADAPTIVE_COMPRESSION_LEVEL_TRAINING_MODE_NAME_IN_CONFIG,
    "description": f"Adaptive compression level training mode schema. See "
                   f"[AdaptiveCompressionLevelTraining.md]"
                   f"({ONLINE_DOCS_ROOT}docs/accuracy_aware_model_training/AdaptiveCompressionLevelTraining.md) "
                   f"for more general info on this mode.",
    "properties": {
        "mode": {"const": ADAPTIVE_COMPRESSION_LEVEL_TRAINING_MODE_NAME_IN_CONFIG},
        "params": {
            "type": "object",
            "properties": {
                **COMMON_AA_PROPERTIES,
                "initial_training_phase_epochs":
                    with_attributes(NUMBER,
                                    description="Number of epochs to fine-tune during the initial "
                                                "training phase of the adaptive compression training loop."),
                "initial_compression_rate_step":
                    with_attributes(NUMBER,
                                    description="Initial value for the compression rate increase/decrease "
                                                "training phase of the compression training loop."),
                "compression_rate_step_reduction_factor":
                    with_attributes(NUMBER,
                                    description="Factor used to reduce the compression rate change step "
                                                "in the adaptive compression training loop."),
                "minimal_compression_rate_step":
                    with_attributes(NUMBER,
                                    description="The minimal compression rate change step value "
                                                "after which the training loop is terminated."),
                "patience_epochs":
                    with_attributes(NUMBER,
                                    description="The number of epochs to fine-tune the model "
                                                "for a given compression rate after the initial "
                                                "training phase of the training loop."),
                "maximal_total_epochs":
                    with_attributes(NUMBER,
                                    description="The maximal total fine-tuning epoch count. If the epoch "
                                                "counter reaches this number, the fine-tuning process will "
                                                "stop and the model with the largest compression rate "
                                                "will be returned."),
            },
            "oneOf": [{"required": ["maximal_relative_accuracy_degradation"]},
                      {"required": ["maximal_absolute_accuracy_degradation"]}],
            "required": ["initial_training_phase_epochs", "patience_epochs"],
            "additionalProperties": False
        },

    },
    "required": ["mode", "params"],
    "additionalProperties": False
}
EARLY_EXIT_TRAINING_MODE_NAME_IN_CONFIG = "early_exit"
EARLY_EXIT_TRAINING_SCHEMA = {
    "type": "object",
    "title": EARLY_EXIT_TRAINING_MODE_NAME_IN_CONFIG,
    "description": f"Early exit mode schema. See "
                   f"[EarlyExitTraining.md]({ONLINE_DOCS_ROOT}docs/accuracy_aware_model_training/EarlyExitTraining.md) for "
                   f"more general info on this mode.",
    "properties": {
        "mode": {"const": EARLY_EXIT_TRAINING_MODE_NAME_IN_CONFIG},
        "params": {
            "type": "object",
            "properties": {
                **COMMON_AA_PROPERTIES,
                "maximal_total_epochs":
                    with_attributes(NUMBER,
                                    description="The maximal total fine-tuning epoch count. If the accuracy criteria "
                                                "wouldn't reach during fine-tuning, the most accurate model "
                                                "will be returned."),
            },
            "oneOf": [{"required": ["maximal_relative_accuracy_degradation"]},
                      {"required": ["maximal_absolute_accuracy_degradation"]}],
            "required": ["maximal_total_epochs"],
            "additionalProperties": False
        },
    },
    "required": ["mode", "params"],
    "additionalProperties": False
}
ACCURACY_AWARE_TRAINING_SCHEMA = {
    "type": "object",
    "oneOf": [EARLY_EXIT_TRAINING_SCHEMA,
              ADAPTIVE_COMPRESSION_LEVEL_TRAINING_SCHEMA],
}
ACCURACY_AWARE_MODES_VS_SCHEMA = {
    ADAPTIVE_COMPRESSION_LEVEL_TRAINING_MODE_NAME_IN_CONFIG: ADAPTIVE_COMPRESSION_LEVEL_TRAINING_SCHEMA,
    EARLY_EXIT_TRAINING_MODE_NAME_IN_CONFIG: EARLY_EXIT_TRAINING_SCHEMA
}
