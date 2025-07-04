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

ONLINE_DOCS_ROOT = "https://github.com/openvinotoolkit/nncf/tree/develop/"
SCHEMA_VISUALIZATION_URL = "https://openvinotoolkit.github.io/nncf/"

ADAPTIVE_COMPRESSION_LEVEL_TRAINING_MODE_NAME_IN_CONFIG = "adaptive_compression_level"
EARLY_EXIT_TRAINING_MODE_NAME_IN_CONFIG = "early_exit"
EXPERIMENTAL_QUANTIZATION_ALGO_NAME_IN_CONFIG = "experimental_quantization"
BOOTSTRAP_NAS_ALGO_NAME_IN_CONFIG = "bootstrapNAS"
CONST_SPARSITY_ALGO_NAME_IN_CONFIG = "const_sparsity"
FILTER_PRUNING_ALGO_NAME_IN_CONFIG = "filter_pruning"
KNOWLEDGE_DISTILLATION_ALGO_NAME_IN_CONFIG = "knowledge_distillation"
MAGNITUDE_SPARSITY_ALGO_NAME_IN_CONFIG = "magnitude_sparsity"
MOVEMENT_SPARSITY_ALGO_NAME_IN_CONFIG = "movement_sparsity"
QUANTIZATION_ALGO_NAME_IN_CONFIG = "quantization"
RB_SPARSITY_ALGO_NAME_IN_CONFIG = "rb_sparsity"

ALGO_NAME_VS_README_URL = {
    QUANTIZATION_ALGO_NAME_IN_CONFIG: "docs/compression_algorithms/Quantization.md",
    FILTER_PRUNING_ALGO_NAME_IN_CONFIG: "docs/compression_algorithms/Pruning.md",
    MAGNITUDE_SPARSITY_ALGO_NAME_IN_CONFIG: "docs/compression_algorithms/Sparsity.md",
    RB_SPARSITY_ALGO_NAME_IN_CONFIG: "docs/compression_algorithms/Sparsity.md",
    CONST_SPARSITY_ALGO_NAME_IN_CONFIG: "docs/compression_algorithms/Sparsity.md",
    KNOWLEDGE_DISTILLATION_ALGO_NAME_IN_CONFIG: "docs/compression_algorithms/KnowledgeDistillation.md",
}
