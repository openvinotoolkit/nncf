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

from nncf.config.definitions import FILTER_PRUNING_ALGO_NAME_IN_CONFIG
from nncf.config.definitions import ONLINE_DOCS_ROOT
from nncf.config.schemata.basic import BOOLEAN
from nncf.config.schemata.basic import NUMBER
from nncf.config.schemata.basic import STRING
from nncf.config.schemata.basic import with_attributes
from nncf.config.schemata.common.compression import BASIC_COMPRESSION_ALGO_SCHEMA
from nncf.config.schemata.common.targeting import GENERIC_INITIALIZER_SCHEMA
from nncf.config.schemata.common.targeting import SCOPING_PROPERTIES
from nncf.config.schemata.defaults import PRUNE_BATCH_NORMS
from nncf.config.schemata.defaults import PRUNE_DOWNSAMPLE_CONVS
from nncf.config.schemata.defaults import PRUNE_FIRST_CONV
from nncf.config.schemata.defaults import PRUNING_ALL_WEIGHTS
from nncf.config.schemata.defaults import PRUNING_FILTER_IMPORTANCE
from nncf.config.schemata.defaults import PRUNING_INIT
from nncf.config.schemata.defaults import PRUNING_INTERLAYER_RANKING_TYPE
from nncf.config.schemata.defaults import PRUNING_LEGR_GENERATIONS
from nncf.config.schemata.defaults import PRUNING_LEGR_MAX_PRUNING
from nncf.config.schemata.defaults import PRUNING_LEGR_MUTATE_PERCENT
from nncf.config.schemata.defaults import PRUNING_LEGR_NUM_SAMPLES
from nncf.config.schemata.defaults import PRUNING_LEGR_POPULATION_SIZE
from nncf.config.schemata.defaults import PRUNING_LEGR_RANDOM_SEED
from nncf.config.schemata.defaults import PRUNING_LEGR_SIGMA_SCALE
from nncf.config.schemata.defaults import PRUNING_LEGR_TRAIN_STEPS
from nncf.config.schemata.defaults import PRUNING_NUM_INIT_STEPS
from nncf.config.schemata.defaults import PRUNING_SCHEDULE
from nncf.config.schemata.defaults import PRUNING_STEPS
from nncf.config.schemata.defaults import PRUNING_TARGET

FILTER_PRUNING_SCHEDULE_OPTIONS = ["exponential", "exponential_with_bias", "baseline"]
FILTER_IMPORTANCE_OPTIONS = ["L2", "L1", "geometric_median"]
INTERLAYER_RANKING_TYPE_OPTIONS = ["unweighted_ranking", "learned_ranking"]

FILTER_PRUNING_SCHEMA = {
    **BASIC_COMPRESSION_ALGO_SCHEMA,
    "description": f"Applies filter pruning during training of the model to effectively remove entire "
    f"sub-dimensions of tensors in the original model from computation and therefore increase "
    f"performance.\n"
    f"See [Pruning.md]"
    f"({ONLINE_DOCS_ROOT}"
    f"/docs/compression_algorithms/Pruning.md) and the rest of this schema for "
    f"more details and parameters.",
    "properties": {
        "algorithm": {"const": FILTER_PRUNING_ALGO_NAME_IN_CONFIG},
        "initializer": GENERIC_INITIALIZER_SCHEMA,
        "pruning_init": with_attributes(
            NUMBER,
            description="Initial value of the pruning level applied to the prunable operations.",
            default=PRUNING_INIT,
        ),
        "params": {
            "type": "object",
            "properties": {
                "filter_importance": with_attributes(
                    STRING,
                    description="The type of filter importance metric.",
                    enum=FILTER_IMPORTANCE_OPTIONS,
                    default=PRUNING_FILTER_IMPORTANCE,
                ),
                "pruning_target": with_attributes(
                    NUMBER,
                    description="Target value of the pruning level for "
                    "the operations that can be pruned. "
                    "The operations are determined by analysis of the "
                    "model architecture during the pruning algorithm "
                    "initialization stage.",
                    default=PRUNING_TARGET,
                ),
                "pruning_steps": with_attributes(
                    NUMBER,
                    description="Number of epochs during which the pruning level is "
                    "increased from `pruning_init` to `pruning_target`.",
                    default=PRUNING_STEPS,
                ),
                "pruning_flops_target": with_attributes(
                    NUMBER, description="Target value of the pruning level for model FLOPs."
                ),
                "schedule": with_attributes(
                    STRING,
                    description="The type of scheduling to use for adjusting the target pruning level.",
                    enum=FILTER_PRUNING_SCHEDULE_OPTIONS,
                    default=PRUNING_SCHEDULE,
                ),
                "num_init_steps": with_attributes(
                    NUMBER,
                    description="Number of epochs for model pretraining before starting filter pruning.",
                    default=PRUNING_NUM_INIT_STEPS,
                ),
                "interlayer_ranking_type": with_attributes(
                    STRING,
                    description="The type of filter ranking across the layers.",
                    enum=INTERLAYER_RANKING_TYPE_OPTIONS,
                    default=PRUNING_INTERLAYER_RANKING_TYPE,
                ),
                "all_weights": with_attributes(
                    BOOLEAN,
                    description="Whether to prune layers independently (choose filters "
                    "with the smallest importance in each layer separately) "
                    "or not.",
                    default=PRUNING_ALL_WEIGHTS,
                ),
                "prune_first_conv": with_attributes(
                    BOOLEAN,
                    description="Whether to prune first convolutional layers or "
                    "not. A 'first' convolutional layer is such a "
                    "layer that the path from model input to "
                    "this layer has no other "
                    "convolution operations on it.",
                    default=PRUNE_FIRST_CONV,
                ),
                "prune_downsample_convs": with_attributes(
                    BOOLEAN,
                    description="Whether to prune downsampling convolutional layers (with stride > 1) or not.",
                    default=PRUNE_DOWNSAMPLE_CONVS,
                ),
                "prune_batch_norms": with_attributes(
                    BOOLEAN,
                    description="Whether to prune parameters of the Batch Norm "
                    "layer that corresponds to pruned filters of the "
                    "convolutional layer which feeds its output to "
                    "this Batch Norm.",
                    default=PRUNE_BATCH_NORMS,
                ),
                "legr_params": {
                    "type": "object",
                    "description": f"Describes parameters specific to the LeGR pruning algorithm."
                    f"See [Pruning.md]"
                    f"({ONLINE_DOCS_ROOT}"
                    f"/docs/compression_algorithms/Pruning.md#interlayer-ranking-types) "
                    f"for more details.",
                    "properties": {
                        "generations": with_attributes(
                            NUMBER,
                            description="Number of generations for the evolution algorithm.",
                            default=PRUNING_LEGR_GENERATIONS,
                        ),
                        "train_steps": with_attributes(
                            NUMBER,
                            description="Number of training steps to estimate pruned model accuracy.",
                            default=PRUNING_LEGR_TRAIN_STEPS,
                        ),
                        "max_pruning": with_attributes(
                            NUMBER,
                            description="Pruning level for the model to train "
                            "LeGR algorithm on it. If learned ranking "
                            "will be used for multiple pruning "
                            "levels, the highest should be used as "
                            "`max_pruning`. If model will be pruned "
                            "with one pruning level, this target should "
                            "be used.",
                            default=PRUNING_LEGR_MAX_PRUNING,
                        ),
                        "random_seed": with_attributes(
                            NUMBER,
                            description="Random seed for LeGR coefficients generation.",
                            default=PRUNING_LEGR_RANDOM_SEED,
                        ),
                        "population_size": with_attributes(
                            NUMBER,
                            description="Size of population for the evolution algorithm.",
                            default=PRUNING_LEGR_POPULATION_SIZE,
                        ),
                        "num_samples": with_attributes(
                            NUMBER,
                            description="Number of samples for the evolution algorithm.",
                            default=PRUNING_LEGR_NUM_SAMPLES,
                        ),
                        "mutate_percent": with_attributes(
                            NUMBER,
                            description="Percent of mutate for the evolution algorithm.",
                            default=PRUNING_LEGR_MUTATE_PERCENT,
                        ),
                        "scale_sigma": with_attributes(
                            NUMBER,
                            description="Scale sigma for the evolution algorithm.",
                            default=PRUNING_LEGR_SIGMA_SCALE,
                        ),
                    },
                    "additionalProperties": False,
                },
                "save_ranking_coeffs_path": with_attributes(STRING),  # TODO(vshampor): is this important?
                "load_ranking_coeffs_path": with_attributes(STRING),  # TODO(vshampor): is this important?
            },
            "additionalProperties": False,
        },
        **SCOPING_PROPERTIES,
    },
    "additionalProperties": False,
}
