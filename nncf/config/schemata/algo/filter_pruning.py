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
from nncf.config.schemata.common.compression import BASIC_COMPRESSION_ALGO_SCHEMA
from nncf.config.schemata.basic import BOOLEAN
from nncf.config.schemata.basic import NUMBER
from nncf.config.schemata.basic import STRING
from nncf.config.schemata.basic import with_attributes
from nncf.config.schemata.common.targeting import GENERIC_INITIALIZER_SCHEMA
from nncf.config.schemata.common.targeting import SCOPING_PROPERTIES

FILTER_PRUNING_ALGO_NAME_IN_CONFIG = 'filter_pruning'
FILTER_PRUNING_SCHEMA = {
    **BASIC_COMPRESSION_ALGO_SCHEMA,
    "properties": {
        "algorithm": {
            "const": FILTER_PRUNING_ALGO_NAME_IN_CONFIG
        },
        "initializer": GENERIC_INITIALIZER_SCHEMA,
        "pruning_init": with_attributes(NUMBER,
                                        description="Initial value of the pruning level applied to the "
                                                    "convolutions that can be pruned. "
                                                    "0.0 by default."),
        "params":
            {
                "type": "object",
                "properties": {
                    "schedule": with_attributes(STRING,
                                                description="The type of scheduling to use for adjusting the target"
                                                            " pruning level. Either `exponential`, `exponential_with"
                                                            "_bias`,  or `baseline`, by default it is `exponential`"),
                    "pruning_target": with_attributes(NUMBER,
                                                      description="Target value of the pruning level for "
                                                                  "the convolutions that can be pruned. "
                                                                  "These convolutions are determined by the model "
                                                                  "architecture."
                                                                  " 0.5 by default."),
                    "pruning_flops_target": with_attributes(NUMBER,
                                                            description="Target value of the pruning level for model"
                                                                        " FLOPs."),
                    "num_init_steps": with_attributes(NUMBER,
                                                      description="Number of epochs for model pretraining before"
                                                                  " starting filter pruning. 0 by default."),
                    "pruning_steps": with_attributes(NUMBER,
                                                     description="Number of epochs during which the pruning level is"
                                                                 " increased from `pruning_init` to `pruning_target`"
                                                                 " value."),
                    "filter_importance": with_attributes(STRING,
                                                         description="The type of filter importance metric. Can be"
                                                                     " one of `L1`, `L2`, `geometric_median`."
                                                                     " `L2` by default."),
                    "interlayer_ranking_type": with_attributes(STRING,
                                                               description="The type of filter ranking across the "
                                                                           "layers. Can be one of `unweighted_ranking`"
                                                                           " or `learned_ranking`."),
                    "all_weights": with_attributes(BOOLEAN,
                                                   description="Whether to prune layers independently (choose filters"
                                                               " with the smallest importance in each layer separately)"
                                                               " or not. `False` by default.",
                                                   default=False),
                    "prune_first_conv": with_attributes(BOOLEAN,
                                                        description="Whether to prune first Convolutional layers or"
                                                                    " not. First means that it is a convolutional layer"
                                                                    " such that there is a path from model input to "
                                                                    "this layer such that there are no other "
                                                                    "convolution operations on it. `False` by default.",
                                                        default=False
                                                        ),
                    "prune_downsample_convs": with_attributes(BOOLEAN,
                                                              description="Whether to prune downsample Convolutional"
                                                                          " layers (with stride > 1) or not. `False`"
                                                                          " by default.",
                                                              default=False
                                                              ),
                    "prune_batch_norms": with_attributes(BOOLEAN,
                                                         description="Whether to nullifies parameters of Batch Norm"
                                                                     " layer corresponds to zeroed filters of"
                                                                     " convolution corresponding to this Batch Norm."
                                                                     " `False` by default.",
                                                         default=False
                                                         ),
                    "save_ranking_coeffs_path": with_attributes(STRING),
                    "load_ranking_coeffs_path": with_attributes(STRING),
                    "legr_params":
                        {
                            "type": "object",
                            "properties": {
                                "generations": with_attributes(NUMBER,
                                                               description="Number of generations for evolution"
                                                                           "algorithm."),
                                "train_steps": with_attributes(NUMBER,
                                                               description="Number of training steps to estimate"
                                                                           "pruned model accuracy."),
                                "max_pruning": with_attributes(NUMBER,
                                                               description="Pruning level for the model to train"
                                                                           " LeGR algorithm on it. If learned ranking"
                                                                           " will be used for multiple pruning"
                                                                           " rates, the highest should be used as"
                                                                           "`max_pruning`. If model will be pruned"
                                                                           " with one pruning level, this target should"
                                                                           "be used."),
                                "random_seed": with_attributes(NUMBER,
                                                               description="Random seed for LeGR coefficients"
                                                                           " generation.")
                            }
                        },

                },
                "additionalProperties": False,
            },
        **SCOPING_PROPERTIES
    },
    "additionalProperties": False
}
