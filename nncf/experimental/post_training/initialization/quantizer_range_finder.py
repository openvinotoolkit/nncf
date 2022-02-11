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

from typing import List

from nncf.common.quantization.structs import QuantizerConfig
from nncf.experimental.post_training.initialization.statistics_collector import LayerStatistic
from nncf.experimental.post_training.initialization.algorithm import InitializationAlgorithm
from nncf.experimental.post_training.initialization.algorithm import InitizalizationParameters

from nncf.experimental.post_training.initialization.statistics_collector import WEIGHTS_ESTIMATOR_FUNCTION
from nncf.experimental.post_training.initialization.statistics_collector import ACTIVATIONS_ESTIMATOR_FUNCTION
from nncf.experimental.post_training.initialization.statistics_collector import BATCH_AGGREGATION_FUNCTION


class QuantizerRangeFinderParameters(InitizalizationParameters):
    def __init__(self, weight_min_func: WEIGHTS_ESTIMATOR_FUNCTION,
                 weight_max_func: WEIGHTS_ESTIMATOR_FUNCTION,
                 activation_min_func: ACTIVATIONS_ESTIMATOR_FUNCTION,
                 activation_max_func: ACTIVATIONS_ESTIMATOR_FUNCTION,
                 batch_aggregation_min_func: BATCH_AGGREGATION_FUNCTION,
                 batch_aggregation_max_func: BATCH_AGGREGATION_FUNCTION
                 ):
        self.weight_min_func = weight_min_func
        self.weight_max_func = weight_max_func
        self.activation_min_func = activation_min_func
        self.activation_max_func = activation_max_func
        self.batch_aggregation_min_func = batch_aggregation_min_func
        self.batch_aggregation_max_func = batch_aggregation_max_func


class QuantizerRangeFinderAlgorithm(InitializationAlgorithm):
    """

    """

    def get_layers_for_statistics(self, weight_quantizer_config: QuantizerConfig,
                                  activation_quantizer_config: QuantizerConfig) -> List[LayerStatistic]:
        pass
