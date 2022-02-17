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
from abc import ABC
from abc import abstractmethod

from typing import List

from nncf.common.quantization.structs import QuantizerConfig
from nncf.experimental.post_training.compressed_model import CompressedModel
from nncf.experimental.post_training.initialization.statistics_collector import MinMaxLayerStatistic
from nncf.experimental.post_training.initialization.algorithm import InitializationAlgorithm
from nncf.experimental.post_training.initialization.algorithm import InitizalizationParameters

from nncf.experimental.post_training.initialization.statistics_collector import WEIGHTS_ESTIMATOR_FUNCTION
from nncf.experimental.post_training.initialization.statistics_collector import ACTIVATIONS_ESTIMATOR_FUNCTION
from nncf.experimental.post_training.initialization.statistics_collector import BATCH_AGGREGATION_FUNCTION
from nncf.experimental.post_training.initialization.statistics_collector import STATISTICS_AGGREGATION_FUNCTION


# from nncf.experimental.post_training.quantization.parameters import DEVICE


class QuantizerRangeFinderParameters(InitizalizationParameters):
    def __init__(self, weight_min_func: WEIGHTS_ESTIMATOR_FUNCTION,
                 weight_max_func: WEIGHTS_ESTIMATOR_FUNCTION,
                 activation_min_func: ACTIVATIONS_ESTIMATOR_FUNCTION,
                 activation_max_func: ACTIVATIONS_ESTIMATOR_FUNCTION,
                 batch_aggregation_min_func: BATCH_AGGREGATION_FUNCTION,
                 batch_aggregation_max_func: BATCH_AGGREGATION_FUNCTION,
                 statistics_aggregator_func: STATISTICS_AGGREGATION_FUNCTION,
                 weight_quantizer_config: QuantizerConfig,
                 activation_quantizer_config: QuantizerConfig,
                 # target_device: DEVICE,
                 target_device,
                 quatize_outputs: bool = False,
                 ignored_scopes: List[str] = None,
                 ):
        self.weight_min_func = weight_min_func
        self.weight_max_func = weight_max_func
        self.activation_min_func = activation_min_func
        self.activation_max_func = activation_max_func
        self.batch_aggregation_min_func = batch_aggregation_min_func
        self.batch_aggregation_max_func = batch_aggregation_max_func
        self.statistics_aggregator_func = statistics_aggregator_func
        self.weight_quantizer_config = weight_quantizer_config
        self.activation_quantizer_config = activation_quantizer_config
        self.ignored_scopes = ignored_scopes
        self.target_device = target_device
        self.quantize_outputs = quatize_outputs


class QuantizerRangeFinderAlgorithm(InitializationAlgorithm, ABC):
    """

    """

    def __init__(self, compressed_model: CompressedModel, engine, parameters: QuantizerRangeFinderParameters):
        super().__init__(compressed_model, engine, parameters)
        self._determine_aggregation_func()
        self.weight_quantizer_config = parameters.weight_quantizer_config
        self.activation_quantizer_config = parameters.activation_quantizer_config
        self.ignored_scopes = parameters.ignored_scopes
        self.target_device = parameters.target_device
        self.quantize_outputs = parameters.quantize_outputs

    @abstractmethod
    def _determine_aggregation_func(self):
        pass

    @abstractmethod
    def get_layers_for_statistics(self, weight_quantizer_config: QuantizerConfig,
                                  activation_quantizer_config: QuantizerConfig) -> List[MinMaxLayerStatistic]:
        pass

    @abstractmethod
    def get_transformation_commands(self, layers_statistics):
        pass
