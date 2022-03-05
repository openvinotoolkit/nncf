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
from copy import deepcopy

from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import QuantizationMode
from nncf.experimental.post_training.compressed_model import CompressedModel
from nncf.experimental.post_training.algorithms import Algorithm
from nncf.experimental.post_training.algorithms import AlgorithmParameters


class MinMaxQuantizationParameters(AlgorithmParameters):
    def __init__(self,
                 weight_quantizer_config: QuantizerConfig = None,
                 activation_quantizer_config: QuantizerConfig = None,
                 target_device: str = 'CPU',
                 range_type: str = 'min_max',
                 quatize_outputs: bool = False,
                 ignored_scopes: List[str] = None,
                 ):
        self.weight_quantizer_config = weight_quantizer_config
        self.activation_quantizer_config = activation_quantizer_config
        self.target_device = target_device
        self.range_type = range_type
        self.ignored_scopes = ignored_scopes
        self.quantize_outputs = quatize_outputs


class MinMaxQuantization(Algorithm, ABC):
    """

    """

    DEFAULT_QCONFIG = QuantizerConfig(num_bits=8,
                                      mode=QuantizationMode.SYMMETRIC,
                                      signedness_to_force=None,
                                      per_channel=False)

    def __init__(self, statistics_collector,
                 parameters: MinMaxQuantizationParameters):
        self.statistics_collector = statistics_collector
        self.weight_quantizer_config = parameters.weight_quantizer_config if parameters.weight_quantizer_config is not None else self._get_default_qconfig()
        self.activation_quantizer_config = parameters.activation_quantizer_config if parameters.activation_quantizer_config is not None else self._get_default_qconfig()
        self.target_device = parameters.target_device
        self.range_type = parameters.range_type
        self.quantize_outputs = parameters.quantize_outputs
        self.ignored_scopes = parameters.ignored_scopes

    def _get_default_qconfig(self) -> QuantizerConfig:
        qconfig = deepcopy(self.DEFAULT_QCONFIG)
        return qconfig

    @abstractmethod
    def _create_model_transformer(self):
        """

        """

    @abstractmethod
    def get_layers_for_statistics(self, compressed_model: CompressedModel):
        """

        """
