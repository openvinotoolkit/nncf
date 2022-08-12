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

from typing import Dict
from typing import Union
from typing import List
from typing import TypeVar
from copy import deepcopy

from nncf.common.utils.ordered_enum import OrderedEnum
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import QuantizationMode
from nncf.common.hardware.config import HWConfigType
from nncf.common.graph.model_transformer import ModelTransformer

from nncf.experimental.post_training.algorithms import Algorithm
from nncf.experimental.post_training.algorithms import AlgorithmParameters

ModelType = TypeVar('ModelType')


class Preset(OrderedEnum):
    PERFOMANCE = 'perfomance'
    MIXED = 'mixed'
    ACCURACY = 'accuracy'


class Granularity(OrderedEnum):
    PERTENSOR = 'pertensor'
    PERCHANNEL = 'perchannel'


class RangeType(OrderedEnum):
    MINMAX = 'min_max'
    MEAN_MINMAX = 'mean_min_max'


class MinMaxQuantizationParameters(AlgorithmParameters):
    """
    Base class of MinMaxQuantization parameters.
    """

    def __init__(self,
                 weight_quantizer_config: QuantizerConfig = None,
                 activation_quantizer_config: QuantizerConfig = None,
                 number_samples: int = 100,
                 target_device: HWConfigType = HWConfigType.CPU,
                 range_type: RangeType = RangeType.MEAN_MINMAX,
                 quatize_outputs: bool = False,
                 ignored_scopes: List[str] = None,
                 ):
        self.weight_quantizer_config = weight_quantizer_config
        self.activation_quantizer_config = activation_quantizer_config
        self.number_samples = number_samples
        self.target_device = target_device
        self.range_type = range_type
        self.ignored_scopes = [] if ignored_scopes is None else ignored_scopes
        self.quantize_outputs = quatize_outputs

    def to_json(self) -> Dict[str, Union[str, float, int]]:
        """
        Serialize all MinMaxQuantization parameters to JSON.
        """


class MinMaxQuantization(Algorithm, ABC):
    """
    Base class of MinMaxQuantization algorithm. It has the default quantization config.
    """

    DEFAULT_QCONFIG = QuantizerConfig(num_bits=8,
                                      mode=QuantizationMode.SYMMETRIC,
                                      signedness_to_force=None,
                                      per_channel=False)

    def __init__(self, parameters: MinMaxQuantizationParameters):
        self.weight_quantizer_config = parameters.weight_quantizer_config \
            if parameters.weight_quantizer_config is not None else self._get_default_qconfig()
        self.activation_quantizer_config = parameters.activation_quantizer_config \
            if parameters.activation_quantizer_config is not None else self._get_default_qconfig()
        self.number_samples = parameters.number_samples
        self.target_device = parameters.target_device
        self.range_type = parameters.range_type
        self.quantize_outputs = parameters.quantize_outputs
        self.ignored_scopes = parameters.ignored_scopes

    def _get_default_qconfig(self) -> QuantizerConfig:
        qconfig = deepcopy(self.DEFAULT_QCONFIG)
        return qconfig

    @abstractmethod
    def _create_model_transformer(self, model: ModelType) -> ModelTransformer:
        """
        Create framework-specific ModelTransformer.
        """
