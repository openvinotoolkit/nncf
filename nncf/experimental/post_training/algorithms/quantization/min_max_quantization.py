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
from typing import Dict, List, Union, Tuple, TypeVar

from abc import ABC
from copy import deepcopy

from nncf.common.utils.ordered_enum import OrderedEnum
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import QuantizationMode
from nncf.common.quantization.structs import QuantizationPreset
from nncf.common.quantization.structs import QuantizerGroup
from nncf.common.hardware.config import HWConfigType

from nncf.experimental.post_training.algorithms import Algorithm
from nncf.experimental.post_training.algorithms import AlgorithmParameters

ModelType = TypeVar('ModelType')


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
                 preset: QuantizationPreset = QuantizationPreset.PERFORMANCE,
                 weight_bits: int = 8,
                 weight_granularity: Granularity = Granularity.PERCHANNEL,
                 activation_bits: int = 8,
                 activation_granularity: Granularity = Granularity.PERTENSOR,
                 range_type: RangeType = RangeType.MEAN_MINMAX,
                 number_samples: int = 100,
                 target_device: HWConfigType = HWConfigType.CPU,
                 quantize_outputs: bool = False,
                 ignored_scopes: List[str] = None
                 ):
        weight_mode, activation_mode = self._determine_weight_activation_modes(preset)
        self.weight_quantizer_config = self._determine_quantizer_config(weight_bits, weight_granularity, weight_mode)
        self.activation_quantizer_config = self._determine_quantizer_config(activation_bits, activation_granularity,
                                                                            activation_mode)
        self.number_samples = number_samples
        self.target_device = target_device
        self.range_type = range_type
        self.quantize_outputs = quantize_outputs
        self.ignored_scopes = [] if ignored_scopes is None else ignored_scopes

    def to_json(self) -> Dict[str, Union[str, float, int]]:
        """
        Serialize all MinMaxQuantization parameters to JSON.
        """

    def _determine_weight_activation_modes(self, preset: QuantizationPreset) -> Tuple[
        QuantizationMode, QuantizationMode]:
        weight_mode = QuantizationPreset.get_params_configured_by_preset(preset, QuantizerGroup.WEIGHTS)['mode']
        activation_mode = QuantizationPreset.get_params_configured_by_preset(preset, QuantizerGroup.ACTIVATIONS)['mode']
        return weight_mode, activation_mode

    def _determine_quantizer_config(self, number_bits: int,
                                    granularity: Granularity, mode: QuantizationMode) -> QuantizerConfig:
        return QuantizerConfig(num_bits=number_bits, mode=mode,
                               per_channel=granularity == Granularity.PERCHANNEL)


class MinMaxQuantization(Algorithm, ABC):
    """
    Base class of MinMaxQuantization algorithm. It has the default quantization config.
    """

    DEFAULT_QCONFIG = QuantizerConfig(num_bits=8,
                                      mode=QuantizationMode.SYMMETRIC,
                                      signedness_to_force=None,
                                      per_channel=False)

    def __init__(self, parameters: MinMaxQuantizationParameters):
        super().__init__()
        self.weight_quantizer_config = parameters.weight_quantizer_config
        self.activation_quantizer_config = parameters.activation_quantizer_config
        self.range_type = parameters.range_type
        self.number_samples = parameters.number_samples
        self.target_device = parameters.target_device
        self.quantize_outputs = parameters.quantize_outputs
        self.ignored_scopes = parameters.ignored_scopes

    def _get_default_qconfig(self) -> QuantizerConfig:
        qconfig = deepcopy(self.DEFAULT_QCONFIG)
        return qconfig
