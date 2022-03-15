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

from typing import Dict
from typing import List
from typing import Union
from typing import Optional

from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import QuantizationMode
from nncf.common.hardware.config import HWConfigType

from nncf.experimental.post_training.algorithms import AlgorithmParameters
from nncf.experimental.post_training.algorithms import PostTrainingAlgorithms

from nncf.experimental.post_training.algorithms.min_max_quantization import MinMaxQuantizationParameters
from nncf.experimental.post_training.algorithms.min_max_quantization import Preset
from nncf.experimental.post_training.algorithms.min_max_quantization import Granularity
from nncf.experimental.post_training.algorithms.min_max_quantization import RangeType


class PostTrainingQuantizationParameters(AlgorithmParameters):
    """
    This class handles parameters for PostTrainingQuantization algorithm.
    """

    def __init__(self,
                 preset: Preset = Preset.MIXED,
                 weight_bits: int = 8,
                 weight_granularity: Granularity = Granularity.PERCHANNEL,
                 activation_bits: int = 8,
                 activation_granularity: Granularity = Granularity.PERTENSOR,
                 range_type: RangeType = RangeType.MEAN_MINMAX,
                 number_samples: int = 300,
                 target_device: HWConfigType = HWConfigType.CPU,
                 ignored_scopes: Optional[List[str]] = None
                 ):
        weight_mode, activation_mode = self._determine_weight_activation_modes(preset)
        self.weight_quantizer_config = self._determine_quantizer_config(weight_bits, weight_granularity, weight_mode)
        self.activation_quantizer_config = self._determine_quantizer_config(activation_bits, activation_granularity,
                                                                            activation_mode)

        self.algorithms = {PostTrainingAlgorithms.MinMaxQuantization: MinMaxQuantizationParameters(
            weight_quantizer_config=self.weight_quantizer_config,
            activation_quantizer_config=self.activation_quantizer_config,
            range_type=range_type,
            ignored_scopes=ignored_scopes,
            target_device=target_device
        )}

        self.number_samples = number_samples
        self.target_device = target_device
        self.ignored_scopes = ignored_scopes

    def to_json(self) -> Dict[str, Union[str, float, int]]:
        pass

    def _determine_weight_activation_modes(self, preset: Preset):
        # TODO(kshpv): add support of presets
        weight_mode = QuantizationMode.SYMMETRIC
        activation_mode = QuantizationMode.SYMMETRIC
        return weight_mode, activation_mode

    def _determine_quantizer_config(self, number_bits: int,
                                    granularity: Granularity, mode: QuantizationMode):
        return QuantizerConfig(num_bits=number_bits, mode=mode,
                               per_channel=granularity == Granularity.PERCHANNEL)
