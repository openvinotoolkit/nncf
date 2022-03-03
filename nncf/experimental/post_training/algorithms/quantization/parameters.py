from typing import List
from typing import Optional

from nncf.common.utils.ordered_enum import OrderedEnum
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import QuantizationMode

from nncf.experimental.post_training.algorithms import AlgorithmParameters
from nncf.experimental.post_training.algorithms import PostTrainingAlgorithms

from nncf.experimental.post_training.algorithms.quantizer_range_finder import QuantizerRangeFinderParameters


class PRESET(OrderedEnum):
    PERFOMANCE = 'perfomance'
    MIXED = 'mixed'
    ACCURACY = 'accuracy'


class GRANULARITY(OrderedEnum):
    PERTENSOR = 'pertensor'
    PERCHANNEL = 'perchannel'


class PostTrainingQuantizationParameters(AlgorithmParameters):
    def __init__(self,
                 preset: PRESET = PRESET.MIXED,
                 weight_bits: int = 8,
                 weight_granularity: GRANULARITY = GRANULARITY.PERCHANNEL,
                 activation_bits: int = 8,
                 activation_granularity: GRANULARITY = GRANULARITY.PERTENSOR,
                 range_type: str = 'min_max',
                 number_samples: int = 300,
                 target_device: str = 'CPU',
                 ignored_scopes: Optional[List[str]] = None
                 ):
        self._determine_weight_activation_quantizers_config(
            preset,
            weight_bits,
            weight_granularity,
            activation_bits,
            activation_granularity
        )

        self.algorithms = {PostTrainingAlgorithms.QuantizerRangeFinder: QuantizerRangeFinderParameters(
            weight_quantizer_config=self.weight_quantizer_config,
            activation_quantizer_config=self.activation_quantizer_config,
            range_type=range_type,
            ignored_scopes=ignored_scopes,
            target_device=target_device
        )}

        self.number_samples = number_samples
        self.target_device = target_device
        self.ignored_scopes = ignored_scopes

    def _determine_weight_activation_quantizers_config(self, preset: PRESET, weight_bits: int,
                                                       weights_granularity: GRANULARITY, activation_bits: int,
                                                       activations_granularity: GRANULARITY):
        def _determine_weight_activation_modes(preset: PRESET):
            weight_mode = QuantizationMode.SYMMETRIC
            activation_mode = QuantizationMode.SYMMETRIC
            return weight_mode, activation_mode

        weight_mode, activation_mode = _determine_weight_activation_modes(preset)

        weights_per_channel = True if weights_granularity == GRANULARITY.PERCHANNEL else False
        activation_per_channel = True if activations_granularity == GRANULARITY.PERCHANNEL else False

        self.weight_quantizer_config = QuantizerConfig(num_bits=weight_bits, mode=weight_mode,
                                                       per_channel=weights_per_channel)
        self.activation_quantizer_config = QuantizerConfig(num_bits=activation_bits, mode=activation_mode,
                                                           per_channel=activation_per_channel)
