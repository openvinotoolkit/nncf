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
        self._determine_weight_activation_quantizers_config(
            preset,
            weight_bits,
            weight_granularity,
            activation_bits,
            activation_granularity
        )

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

    def _determine_weight_activation_quantizers_config(self, preset: Preset, weight_bits: int,
                                                       weights_granularity: Granularity, activation_bits: int,
                                                       activations_granularity: Granularity):
        # TODO (kshpv): make one func for activations and weights
        def _determine_weight_activation_modes(preset: Preset):
            weight_mode = QuantizationMode.SYMMETRIC
            activation_mode = QuantizationMode.SYMMETRIC
            return weight_mode, activation_mode

        weight_mode, activation_mode = _determine_weight_activation_modes(preset)

        weights_per_channel = True if weights_granularity == Granularity.PERCHANNEL else False
        activation_per_channel = True if activations_granularity == Granularity.PERCHANNEL else False

        self.weight_quantizer_config = QuantizerConfig(num_bits=weight_bits, mode=weight_mode,
                                                       per_channel=weights_per_channel)
        self.activation_quantizer_config = QuantizerConfig(num_bits=activation_bits, mode=activation_mode,
                                                           per_channel=activation_per_channel)
