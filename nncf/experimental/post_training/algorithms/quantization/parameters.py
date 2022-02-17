from typing import List
from typing import Dict
from typing import Optional

from nncf.common.utils.ordered_enum import OrderedEnum
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import QuantizationMode
from nncf.common.hardware.config import HWConfigType

from nncf.experimental.post_training.algorithms import AlgorithmParameters
from nncf.experimental.post_training.algorithms import PostTrainingAlgorithms

from nncf.experimental.post_training.algorithms.quantizer_range_finder import QuantizerRangeFinderAlgorithm
from nncf.experimental.post_training.algorithms.quantizer_range_finder import QuantizerRangeFinderParameters
from nncf.experimental.post_training.algorithms.bias_correction import BiasCorrectionAlgorithm
from nncf.experimental.post_training.algorithms.bias_correction import BiasCorrectionAlgorithmParameters

from nncf.experimental.post_training.statistics.statistics_collector import WEIGHTS_ESTIMATOR_FUNCTION
from nncf.experimental.post_training.statistics.statistics_collector import ACTIVATIONS_ESTIMATOR_FUNCTION
from nncf.experimental.post_training.statistics.statistics_collector import BATCH_AGGREGATION_FUNCTION
from nncf.experimental.post_training.statistics.statistics_collector import STATISTICS_AGGREGATION_FUNCTION


class PRESET(OrderedEnum):
    PERFOMANCE = 'perfomance'
    MIXED = 'mixed'
    ACCURACY = 'accuracy'


class GRANULARITY(OrderedEnum):
    PERTENSOR = 'pertensor'
    PERCHANNEL = 'perchannel'


class WeightsRangeEstimatorParameters:
    def __init__(self, min_estimator_function: WEIGHTS_ESTIMATOR_FUNCTION = WEIGHTS_ESTIMATOR_FUNCTION.MIN,
                 max_estimator_function: WEIGHTS_ESTIMATOR_FUNCTION = WEIGHTS_ESTIMATOR_FUNCTION.MAX):
        self.min_estimator_function = min_estimator_function
        self.max_estimator_function = max_estimator_function


class ActivationsRangeEstimatorParameters:
    def __init__(self, min_batch_aggregator: BATCH_AGGREGATION_FUNCTION = BATCH_AGGREGATION_FUNCTION.MEAN,
                 min_estimator_function: ACTIVATIONS_ESTIMATOR_FUNCTION = ACTIVATIONS_ESTIMATOR_FUNCTION.MIN,
                 max_batch_aggregator: BATCH_AGGREGATION_FUNCTION = BATCH_AGGREGATION_FUNCTION.MEAN,
                 max_estimator_function: ACTIVATIONS_ESTIMATOR_FUNCTION = ACTIVATIONS_ESTIMATOR_FUNCTION.MAX):
        self.min_batch_aggregator = min_batch_aggregator
        self.min_estimator_function = min_estimator_function
        self.max_batch_aggregator = max_batch_aggregator
        self.max_estimator_function = max_estimator_function


class StatisticsAggregationParameters:
    def __init__(self,
                 statistics_aggregation_func: STATISTICS_AGGREGATION_FUNCTION = STATISTICS_AGGREGATION_FUNCTION.MEAN):
        self.statistics_aggregation_func = statistics_aggregation_func


class PostTrainingQuantizationParameters(AlgorithmParameters):
    def __init__(self,
                 preset: PRESET = PRESET.MIXED,
                 weight_bits: int = 8,
                 weight_granularity: GRANULARITY = GRANULARITY.PERCHANNEL,
                 weight_range_estimator: WeightsRangeEstimatorParameters = WeightsRangeEstimatorParameters(),
                 activation_bits: int = 8,
                 activation_granularity: GRANULARITY = GRANULARITY.PERTENSOR,
                 activation_range_estimator: ActivationsRangeEstimatorParameters = ActivationsRangeEstimatorParameters(),
                 statistics_aggregation_function: StatisticsAggregationParameters = StatisticsAggregationParameters(),
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
            weight_min_func=weight_range_estimator.min_estimator_function,
            weight_max_func=weight_range_estimator.max_estimator_function,
            activation_min_func=activation_range_estimator.min_estimator_function,
            activation_max_func=activation_range_estimator.max_estimator_function,
            batch_aggregation_min_func=activation_range_estimator.min_batch_aggregator,
            batch_aggregation_max_func=activation_range_estimator.max_batch_aggregator,
            statistics_aggregator_func=statistics_aggregation_function.statistics_aggregation_func,
            weight_quantizer_config=self.weight_quantizer_config,
            activation_quantizer_config=self.activation_quantizer_config,
            ignored_scopes=ignored_scopes,
            target_device=target_device
        ),
            PostTrainingAlgorithms.BiasCorrection: BiasCorrectionAlgorithmParameters()}  # type: Dict[InitializationAlgorithms, InitizalizationParameters]

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

        weights_per_channel, activation_per_channel = None, None
        if weights_granularity == GRANULARITY.PERCHANNEL:
            weights_per_channel = True if weights_granularity == GRANULARITY.PERCHANNEL else None
        elif weights_granularity == GRANULARITY.PERTENSOR:
            weights_per_channel = False
        if activations_granularity == GRANULARITY.PERCHANNEL:
            activation_per_channel = True
        elif activations_granularity == GRANULARITY.PERTENSOR:
            activation_per_channel = False
        self.weight_quantizer_config = QuantizerConfig(num_bits=weight_bits, mode=weight_mode,
                                                       per_channel=weights_per_channel)
        self.activation_quantizer_config = QuantizerConfig(num_bits=activation_bits, mode=activation_mode,
                                                           per_channel=activation_per_channel)
