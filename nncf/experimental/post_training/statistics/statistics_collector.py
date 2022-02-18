from abc import ABC
from abc import abstractmethod

from typing import List
from typing import TypeVar
from typing import Optional
from typing import Callable
from typing import Type

from nncf.common.utils.ordered_enum import OrderedEnum
from nncf.experimental.post_training.compressed_model import CompressedModel
from nncf.experimental.post_training.api.engine import Engine

TensorType = TypeVar('TensorType')


class WEIGHTS_ESTIMATOR_FUNCTION(OrderedEnum):
    MIN = 'min'
    MAX = 'max'


class ACTIVATIONS_ESTIMATOR_FUNCTION(OrderedEnum):
    MAX = 'max'
    MIN = 'min'
    MEAN = 'mean'


class BATCH_AGGREGATION_FUNCTION(OrderedEnum):
    MEAN = 'mean'


class STATISTICS_AGGREGATION_FUNCTION(OrderedEnum):
    MIN_MAX = 'min_max'
    MEAN = 'mean'


class CalculateTensorValueFunc(ABC):
    @staticmethod
    @abstractmethod
    def __call__(tensor: TensorType, axis: int):
        pass


class BatchAggregatorFunc(ABC):
    @staticmethod
    @abstractmethod
    def __call__(tensor: TensorType):
        pass


class StatisticsCalculationFunc(ABC):
    @staticmethod
    @abstractmethod
    def __call__(tensors: List[TensorType]):
        pass


class MinMaxLayerStatistic(ABC):
    def __init__(self, layer_name: str,
                 min_value_func: Type[CalculateTensorValueFunc],
                 max_value_func: Type[CalculateTensorValueFunc],
                 min_batch_aggregator_func: Type[BatchAggregatorFunc],
                 max_batch_aggregator_func: Type[BatchAggregatorFunc],
                 statistics_aggregation_func: Type[StatisticsCalculationFunc],
                 axis: Optional[int] = None):
        self.layer_name = layer_name
        self.min_value_func = min_value_func
        self.max_value_func = max_value_func
        self.min_batch_aggregator_func = min_batch_aggregator_func
        self.max_batch_aggregator_func = max_batch_aggregator_func
        self.statistics_aggregation_func = statistics_aggregation_func
        self.axis = axis
        self.min_values = []  # type: List[TensorType]
        self.max_values = []  # type: List[TensorType]

    def add_tensor_statistic(self, tensor: TensorType) -> None:
        batch_min_value = self.min_batch_aggregator_func.__call__(tensor)
        batch_max_value = self.max_batch_aggregator_func.__call__(tensor)
        self.min_values.append(self.min_value_func.__call__(batch_min_value, axis=self.axis))
        self.max_values.append(self.max_value_func.__call__(batch_max_value, axis=self.axis))

    def get_global_min_value(self):
        return self.statistics_aggregation_func.__call__(self.min_values)

    def get_global_max_value(self):
        return self.statistics_aggregation_func.__call__(self.max_values)


class LayerStatistic(ABC):
    def __init__(self, layer_name: str,
                 axis: Optional[int] = None):
        self.layer_name = layer_name
        self.values = []

    def add_tensor_statistic(self, tensor: TensorType) -> None:
        self.values.append(tensor)


class StatisticsCollector(ABC):
    def __init__(self, engine: Engine):
        self.engine = engine
        self.is_calculate_metric = False
        self.layers_statistics = []  # type: List[MinMaxLayerStatistic]

    @abstractmethod
    def collect_statistics(self, layers_to_collect_statistics: List[str], num_iters: int) -> List[MinMaxLayerStatistic]:
        pass

    def register_layer_statistics(self, layer_statistics: List[MinMaxLayerStatistic]):
        # TODO: potentially could be intersection in layers_to_collect_statistics
        self.layers_statistics.extend(layer_statistics)
