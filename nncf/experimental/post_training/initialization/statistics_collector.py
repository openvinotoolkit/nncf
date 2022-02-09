from abc import ABC
from abc import abstractmethod

from typing import List
from typing import TypeVar
from typing import Optional
from typing import Callable

from nncf.experimental.post_training.compressed_model import CompressedModel
from nncf.experimental.post_training.api.engine import Engine

TensorType = TypeVar('TensorType')


class CalculateTensorValueFunc(ABC):
    @staticmethod
    @abstractmethod
    def __call__(tensor: TensorType, axis: int):
        pass


class LayerStatistic(ABC):
    def __init__(self, layer_name: str,
                 min_value_func: CalculateTensorValueFunc,
                 max_value_func: CalculateTensorValueFunc,
                 axis: Optional[int] = None):
        self.layer_name = layer_name
        self.min_value_func = min_value_func
        self.max_value_func = max_value_func
        self.axis = axis
        self.min_values = []  # type: List[TensorType]
        self.max_values = []  # type: List[TensorType]

    def add_tensor_statistic(self, tensor: TensorType) -> None:
        self.min_values.append(self.min_value_func.__call__(tensor, axis=self.axis))
        self.max_values.append(self.max_value_func.__call__(tensor, axis=self.axis))

    @abstractmethod
    def get_global_min_value(self):
        pass

    @abstractmethod
    def get_global_max_value(self):
        pass


class StatisticsCollector(ABC):
    def __init__(self, compressed_model: CompressedModel, engine: Engine):
        self.compressed_model = compressed_model
        self.engine = engine
        self.is_calculate_metric = False
        self.layers_statistics = []  # type: List[LayerStatistic]

    @abstractmethod
    def collect_statistics(self, layers_to_collect_statistics: List[str], num_iters: int) -> None:
        pass
