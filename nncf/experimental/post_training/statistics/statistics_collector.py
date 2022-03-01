from abc import ABC
from abc import abstractmethod

from typing import List
from typing import TypeVar

from typing import Dict

from nncf.experimental.post_training.api.engine import Engine
from nncf.experimental.onnx.statistics.collectors import ONNXMinMaxStatisticCollector

TensorType = TypeVar('TensorType')


class StatisticsCollector(ABC):
    def __init__(self, engine: Engine):
        self.engine = engine
        self.is_calculate_metric = False
        self.layers_statistics = {}  # type: Dict[str, ONNXMinMaxStatisticCollector]

    @abstractmethod
    def collect_statistics(self, layers_to_collect_statistics: List[str], num_iters: int):
        pass

    def register_layer_statistics(self, layer_statistics: Dict[str, ONNXMinMaxStatisticCollector]):
        # TODO: potentially could be intersection in layers_to_collect_statistics
        self.layers_statistics = layer_statistics
