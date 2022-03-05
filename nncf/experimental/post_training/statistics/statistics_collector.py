from abc import ABC
from abc import abstractmethod

from typing import TypeVar

from typing import Dict

from nncf.common.tensor_statistics.collectors import TensorStatisticCollectorBase
from nncf.experimental.post_training.api.engine import Engine

TensorType = TypeVar('TensorType')
ModelType = TypeVar('ModelType')


class StatisticsCollector(ABC):
    # TODO: should be reviewed.
    """
    Base class for statistics collection.
    """

    def __init__(self, engine: Engine, number_iterations: int):
        self.engine = engine
        self.number_iterations = number_iterations
        self.is_calculate_metric = False
        self.layers_statistics = {}  # type: Dict[str, TensorStatisticCollectorBase]

    @abstractmethod
    def collect_statistics(self, model: ModelType) -> None:
        """
        Collects statistics for layers determined in self.layers_statistics.
        """

    def register_layer_statistics(self, layer_statistics: Dict[str, TensorStatisticCollectorBase]):
        """
        Registered layer for statistics collection.
        """
        # TODO: potentially could be intersection in layers_to_collect_statistics
        self.layers_statistics = layer_statistics
