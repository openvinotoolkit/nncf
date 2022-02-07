from abc import ABC
from abc import abstractmethod

from typing import List
from typing import TypeVar

from nncf.experimental.post_training.compressed_model import CompressedModel
from nncf.experimental.post_training.api.engine import Engine

TensorType = TypeVar('TensorType')


class StatisticsCollector(ABC):
    def __init__(self, compressed_model: CompressedModel, engine: Engine):
        self.compressed_model = compressed_model
        self.engine = engine
        self.is_calculate_metric = False
        self.statistics = {}  # type: Dict[str, TensorType]

    @abstractmethod
    def collect_statistics(self, layers_to_collect_statistics: List[str], num_iters: int) -> None:
        pass
