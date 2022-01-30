from typing import List

from nncf.common.utils.priority_queue import PriorityQueue
from nncf.experimental.post_training_api.compressed_model import CompressedModel
from nncf.experimental.post_training_api.quantization.initialization.algorithms.algorithm import InitializationAlgorithm

# TODO: make ENUM
DefaultPriority = 0


class Initializer:
    """
    The base class controls the initialization flow of the model.
    """

    def __init__(self, engine, dataloader, initializer_config):
        self.engine = engine
        self.dataloader = dataloader
        self.initializer_config = initializer_config

        algorithms = self._create_algorithms_from_config(self.initializer_config)
        algorithms_with_priority = [self._set_algorithm_priority(algorithm) for algorithm in algorithms]

        self.algorithms = PriorityQueue(algorithms_with_priority)

    def initialize_model(self, compressed_model: CompressedModel):
        while not self.algorithms.is_empty():
            algorithm = self.algorithms.pop()
            compressed_model = self._run_algorithm(algorithm, compressed_model)
        return compressed_model

    def _run_algorithm(self, algorithm: InitializationAlgorithm, model: CompressedModel) -> CompressedModel:
        return algorithm.run(model)

    def add_algorithm(self, algorithm: InitializationAlgorithm, priority: int = DefaultPriority) -> None:
        self._set_algorithm_priority(algorithm, priority)
        self.algorithms.add(algorithm)

    def _set_algorithm_priority(self, algorithm: InitializationAlgorithm,
                                priority: int = DefaultPriority) -> InitializationAlgorithm:
        """
        Should define the algorithm priority based on the DefaultPriorities for algos
        """

    def _create_algorithms_from_config(self, initializer_config) -> List[InitializationAlgorithm]:
        """
        Should define the default algorithms if the initializer_config is None
        """
