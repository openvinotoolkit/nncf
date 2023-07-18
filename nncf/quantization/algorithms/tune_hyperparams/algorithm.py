# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, Dict, Iterable, List, Tuple, Type, TypeVar, Union

from nncf.common.factory import StatisticsAggregatorFactory
from nncf.common.logging import nncf_logger
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import copy_model
from nncf.common.utils.backend import get_backend
from nncf.data.dataset import Dataset
from nncf.quantization.algorithms.algorithm import Algorithm
from nncf.quantization.algorithms.tune_hyperparams.params_transformation import ParamsTransformation
from nncf.quantization.algorithms.tune_hyperparams.params_transformation import create_combinations
from nncf.quantization.algorithms.tune_hyperparams.params_transformation import create_params_transformation

SearchSpace = Dict[str, Union[List[Any], "SearchSpace"]]
TModel = TypeVar("TModel")


class ParamsGridSearchAlgorithm:
    """ """

    def __init__(
        self,
        algorithm_cls: Type[Algorithm],
        init_params: Dict[str, Any],
        search_space: SearchSpace,
        statistic_dataset: Dataset,
        validation_fn: Callable[[Any, Iterable[Any]], float],
    ):
        """
        :param algorithm_cls:
        :param init_params:
        :param search_space:
        :param statistic_dataset:
        :param validation_fn:
        """
        self._algorithm_cls = algorithm_cls
        self._init_params = init_params
        self._search_space = search_space
        self._statistic_dataset = statistic_dataset
        self._validation_fn = validation_fn
        self._backend_entity = None

    def apply(self, model: TModel, validation_dataset: Dataset, subset_indices: List[int]) -> TModel:
        """
        :param model:
        :param validation_dataset:
        :param subset_indices:
        :return:
        """
        self._set_backend_entity(model)

        params_transformations = create_params_transformation(self._search_space)
        combinations = create_combinations(params_transformations)
        algorithms = ParamsGridSearchAlgorithm._create_algorithms(self._algorithm_cls, self._init_params, combinations)
        statistic_points = ParamsGridSearchAlgorithm._collect_statistics(model, self._statistic_dataset, algorithms)

        best_score = None
        best_combination = ()

        for param_name, transformations in params_transformations.items():
            nncf_logger.info(f"Searching best value for the {param_name} parameter")

            param_best_score = None
            param_best_value = None

            for param_value in range(len(transformations)):
                algorithm_key = (*best_combination, param_value)
                algorithm = algorithms[algorithm_key]

                nncf_logger.info(f"Current combination: {combinations[algorithm_key]._changes}")

                curr_model = algorithm.apply(copy_model(model), statistic_points)
                score, _ = self._validation_fn(
                    self._backend_entity.prepare_for_inference(curr_model), validation_dataset.get_data(subset_indices)
                )

                nncf_logger.info(f"Score: {score}")

                if param_best_score is None or param_best_score < score:
                    param_best_score = score
                    param_best_value = param_value

            # Include `param_value` into best combination
            if best_score is None or best_score < param_best_score:
                best_score = param_best_score
                best_combination = (*best_combination, param_best_value)

        # Apply best combination
        nncf_logger.info(f"Best combination: {combinations[best_combination]._changes}")
        algorithm = algorithms[best_combination]
        retval = algorithm.apply(model, statistic_points)

        return retval

    def _set_backend_entity(self, model: TModel) -> None:
        """
        Sets an entity with backend-specific logic of the algorithm.

        :param model: Backend-specific model.
        """
        model_backend = get_backend(model)
        if model_backend == BackendType.OPENVINO:
            from nncf.quantization.algorithms.tune_hyperparams.openvino_backend import OVParamsGridSearchAlgoBackend

            self._backend_entity = OVParamsGridSearchAlgoBackend()
        else:
            raise RuntimeError(f"Cannot set backend-specific entity because {model_backend} is not supported!")

    @staticmethod
    def _create_algorithms(
        algorithm_cls: Type[Algorithm],
        init_params: Dict[str, Any],
        combinations: Dict[Tuple[int, ...], ParamsTransformation],
    ) -> Dict[Tuple[int, ...], Algorithm]:
        """
        :param algorithm_cls:
        :param init_params:
        :param combinations:
        :return:
        """
        algorithms = {}
        for combination_key, params_transformation in combinations.items():
            params = params_transformation.apply(init_params)
            algorithms[combination_key] = algorithm_cls(**params)

        return algorithms

    @staticmethod
    def _collect_statistics(
        model: TModel, dataset: Dataset, algorithms: Dict[Tuple[int, ...], Algorithm]
    ) -> StatisticPointsContainer:
        """
        :param model:
        :param dataset:
        :param algorithms:
        :return:
        """
        stats_aggregator = StatisticsAggregatorFactory.create(model, dataset)
        for algorithm in algorithms.values():
            stats_aggregator.register_statistic_points(algorithm.get_statistic_points(model))
        stats_aggregator.collect_statistics(model)

        return stats_aggregator.statistic_points
