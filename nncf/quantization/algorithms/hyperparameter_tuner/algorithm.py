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
from nncf.quantization.algorithms.hyperparameter_tuner.params_transformation import ParamsTransformation
from nncf.quantization.algorithms.hyperparameter_tuner.params_transformation import create_combinations
from nncf.quantization.algorithms.hyperparameter_tuner.params_transformation import create_params_transformation

SearchSpace = Dict[str, Union[List[Any], "SearchSpace"]]
TModel = TypeVar("TModel")


class HyperparameterTuner:
    """
    Algorithm used to find a best combination of provided parameters.
    Possible values of parameters are represented as the `SearchSpace`.
    The `SearchSpace` has the following structure

        {
            "param_name": [v_0, v_1, ..., v_n]
        }

    where v_0, v_1, ..., v_n are possible values of "param_name" parameter.
    In case when "param_name" is a dataclass object there is a way to specify
    possible values for his fields

        {
            "param_name": {
                "field_name_0": [x0, x1, ..., x_k],
                "field_name_1": [y_0, y_1, ..., y_m]
            }
        }

    This rule is applied recursively.
    """

    def __init__(
        self,
        algorithm_cls: Type[Algorithm],
        init_params: Dict[str, Any],
        search_space: SearchSpace,
        calibration_dataset: Dataset,
        validation_fn: Callable[[Any, Iterable[Any]], float],
    ):
        """
        :param algorithm_cls: Class of algorithm.
        :param init_params: Initial set of parameters used to create algorithm.
        :param search_space: Describes possible values for parameters.
        :param calibration_dataset: Dataset used to collect statistics for algorithm.
        :param validation_fn: Validation function used to validated model.
        """
        self._algorithm_cls = algorithm_cls
        self._init_params = init_params
        self._search_space = search_space
        self._calibration_dataset = calibration_dataset
        self._validation_fn = validation_fn
        self._backend_entity = None

    def apply(self, model: TModel, validation_dataset: Dataset, subset_indices: List[int]) -> TModel:
        """
        Applies algorithm to provided model.

        :param model: Model to apply the algorithm.
        :param validation_dataset: Dataset used to validate resulted model.
        :param subset_indices: Zero-based indices of data items that should be selected
            from the dataset and used to validate model.
        :return: Resulted model.
        """
        self._set_backend_entity(model)

        params_transformations = create_params_transformation(self._search_space)
        combinations = create_combinations(params_transformations)
        algorithms = HyperparameterTuner._create_algorithms(self._algorithm_cls, self._init_params, combinations)
        statistic_points = HyperparameterTuner._collect_statistics(model, self._calibration_dataset, algorithms)

        best_score = None
        best_combination = ()

        for param_name, transformations in params_transformations.items():
            nncf_logger.info(f"Searching best value for the {param_name} parameter")

            param_best_score = None
            param_best_value = None

            for param_value in range(len(transformations)):
                algorithm_key = (*best_combination, param_value)
                algorithm = algorithms[algorithm_key]

                nncf_logger.info(f"Current combination:\n{combinations[algorithm_key].as_str()}")

                curr_model = algorithm.apply(copy_model(model), statistic_points)
                score, _ = self._validation_fn(
                    self._backend_entity.prepare_for_inference(curr_model), validation_dataset.get_data(subset_indices)
                )

                nncf_logger.info(f"Score: {score}")

                if param_best_score is None or param_best_score < score:
                    param_best_score = score
                    param_best_value = param_value

            # Include `param_value` into best combination
            if best_score is None or best_score <= param_best_score:
                best_score = param_best_score
                best_combination = (*best_combination, param_best_value)

        # Apply best combination
        nncf_logger.info(f"Best combination:\n{combinations[best_combination].as_str()}")
        nncf_logger.info(f"Best score: {best_score}")
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
            from nncf.quantization.algorithms.hyperparameter_tuner.openvino_backend import (
                OVHyperparameterTunerAlgoBackend,
            )

            self._backend_entity = OVHyperparameterTunerAlgoBackend()
        else:
            raise RuntimeError(f"Cannot set backend-specific entity because {model_backend} is not supported!")

    @staticmethod
    def _create_algorithms(
        algorithm_cls: Type[Algorithm],
        init_params: Dict[str, Any],
        combinations: Dict[Tuple[int, ...], ParamsTransformation],
    ) -> Dict[Tuple[int, ...], Algorithm]:
        """
        Creates algorithm for each combination of parameters from `combinations` list.

        :param algorithm_cls: Class of algorithm.
        :param init_params: Initial set of parameters used to create algorithm.
        :param combinations: Combinations of parameters.
        :return: List of created algorithms.
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
        Collects statistics using common statistics points for `algorithms`.

        :param model: Model used to collect statistics.
        :param dataset: Dataset used to collect statistics.
        :param algorithms: List of algorithms for which statistics should be collected.
        :return: Collected statistics.
        """
        stats_aggregator = StatisticsAggregatorFactory.create(model, dataset)
        for algorithm in algorithms.values():
            stats_aggregator.register_statistic_points(algorithm.get_statistic_points(model))
        stats_aggregator.collect_statistics(model)

        return stats_aggregator.statistic_points
