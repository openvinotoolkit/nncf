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

import copy
import dataclasses
import functools
import itertools
from typing import Any, Callable, Dict, Iterable, List, Tuple, Type, TypeVar, Union

from nncf.common.factory import StatisticsAggregatorFactory
from nncf.common.logging import nncf_logger
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.common.utils.timer import timer
from nncf.data.dataset import Dataset
from nncf.quantization.algorithms.algorithm import Algorithm

TModel = TypeVar("TModel")
CombinationKey = Tuple[int, ...]
# Key of combination without the trailing zeros
TrimedCombinationKey = Tuple[int, ...]
Combination = Dict[str, Any]


def create_combinations(param_settings: Dict[str, List[Any]]) -> Dict[CombinationKey, Combination]:
    """
    :param param_settings:
    :return:
    """

    # TODO:Describe how build combination key

    simple_changes = [[{}, *[{param_name: v} for v in values]] for param_name, values in param_settings.items()]

    combinations = {}
    for num_params in range(1, len(simple_changes) + 1):
        for group in itertools.product(*map(enumerate, simple_changes[:num_params])):
            combination_key, members = zip(*group)

            combination = {}
            for m in members:
                combination.update(m)
            combinations[combination_key] = combination

    return combinations


def is_dataclass_instance(obj: Any):
    return dataclasses.is_dataclass(obj) and not isinstance(obj, type)


def apply_combination(init_params: Dict[str, Any], combination: Combination) -> Dict[str, Any]:
    """
    :param init_params:
    :param combination:
    :return:
    """
    DELIMITER = ":"
    params = copy.deepcopy(init_params)
    for param_key, param_value in combination.items():
        if DELIMITER in param_key:
            main_key, *path_to_attr, attr_name = param_key.split(DELIMITER)
            obj = params[main_key]
            assert is_dataclass_instance(obj)
            for name in path_to_attr:
                obj = getattr(obj, name)
                assert is_dataclass_instance(obj)
            setattr(obj, attr_name, param_value)
        else:
            params[param_key] = param_value

    return params


def trim_zeros(t: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Trim the trailing zeros from a tuple.

    :param t: Input tuple.
    :return: The result of trimming the input.
    """
    pos = len(t)
    while pos != 0 and t[pos - 1] == 0:
        pos = pos - 1
    return t[:pos]


def filter_combinations(combinations: Dict[CombinationKey, Combination]) -> Dict[TrimedCombinationKey, Combination]:
    """
    :param combinations:
    :return:
    """
    filtered = {}
    for combination_key, combination in combinations.items():
        trimed_combination_key = trim_zeros(combination_key)
        if trimed_combination_key not in filtered:
            filtered[trimed_combination_key] = combination

    return filtered


def print_combination_and_score(title: str, combination: Combination, combination_score: float) -> None:
    """
    :param title:
    :param combination:
    :param combination_score:
    """
    if not combination:
        message = "Parameters were not changed"
    else:
        message = ", ".join(f"{name} = {v}" for name, v in combination.items())
    message = f"{title} {message}"

    nncf_logger.info(message)
    nncf_logger.info(f"Score: {combination_score}")


def find_best_combination(
    combinations: Dict[CombinationKey, Combination],
    combination_score_func: Callable[[CombinationKey], float],
    param_settings: Dict[str, List[Any]],
) -> CombinationKey:
    """
    :param combinations:
    :param combination_score_func:
    :param param_settings:
    :return:
    """
    best_combination_key = ()
    best_combination_score = None

    for param_name, values in param_settings.items():
        nncf_logger.info(f"Start search best value for the '{param_name}' parameter")

        num_values = len(values) + 1
        param_best_combination_key = None
        param_best_combination_score = None

        for idx in range(num_values):
            combination_key = (*best_combination_key, idx)
            combination_score = combination_score_func(combination_key)

            if param_best_combination_score is None or param_best_combination_score < combination_score:
                param_best_combination_score = combination_score
                param_best_combination_key = combination_key

            print_combination_and_score(
                "Current combination of parameters:", combinations[combination_key], combination_score
            )

        if best_combination_score is None or best_combination_score <= param_best_combination_score:
            best_combination_score = param_best_combination_score
            best_combination_key = param_best_combination_key

    print_combination_and_score(
        "Best combination of parameters:", combinations[best_combination_key], best_combination_score
    )

    return best_combination_key


class HyperparameterTuner:
    """
    This algorithm is used to find a best combination of parameters from `param_settings`.

    The `param_settings` in simple case is a dictionary with parameters names
    as keys and list of parameter settings to try as values.

        param_settings = {
            "param_name": [0.1, 0.2],
        }

    The parameters names should be same as in `algorithm_cls.__init__()` method.
    In case when "param_name" parameter is a dataclass object there is a way to specify settings
    to try for his fields using marker ":"

        param_settings = {
            "param_name:field_a": [10, 20],
            "param_name:field_b:x": [0.1, 0.2],
        }

    In the example above the `param_name` and "param_name:field_b" parameters are dataclasses.
    This rule is applied recursively.

    The algorithm works as follow: let we have the following `param_settings`

        param_settings = {
            "param_name_0" : [0.2, 0.4, 0.6],
            "param_name_1:x": [-1, -2, -3],
            "param_name_2": [True, False],
        }

    First of all, algorithm finds the best value for parameter "param_name_0".
    Further, taking into account the found value, the best value for the "param_name_1:x" parameter
    is sought. After that, taking into account the found values for "param_name_0" and "param_name_1:x"
    parameters, the best value for the "param_name_2" is sought.
    """

    def __init__(
        self,
        algorithm_cls: Type[Algorithm],
        init_params: Dict[str, Any],
        param_settings: Dict[str, List[Any]],
        calibration_dataset: Dataset,
        validation_fn: Callable[[Any, Iterable[Any]], float],
    ):
        """
        :param algorithm_cls: Class of algorithm.
        :param init_params: Initial set of parameters used to create algorithm.
        :param param_settings: Dictionary with parameters names as keys and list of
            parameter settings to try as values.
        :param calibration_dataset: Dataset used to collect statistics for algorithm.
        :param validation_fn: Validation function used to validated model.
        """
        self._algorithm_cls = algorithm_cls
        self._init_params = init_params
        self._param_settings = param_settings
        self._calibration_dataset = calibration_dataset
        self._validation_fn = validation_fn

        # Will be initialized inside `_set_backend_entity()` method
        self._backend_entity = None

        # Will be initialized inside `_initialize_algorithms` method
        self._algorithms = {}  # type: Dict[TrimedCombinationKey, Algorithm]
        self._statistic_points = None

        self._calculated_scores = {}  # type: Dict[TrimedCombinationKey, float]

    def apply(self, model: TModel, validation_dataset: Dataset, subset_indices: List[int]) -> TModel:
        """
        Applies algorithm to input model.

        :param model: Input model.
        :param validation_dataset: Dataset used to validate resulted model.
        :param subset_indices: Zero-based indices of data items that should be selected
            from the dataset and used to validate model.
        :return: Resulted model.
        """
        self._set_backend_entity(model)

        combinations = create_combinations(self._param_settings)

        nncf_logger.info("Start initialization of algorithms")
        with timer():
            self._initialize_algorithms(model, combinations)

        combination_score_fn = functools.partial(
            self._calculate_combination_score,
            initial_model=model,
            dataset=validation_dataset,
            subset_indices=subset_indices,
        )

        nncf_logger.info("Start search best combination of parameters")
        with timer():
            best_combination_key = find_best_combination(combinations, combination_score_fn, self._param_settings)

        algorithm = self._algorithms[trim_zeros(best_combination_key)]
        result_model = algorithm.apply(model, self._statistic_points)

        return result_model

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

    def _initialize_algorithms(self, initial_model: TModel, combinations: Dict[CombinationKey, Combination]) -> None:
        """
        :param initial_model:
        :param combinations:
        """
        # Some combinations in `combinations` produce the same algorithm.
        # More formally, two combination produce the same algorithm if their
        # keys are equal after removing trailing zeros. So it is sufficient to
        # consider only one such algorithm and discard the rest.
        filtered_combinations = filter_combinations(combinations)

        for trimed_combination_key, combination in filtered_combinations.items():
            kwargs = apply_combination(self._init_params, combination)
            self._algorithms[trimed_combination_key] = self._algorithm_cls(**kwargs)

        # Collect required statistics for created algorithms
        stats_aggregator = StatisticsAggregatorFactory.create(initial_model, self._calibration_dataset)
        for algorithm in self._algorithms.values():
            stats_aggregator.register_statistic_points(algorithm.get_statistic_points(initial_model))
        stats_aggregator.collect_statistics(initial_model)
        self._statistic_points = stats_aggregator.statistic_points

    def _calculate_combination_score(
        self, combination_key: CombinationKey, initial_model: TModel, dataset: Dataset, subset_indices: List[int]
    ) -> float:
        """
        :param initial_model:
        :param dataset:
        :param subset_indices:
        :param combination_key:
        :return:
        """
        trimed_combination_key = trim_zeros(combination_key)

        # Calculate score only once for combinations that produce the same algorithm
        if trimed_combination_key in self._calculated_scores:
            return self._calculated_scores[trimed_combination_key]

        algorithm = self._algorithms[trimed_combination_key]
        model = algorithm.apply(initial_model, self._statistic_points)
        model_for_inference = self._backend_entity.prepare_for_inference(model)
        validation_subset = dataset.get_data(subset_indices)
        score, _ = self._validation_fn(model_for_inference, validation_subset)
        self._calculated_scores[trimed_combination_key] = score

        return score
