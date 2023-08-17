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
import operator
from typing import Any, Callable, Dict, Iterable, List, Tuple, Type, TypeVar, Union

from nncf.common.factory import NNCFGraphFactory
from nncf.common.factory import StatisticsAggregatorFactory
from nncf.common.graph.graph import NNCFGraph
from nncf.common.logging import nncf_logger
from nncf.common.utils.backend import get_backend
from nncf.common.utils.timer import timer
from nncf.data.dataset import Dataset
from nncf.quantization.algorithms.accuracy_control.evaluator import Evaluator
from nncf.quantization.algorithms.accuracy_control.evaluator import MetricResults
from nncf.quantization.algorithms.accuracy_control.rank_functions import create_normalized_mse_func
from nncf.quantization.algorithms.accuracy_control.subset_selection import select_subset
from nncf.quantization.algorithms.algorithm import Algorithm

TModel = TypeVar("TModel")
TTensor = TypeVar("TTensor")
CombinationKey = Tuple[int, ...]
Combination = Dict[str, Any]


def create_combinations(param_grid: Dict[str, List[Any]]) -> Dict[CombinationKey, Combination]:
    """
    Creates combinations as follows:
      * All keys in `param_grid` are numbered using integers from 0 to N = len(param_grid)-1
        Order of keys is used. Let key_j is a key from param_grid.keys() that corresponds
        integer j in {0, 1, ..., N}.
      * Set of combination keys (CK) are created as Cartesian product the following sets

          CK = {None, 0, 1, ..., num_val_0} x {None, 0, 1, ..., num_val_1} x ... x {None, 0, 1, ..., num_val_N},

        where num_val_j is a number of values in param_grid[key_j].
      * Creates combination for each combination key. If combination_key[i] is None then parameter with key_i
        name is not changed. Otherwise, the param_grid[key_i][combination_key[i]] value should be included
        to combination as new value for parameter with key_i name.

    :param param_grid: Dictionary with parameters names as keys and list of
        parameter settings to try as values.
    :return: Created combination.
    """
    simple_changes = []
    indices = []

    for param_name, values in param_grid.items():
        indices.append([None, *range(len(values))])
        simple_changes.append([{param_name: v} for v in values])

    combinations: Dict[CombinationKey, Combination] = {}

    for combination_key in itertools.product(*indices):
        combination: Combination = {}
        for param_idx, value_idx in enumerate(combination_key):
            if value_idx is None:
                continue
            combination.update(simple_changes[param_idx][value_idx])

        combinations[combination_key] = combination

    return combinations


def is_dataclass_instance(obj: Any) -> bool:
    """
    Returns `True` if object is a dataclass instance, `False` otherwise.

    :param obj: Object to check.
    :return: `True` if object is a dataclass instance, `False` otherwise.
    """
    return dataclasses.is_dataclass(obj) and not isinstance(obj, type)


def apply_combination(init_params: Dict[str, Any], combination: Combination) -> Dict[str, Any]:
    """
    Applies combination of parameters to initial parameters.

    :param init_params: Initial set of parameters.
    :param combination: Combination of parameters.
    :return: Returns `init_params` where some values of parameters were changed according to
        provided combination.
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


def print_combination_and_score(title: str, combination: Combination, combination_score: float) -> None:
    """
    Prints combination and score.

    :param title: Title.
    :param combination: Combination to print.
    :param combination_score: Score of combination.
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
    param_grid: Dict[str, List[Any]],
) -> CombinationKey:
    """
    Finds best combination.

    :param combinations: Combinations.
    :param combination_score_func: Combination score function.
    :param param_grid: Dictionary with parameters names as keys and list of
        parameter settings to try as values.
    :return: Best combination key.
    """
    best_combination_key = tuple(None for _ in param_grid)
    best_combination_score = None

    for param_idx, (param_name, values) in enumerate(param_grid.items()):
        nncf_logger.info(f"Start search best value for the '{param_name}' parameter")
        values_indices = [None, *range(len(values))]
        param_best_combination_key = None
        param_best_combination_score = None

        for value_idx in values_indices:
            combination_key = (*best_combination_key[:param_idx], value_idx, *best_combination_key[param_idx + 1 :])
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
    This algorithm is used to find a best combination of parameters from `param_grid`.

    The `param_grid` in simple case is a dictionary with parameters names
    as keys and list of parameter settings to try as values.

        param_grid = {
            "param_name": [0.1, 0.2],
        }

    The parameters names should be same as in `algorithm_cls.__init__()` method.
    In case when "param_name" parameter is a dataclass object there is a way to specify settings
    to try for his fields using marker ":"

        param_grid = {
            "param_name:field_a": [10, 20],
            "param_name:field_b:x": [0.1, 0.2],
        }

    In the example above the `param_name` and "param_name:field_b" parameters are dataclasses.
    This rule is applied recursively.

    The algorithm works as follow: let we have the following `param_grid`

        param_grid = {
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
        param_grid: Dict[str, List[Any]],
        calibration_dataset: Dataset,
        validation_fn: Callable[[Any, Iterable[Any]], Tuple[float, Union[None, List[float], List[List[TTensor]]]]],
        subset_size: int,
        initial_metric_results: MetricResults,
        quantized_metric_results: MetricResults,
    ):
        """
        :param algorithm_cls: Class of algorithm.
        :param init_params: Initial set of parameters used to create algorithm.
        :param param_grid: Dictionary with parameters names as keys and list of
            parameter settings to try as values.
        :param calibration_dataset: Dataset used to collect statistics for algorithm.
        :param validation_fn: Validation function used to validated model.
        :param subset_size: Number of data items that should be selected
            from the dataset and used to validate model.
        :param initial_metric_results: Metric results for initial model.
        :param quantized_metric_results: Metric results for quantized with `init_params` model.
        """
        self._algorithm_cls = algorithm_cls
        self._init_params = init_params
        self._param_grid = param_grid
        self._calibration_dataset = calibration_dataset
        self._evaluator = Evaluator(validation_fn)
        self._subset_size = subset_size
        self._initial_metric_results = initial_metric_results
        self._quantized_metric_results = quantized_metric_results

        self._is_metric_mode = isinstance(self._initial_metric_results.values_for_each_item[0], float)

        # # Will be initialized inside `apply()` method
        self._error_fn = None

        # Will be initialized inside `_initialize_algorithms()` method
        self._algorithms: Dict[CombinationKey, Algorithm] = {}
        self._statistic_points = None

        self._calculated_scores: Dict[CombinationKey, float] = {}

    def apply(self, model: TModel, validation_dataset: Dataset) -> TModel:
        """
        Applies algorithm to input model.

        :param model: Input model.
        :param validation_dataset: Dataset used to validate resulted model.
        :return: Resulted model.
        """
        if self._is_metric_mode:
            self._error_fn = operator.sub
        else:
            self._error_fn = create_normalized_mse_func(get_backend(model))

        subset_indices = select_subset(
            self._subset_size,
            self._initial_metric_results.values_for_each_item,
            self._quantized_metric_results.values_for_each_item,
            self._error_fn,
        )

        combinations = create_combinations(self._param_grid)

        initial_graph = NNCFGraphFactory.create(model)

        nncf_logger.info("Start initialization of algorithms")
        with timer():
            self._prepare_algorithms(model, initial_graph, combinations)

        combination_score_fn = functools.partial(
            self._calculate_combination_score,
            initial_model=model,
            initial_graph=initial_graph,
            dataset=validation_dataset,
            subset_indices=subset_indices,
        )

        nncf_logger.info("Start search best combination of parameters")
        with timer():
            best_combination_key = find_best_combination(combinations, combination_score_fn, self._param_grid)

        algorithm = self._algorithms[best_combination_key]
        result_model = algorithm.apply(model, initial_graph, self._statistic_points)

        return result_model

    def _prepare_algorithms(
        self, initial_model: TModel, initial_graph: NNCFGraph, combinations: Dict[CombinationKey, Combination]
    ) -> None:
        """
        Creates algorithm for each combination of parameters. Collects statistics for
        created algorithms.

        :param initial_model: Input model used to collect statistics for algorithms.
        :param combinations: Combinations of parameters.
        """
        for combination_key, combination in combinations.items():
            kwargs = apply_combination(self._init_params, combination)
            self._algorithms[combination_key] = self._algorithm_cls(**kwargs)

        # Collect required statistics for created algorithms
        stats_aggregator = StatisticsAggregatorFactory.create(initial_model, self._calibration_dataset)
        for algorithm in self._algorithms.values():
            statistic_points = algorithm.get_statistic_points(initial_model, initial_graph)
            stats_aggregator.register_statistic_points(statistic_points)
        stats_aggregator.collect_statistics(initial_model, initial_graph)
        self._statistic_points = stats_aggregator.statistic_points

    def _calculate_combination_score(
        self,
        combination_key: CombinationKey,
        initial_model: TModel,
        initial_graph: NNCFGraph,
        dataset: Dataset,
        subset_indices: List[int],
    ) -> float:
        """
        Calculates score for provided combination.

        :param combination_key: Combination key.
        :param initial_model: Input model.
        :param dataset: Dataset used to select data items for validation.
        :param subset_indices: Zero-based indices of data items that should be selected
            from the dataset and used to validate model.
        :return: Calculated score.
        """
        if combination_key in self._calculated_scores:
            return self._calculated_scores[combination_key]

        algorithm = self._algorithms[combination_key]
        model = algorithm.apply(initial_model, initial_graph, self._statistic_points)
        score = self._validate_model(model, dataset, subset_indices)
        self._calculated_scores[combination_key] = score

        return score

    def _validate_model(self, model: TModel, dataset: Dataset, subset_indices: List[int]) -> float:
        """
        Validates input model on subset.

        :param model: Input model.
        :param dataset: Dataset used to select data items for validation.
        :param subset_indices: Zero-based indices of data items that should be selected
            from the dataset and used to validate model.
        :return: Calculated metric.
        """
        if self._is_metric_mode:
            metric_value, _ = self._evaluator.validate(model, dataset, subset_indices)
        else:
            approximate_outputs = self._evaluator.collect_values_for_each_item(model, dataset, subset_indices)
            reference_outputs = [self._initial_metric_results.values_for_each_item[i] for i in subset_indices]
            errors = [self._error_fn(a, b) for a, b in zip(reference_outputs, approximate_outputs)]
            metric_value = sum(errors) / len(errors)

        return metric_value
