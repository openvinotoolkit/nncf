# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import csv
from abc import abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, NoReturn, Optional, Tuple, TypeVar

from nncf.common.logging import nncf_logger
from nncf.common.utils.os import safe_open

DataLoaderType = TypeVar("DataLoaderType")
TModel = TypeVar("TModel")
EvalFnType = Callable[[TModel], float]
AccValFnType = Callable[[TModel, DataLoaderType], float]


class BNASEvaluatorStateNames:
    BNAS_EVALUATOR_STAGE = "evaluator_state"


class BaseEvaluator:
    """
    An interface for handling measurements collected on a target device. Evaluators make use
    of functions provided by the users to measure a particular property, e.g., accuracy, latency, etc.
    """

    def __init__(self, name: str, ideal_val: float):
        """
        Initializes evaluator

        :param name: Name of the evaluator
        :param ideal_val: Ideal value for the metric computed by the evaluator
        """
        self.name = name
        self._current_value = -1
        self._ideal_value = ideal_val
        self.cache = {}
        # TODO(pablo): Here we should store some super-network signature that is associated with this evaluator

    @property
    def current_value(self):
        """
        :return: current value
        """
        return self._current_value

    @current_value.setter
    def current_value(self, val: float) -> NoReturn:
        """
        :param val: value to update the current value of the evaluator
        :return:
        """
        self._current_value = val

    def evaluate_and_add_to_cache_from_pymoo(self, pymoo_repr: Tuple[float, ...]) -> float:
        """
        Evaluates active sub-network and uses Pymoo representation for insertion in cache.

        :param model: Active sub-network
        :param pymoo_repr: tuple representing the associated values for the design variables
                            in Pymoo.
        :return: the value obtained from the model evaluation.
        """
        self._current_value = self.evaluate_subnet()
        self.add_to_cache(pymoo_repr, self._current_value)
        return self._current_value

    @abstractmethod
    def evaluate_subnet(self) -> float:
        """This method should implement how to a subnet is evaluated for a particular metric."""

    def add_to_cache(self, subnet_config_repr: Tuple[float, ...], measurement: float) -> NoReturn:
        """
        Adds evaluation result to cache

        :param subnet_config_repr: tuple containing the values for the associated design variables.
        :param measurement: value for the evaluator's metric.
        :return:
        """
        nncf_logger.debug(f"Add to evaluator {self.name}: {subnet_config_repr}, {measurement}")
        self.cache[subnet_config_repr] = measurement

    def retrieve_from_cache(self, subnet_config_repr: Tuple[float, ...]) -> Tuple[bool, float]:
        """
        Checks if sub-network info is in cache and returns the corresponding value.
        :param subnet_config_repr: tuple representing the values for the associated design variables.
        :return: (True if the information is in cache, and corresponding value stored in cache, 0 otherwise)
        """
        if subnet_config_repr in self.cache:
            return True, self.cache[subnet_config_repr]
        return False, 0

    def get_state(self) -> Dict[str, Any]:
        """
        Returns state of the evaluatar

        :return: Dict with the state of the evaluator
        """
        state_dict = {
            "name": self.name,
            "current_value": self._current_value,
            "ideal_value": self._ideal_value,
            "cache": self.cache,
        }
        return state_dict

    def update_from_state(self, state: Dict[str, Any]) -> NoReturn:
        """
        Updates the cache and other values in the evaluator from a saved state.

        :param state: dict with state that should be used for updating this evaluator
        :return:
        """
        new_dict = state.copy()
        self.name = new_dict["name"]
        self._ideal_value = new_dict["ideal_value"]
        self._current_value = new_dict["current_value"]
        self.cache = new_dict["cache"]

    def load_cache_from_csv(self, cache_file_path: str) -> NoReturn:
        """
        Loads cache from CSV file.

        :param cache_file_path: Path to CSV file containing the cache information.
        :return:
        """
        with safe_open(Path(cache_file_path), "r", encoding="utf8") as cache_file:
            reader = csv.reader(cache_file)
            for row in reader:
                rep_tuple = tuple(map(int, row[0][1 : len(row[0]) - 1].split(",")))
                self.add_to_cache(rep_tuple, float(row[1]))

    def export_cache_to_csv(self, cache_file_path: str) -> NoReturn:
        """
        Exports cache information to CSV.

        :param cache_file_path: Path to export a CSV file with the cache information.
        :return:
        """
        with safe_open(Path(cache_file_path) / f"cache_{self.name}.csv", "w", encoding="utf8") as cache_dump:
            writer = csv.writer(cache_dump)
            for key in self.cache:
                row = [key, self.cache[key]]
                writer.writerow(row)


class MACsEvaluator(BaseEvaluator):
    def __init__(self, eval_func):
        super().__init__("MACs", 0)
        self.eval_func = eval_func

    def evaluate_subnet(self) -> float:
        """
        Evaluates metric using model

        :param model: Active sub-network
        :return: value obtained from evaluation.
        """
        self._current_value = self.eval_func()
        return self._current_value


class AccuracyEvaluator(BaseEvaluator):
    """
    A particular kind of evaluator for collecting model's accuracy measurements
    """

    def __init__(
        self, model: TModel, eval_func: AccValFnType, val_loader: DataLoaderType, is_top1: Optional[bool] = True
    ):
        """
        Initializes Accuracy operator

        :param eval_func: function used to validate a sub-network
        :param val_loader: Datq loader used by the validation function
        :param is_top1: Whether is top 1 accuracy or top 5.
        :param ref_acc: Accuracy from a model that is used as input to BootstrapNAS
        """
        name = "top1_acc" if is_top1 else "top5_acc"
        super().__init__(name, -100)
        self._model = model
        self._eval_func = eval_func
        self._val_loader = val_loader
        self._is_top1 = is_top1

    def evaluate_subnet(self) -> float:
        """
        Obtain accuracy from evaluating the model.
        :param model: Active sub-network
        :return: accuracy from active sub-network.
        """
        self._current_value = self._eval_func(self._model, self._val_loader) * -1.0
        return self._current_value

    def get_state(self) -> Dict[str, Any]:
        """
        Get state of Accuracy evaluator.

        :return: Dict with state of evaluator
        """
        state = super().get_state()
        state["is_top1"] = self._is_top1
        return state

    def update_from_state(self, state: Dict[str, Any]) -> NoReturn:
        """

        :param state: dict with state that should be used for updating this evaluator
        :return:
        """

        super().update_from_state(state)
        new_dict = state.copy()
        self._is_top1 = new_dict["is_top1"]
