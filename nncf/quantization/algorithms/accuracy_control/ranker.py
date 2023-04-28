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

import operator
from abc import ABC
from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, TypeVar

import numpy as np

from nncf.common.factory import EngineFactory
from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.logging import nncf_logger
from nncf.common.quantization.quantizer_removal import find_quantizer_nodes_to_cut
from nncf.common.quantization.quantizer_removal import revert_operations_to_floating_point_precision
from nncf.common.utils.timer import timer
from nncf.data.dataset import Dataset
from nncf.quantization.algorithms.accuracy_control.backend import AccuracyControlAlgoBackend
from nncf.quantization.passes import remove_shapeof_subgraphs

TModel = TypeVar("TModel")


def get_ranking_subset_indices(errors: List[float], ranking_subset_size: int) -> List[int]:
    """
    Returns `ranking_subset_size` indices of elements in the `errors` list
    that have the biggest error value. Returned indices are sorted in
    ascending order.

    :param errors: A list of errors.
    :param ranking_subset_size: A number of returned indices.
    :return: Indices of elements in the `errors` list which have the biggest error value.
    """
    ordered_indices = [idx for idx, _ in sorted(enumerate(errors), key=operator.itemgetter(1), reverse=True)]
    end_index = min(ranking_subset_size, len(ordered_indices))
    return sorted(ordered_indices[:end_index])


def get_ranking_subset_indices_pot_version(errors: List[float], ranking_subset_size: int) -> List[int]:
    """
    POT implementation of the `get_ranking_subset_indices()` method.
    """
    ordered_indices = np.flip(np.argsort(errors)).tolist()
    end_index = min(ranking_subset_size, len(ordered_indices))
    return sorted(ordered_indices[:end_index])


@dataclass
class GroupToRank:
    """
    Describes a group of quantizers to rank.

    :param quantizers: List of quantizer nodes.
    :param operations: List of nodes that will be reverted to original precision
        if `GroupToRank.quantizers` are removed.
    """

    quantizers: List[NNCFNode]
    operations: List[NNCFNode]


class Ranker(ABC):
    """
    Encapsulates logic to rank groups of quantizers.
    """

    def __init__(
        self,
        ranking_subset_size: int,
        ranking_fn: Callable[[Any, Any], float],
        dataset: Dataset,
        algo_backend: AccuracyControlAlgoBackend,
    ):
        """
        :param ranking_subset_size: The number of data items that will be selected from
            the dataset to rank groups of quantizers. The `len(dataset)` data items will
            be selected if `ranking_subset_size` parameter is greater than the number of
            elements in the dataset.
        :param ranking_fn: A function that compares values returned by
            `_collect_values_for_each_item()` for initial and quantized models.
        :param dataset: Dataset for the ranking process.
        :param algo_backend: The `AccuracyControlAlgoBackend` algo backend.
        """
        self._ranking_subset_size = ranking_subset_size
        self._ranking_fn = ranking_fn
        self._dataset = dataset
        self._algo_backend = algo_backend
        # We don't need to re-calculate values for the initial model
        # because they don't change. So use this attribute to store
        # them to improve execution time.
        self._ref_values = None

    def find_groups_of_quantizers_to_rank(self, quantized_model_graph: NNCFGraph) -> List[GroupToRank]:
        """
        Finds groups of quantizers to rank.

        :param quantized_model_graph: Graph for quantized model.
        :return: List of groups of quantizers to rank.
        """
        groups_to_rank = []
        processed = {}
        quantizers = [
            x
            for x in quantized_model_graph.topological_sort()
            if x.metatype in self._algo_backend.get_quantizer_metatypes()
        ]

        quantized_model_graph_without_shapeof = remove_shapeof_subgraphs(
            deepcopy(quantized_model_graph), self._algo_backend.get_shapeof_metatypes()
        )

        for quantizer_node in reversed(quantizers):
            if processed.get(quantizer_node.node_name, False):
                continue
            group, operations = find_quantizer_nodes_to_cut(
                quantized_model_graph_without_shapeof,
                quantizer_node,
                self._algo_backend.get_quantizer_metatypes(),
                self._algo_backend.get_const_metatypes(),
                self._algo_backend.get_quantizable_metatypes(),
                self._algo_backend.get_quantize_agnostic_metatypes(),
            )
            for x in group:
                processed[x.node_name] = True

            groups_to_rank.append(GroupToRank(group, operations))

        return groups_to_rank

    def rank_groups_of_quantizers(
        self,
        groups_to_rank: List[GroupToRank],
        initial_model: TModel,
        quantized_model: TModel,
        quantized_model_graph: NNCFGraph,
    ) -> List[GroupToRank]:
        """
        Ranks groups of quantizers by their contribution to accuracy drop. Returns a list of
        ranked groups where `ranked_groups[-1]` group of quantizers has maximal ranking
        score i.e. its contribution to accuracy drop is the greatest.

        :param groups_to_rank: Groups of quantizers that should be ranked.
        :param initial_model: Initial not quantized model.
        :param quantized_model: Quantized model.
        :param quantized_model_graph: NNCF graph for quantized model.
        :return: List of ranked groups of quantizers.
        """
        # See `Ranker.__init__()` to understand why we should do this.
        if self._ref_values is None:
            nncf_logger.info("Collecting metrics for each data item using an initial model")
            with timer():
                self._ref_values = self._collect_values_for_each_item(initial_model, self._get_data_items())

        nncf_logger.info("Collecting metrics for each data item using a quantized model")
        with timer():
            approx_values = self._collect_values_for_each_item(quantized_model, self._get_data_items())

        # Create a subset of data items that will be used to rank groups of quantizers.
        scores = [self._ranking_fn(ref_val, approx_val) for ref_val, approx_val in zip(self._ref_values, approx_values)]
        ranking_subset_indices = get_ranking_subset_indices_pot_version(scores, self._ranking_subset_size)
        # TODO(andrey-churkin): The ranking subset size usually is small. So it is possible
        # to save all ranking data items in memory and don't read them again.
        ranking_data_items = self._get_data_items(ranking_subset_indices)

        nncf_logger.info("Calculating ranking score for groups of quantizers")
        with timer():
            # Calculate ranking score for groups of quantizers.
            ranking_scores = []  # ranking_scores[i] is the ranking score for groups_to_rank[i]
            for current_group in groups_to_rank:
                modified_model = revert_operations_to_floating_point_precision(
                    current_group.operations, current_group.quantizers, quantized_model, quantized_model_graph
                )
                # Calculate the ranking score for the current group of quantizers.
                ranking_score = self._calculate_ranking_score(
                    modified_model, ranking_data_items, ranking_subset_indices
                )
                ranking_scores.append(float(ranking_score))

        # Rank groups.
        ranked_groups = [group for _, group in sorted(zip(ranking_scores, groups_to_rank), key=operator.itemgetter(0))]

        return ranked_groups

    @abstractmethod
    def _get_data_items(self, indices: Optional[List[int]] = None) -> Iterable[Any]:
        """
        Returns the data items used to validate the model and select the ranking dataset.

        :param indices: The zero-based indices of data items that should be selected from
            the data source.
        :return: Data items.
        """

    @abstractmethod
    def _collect_values_for_each_item(self, model: TModel, data_items: Iterable[Any]) -> List[Any]:
        """
        Collects value for each item from `data_items`. A `value` is calculated using
        model and data item.

        :param model: Model.
        :param data_items: Data items.
        :return: Collected values.
        """

    @abstractmethod
    def _calculate_ranking_score(
        self, modified_model: TModel, ranking_data_items: Iterable[Any], ranking_subset_indices: List[int]
    ) -> float:
        """
        Calculates the ranking score for the current group of quantizers.

        :param modified_model: Model from which the current group of quantizers was removed.
        :param ranking_data_items: Data items for ranking score calculation.
        :param ranking_subset_indices: Indices of the `ranking_data_items` in the whole dataset.
        :return: The ranking score for the current group of quantizers.
        """


class LogitsBasedRanker(Ranker):
    """
    Encapsulates logic to rank groups of quantizers based on differences in logits.
    """

    def _get_data_items(self, indices: Optional[List[int]] = None) -> Iterable[Any]:
        """
        Returns data items from which ranking dat
        """
        return self._dataset.get_inference_data(indices)

    def _collect_values_for_each_item(self, model: TModel, data_items: Iterable[Any]) -> List[Any]:
        """
        Infers `model` for each item from the `dataset` and returns collected logits.

        :param model: A model to be inferred.
        :param data_items: Data items.
        :return: A list that contains logits for each item from the dataset.
        """
        engine = EngineFactory.create(model)
        outputs = [engine.infer(data_item) for data_item in data_items]
        return outputs

    def _calculate_ranking_score(
        self, modified_model: TModel, ranking_data_items: Iterable[Any], ranking_subset_indices: List[int]
    ) -> float:
        approx_values_subset = self._collect_values_for_each_item(modified_model, ranking_data_items)
        ref_values_subset = (self._ref_values[i] for i in ranking_subset_indices)
        errors = [self._ranking_fn(a, b) for a, b in zip(ref_values_subset, approx_values_subset)]
        ranking_score = sum(errors) / len(errors)
        return ranking_score


class MetricBasedRanker(Ranker):
    """
    Encapsulates logic to rank groups of quantizers based on differences in metric.
    """

    def __init__(
        self,
        ranking_subset_size: int,
        ranking_fn: Callable[[Any, Any], float],
        dataset: Dataset,
        algo_backend: AccuracyControlAlgoBackend,
        validation_fn: Callable[[Any, Iterable[Any]], float],
    ):
        """
        :param ranking_subset_size: The number of data items that will be selected from
            the dataset to rank groups of quantizers. The `len(dataset)` data items will
            be selected if `ranking_subset_size` parameter is greater than the number of
            elements in the dataset.
        :param ranking_fn: A function that compares values returned by
            `_collect_values_for_each_item()` for initial and quantized models.
        :param dataset: Dataset for the ranking process.
        :param algo_backend: The `AccuracyControlAlgoBackend` algo backend.
        :param validation_fn: A validation function to validate the model.
            It should take two argumets:
                - `model`: model to be validate.
                - `validation_dataset`: dataset that provides data items to
                validate the provided model.
            The function should return the value of the metric with the following
            meaning: A higher value corresponds to better performance of the model.
        """
        super().__init__(ranking_subset_size, ranking_fn, dataset, algo_backend)
        self._validation_fn = validation_fn

    def _get_data_items(self, indices: Optional[List[int]] = None) -> Iterable[Any]:
        return self._dataset.get_data(indices)

    def _collect_values_for_each_item(self, model: TModel, data_items: Iterable[Any]) -> List[Any]:
        """
        Calls `validation_fn` for each item from the `dataset` and returns collected metrics.

        :param model: The model to be inferred.
        :param data_items: Data items.
        :return: A list that contains a metric for each item from the dataset.
        """
        model_for_inference = self._algo_backend.prepare_for_inference(model)

        metrics = []
        for data_item in data_items:
            value = self._validation_fn(model_for_inference, [data_item])
            metrics.append(value)

        return metrics

    def _calculate_ranking_score(
        self, modified_model: TModel, ranking_data_items: Iterable[Any], ranking_subset_indices: List[int]
    ) -> float:
        ranking_score = self._validation_fn(
            self._algo_backend.prepare_for_inference(modified_model), ranking_data_items
        )
        return ranking_score
