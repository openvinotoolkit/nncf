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
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, TypeVar, Union

import numpy as np

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.logging import nncf_logger
from nncf.common.quantization.quantizer_removal import find_quantizer_nodes_to_cut
from nncf.common.quantization.quantizer_removal import revert_operations_to_floating_point_precision
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.common.utils.timer import timer
from nncf.data.dataset import Dataset
from nncf.quantization.algorithms.accuracy_control.backend import AccuracyControlAlgoBackend
from nncf.quantization.algorithms.accuracy_control.evaluator import Evaluator
from nncf.quantization.algorithms.accuracy_control.rank_functions import create_normalized_mse_func
from nncf.quantization.passes import remove_shapeof_subgraphs

TModel = TypeVar("TModel")
TTensor = TypeVar("TTensor")


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


class Ranker:
    """
    Encapsulates logic to rank groups of quantizers.
    """

    def __init__(
        self,
        ranking_subset_size: int,
        dataset: Dataset,
        algo_backend: AccuracyControlAlgoBackend,
        evaluator: Evaluator,
        ranking_fn: Optional[Callable[[Any, Any], float]] = None,
    ):
        """
        :param ranking_subset_size: The number of data items that will be selected from
            the dataset to rank groups of quantizers. The `len(dataset)` data items will
            be selected if `ranking_subset_size` parameter is greater than the number of
            elements in the dataset.
        :param dataset: Dataset for the ranking process.
        :param algo_backend: The `AccuracyControlAlgoBackend` algo backend.
        :param evaluator: Evaluator to validate model.
        :param  ranking_fn: a function that compares values returned by
            `Evaluator.collect_values_for_each_item()` method for initial and quantized model.
        """
        self._ranking_subset_size = ranking_subset_size
        self._dataset = dataset
        self._algo_backend = algo_backend
        self._evaluator = evaluator
        self._ranking_fn = ranking_fn
        # We don't need to re-calculate values for the initial model
        # because they don't change. So use this attribute to store
        # them to improve execution time.
        self._reference_values_for_each_item = None

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
        reference_values_for_each_item: Union[List[float], List[List[TTensor]], None],
        approximate_values_for_each_item: Union[List[float], List[List[TTensor]], None],
    ) -> List[GroupToRank]:
        """
        Ranks groups of quantizers by their contribution to accuracy drop. Returns a list of
        ranked groups where `ranked_groups[-1]` group of quantizers has maximal ranking
        score i.e. its contribution to accuracy drop is the greatest.

        :param groups_to_rank: Groups of quantizers that should be ranked.
        :param initial_model: Initial not quantized model.
        :param quantized_model: Quantized model.
        :param quantized_model_graph: NNCF graph for quantized model.
        :param reference_values_for_each_item: List of reference values.
        :param approximate_values_for_each_item: List of approximate values.
        :return: List of ranked groups of quantizers.
        """
        if self._ranking_fn is None:
            self._ranking_fn = self._create_ranking_fn(get_backend(initial_model))

        if reference_values_for_each_item is None:
            if self._reference_values_for_each_item is None:
                nncf_logger.info("Collecting metrics for each data item using an initial model")
                with timer():
                    self._reference_values_for_each_item = self._evaluator.collect_values_for_each_item(
                        initial_model, self._dataset
                    )
        else:
            self._reference_values_for_each_item = reference_values_for_each_item

        if approximate_values_for_each_item is None:
            nncf_logger.info("Collecting metrics for each data item using a quantized model")
            approximate_values_for_each_item = self._evaluator.collect_values_for_each_item(
                quantized_model, self._dataset
            )

        # Create a subset of data items that will be used to rank groups of quantizers.
        scores = [
            self._ranking_fn(ref_val, approx_val)
            for ref_val, approx_val in zip(self._reference_values_for_each_item, approximate_values_for_each_item)
        ]

        ranking_subset_indices = get_ranking_subset_indices_pot_version(scores, self._ranking_subset_size)

        nncf_logger.info("Calculating ranking score for groups of quantizers")
        with timer():
            # Calculate ranking score for groups of quantizers.
            ranking_scores = []  # ranking_scores[i] is the ranking score for groups_to_rank[i]
            for current_group in groups_to_rank:
                modified_model = revert_operations_to_floating_point_precision(
                    current_group.operations, current_group.quantizers, quantized_model, quantized_model_graph
                )
                # Calculate the ranking score for the current group of quantizers.
                ranking_score = self._calculate_ranking_score(modified_model, ranking_subset_indices)
                ranking_scores.append(float(ranking_score))

        # Rank groups.
        ranked_groups = [group for _, group in sorted(zip(ranking_scores, groups_to_rank), key=operator.itemgetter(0))]

        return ranked_groups

    def _calculate_ranking_score(self, modified_model: TModel, ranking_subset_indices: List[int]) -> float:
        """
        Calculates the ranking score for the current group of quantizers.

        :param modified_model: Model from which the current group of quantizers was removed.
        :param ranking_subset_indices: Indices of the `ranking_data_items` in the whole dataset.
        :return: The ranking score for the current group of quantizers.
        """
        if self._evaluator.is_metric_mode():
            # Calculate ranking score based on metric
            ranking_score, _ = self._evaluator.validate(modified_model, self._dataset, ranking_subset_indices)
        else:
            # Calculate ranking score based on differences in logits
            approximate_outputs = self._evaluator.collect_values_for_each_item(
                modified_model, self._dataset, ranking_subset_indices
            )
            reference_outputs = [self._reference_values_for_each_item[i] for i in ranking_subset_indices]
            errors = [self._ranking_fn(a, b) for a, b in zip(reference_outputs, approximate_outputs)]
            ranking_score = sum(errors) / len(errors)

        return ranking_score

    def _create_ranking_fn(self, backend: BackendType) -> Callable[[List[TTensor], List[TTensor]], float]:
        """
        Creates ranking function.

        :return: The ranking function.
        """
        if self._evaluator.is_metric_mode():
            ranking_fn = operator.sub
            metric_name = "ORIGINAL"
        else:
            ranking_fn = create_normalized_mse_func(backend)
            metric_name = "NMSE"
        nncf_logger.info(f"{metric_name} metric is used to rank quantizers")

        return ranking_fn
