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

import operator
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, TypeVar, Union

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.logging import nncf_logger
from nncf.common.logging.track_progress import track
from nncf.common.quantization.quantizer_removal import find_quantizer_nodes_to_cut
from nncf.common.quantization.quantizer_removal import revert_operations_to_floating_point_precision
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.common.utils.timer import timer
from nncf.data.dataset import Dataset
from nncf.quantization.advanced_parameters import RestoreMode
from nncf.quantization.algorithms.accuracy_control.backend import AccuracyControlAlgoBackend
from nncf.quantization.algorithms.accuracy_control.evaluator import Evaluator
from nncf.quantization.algorithms.accuracy_control.rank_functions import create_normalized_mse_func
from nncf.quantization.algorithms.accuracy_control.subset_selection import select_subset
from nncf.quantization.passes import find_shapeof_subgraphs

TModel = TypeVar("TModel")
TPModel = TypeVar("TPModel")
TTensor = TypeVar("TTensor")


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
        num_workers: int = 1,
        ranking_fn: Optional[Callable[[Any, Any], float]] = None,
        restore_mode: RestoreMode = RestoreMode.ACTIVATIONS_AND_WEIGHTS,
    ):
        """
        :param ranking_subset_size: The number of data items that will be selected from
            the dataset to rank groups of quantizers. The `len(dataset)` data items will
            be selected if `ranking_subset_size` parameter is greater than the number of
            elements in the dataset.
        :param dataset: Dataset for the ranking process.
        :param algo_backend: The `AccuracyControlAlgoBackend` algo backend.
        :param evaluator: Evaluator to validate model.
        :param num_workers: The number of parallel workers that are used to rank quantization
            operations.
        :param  ranking_fn: a function that compares values returned by
            `Evaluator.collect_values_for_each_item()` method for initial and quantized model.
        """
        self._ranking_subset_size = ranking_subset_size
        self._dataset = dataset
        self._algo_backend = algo_backend
        self._evaluator = evaluator
        self._ranking_fn = ranking_fn
        self._num_workers = num_workers
        self._restore_mode = restore_mode

    def find_groups_of_quantizers_to_rank(self, quantized_model_graph: NNCFGraph) -> List[GroupToRank]:
        """
        Finds groups of quantizers to rank.

        :param quantized_model_graph: Graph for quantized model.
        :return: List of groups of quantizers to rank.
        """
        quantizer_metatypes = self._algo_backend.get_quantizer_metatypes()

        if len(quantizer_metatypes) == 2:  # Quantize-Dequantize case
            # Use only Quantize metatype
            quantizer_metatypes = quantizer_metatypes[:1]

        groups_to_rank = []
        processed = {}
        quantizers = [x for x in quantized_model_graph.topological_sort() if x.metatype in quantizer_metatypes]

        input_nodes = [
            *quantized_model_graph.get_nodes_by_metatypes(self._algo_backend.get_op_with_weights_metatypes()),
            *self._algo_backend.get_start_nodes_for_activation_path_tracing(quantized_model_graph),
        ]

        shapeof_subgraphs = find_shapeof_subgraphs(
            quantized_model_graph,
            self._algo_backend.get_shapeof_metatypes(),
            input_nodes,
        )
        quantized_model_graph_without_shapeof = deepcopy(quantized_model_graph)
        quantized_model_graph_without_shapeof.remove_nodes_from(shapeof_subgraphs)

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
        quantized_model: TModel,
        quantized_model_graph: NNCFGraph,
        reference_values_for_each_item: Union[List[float], List[List[TTensor]]],
        approximate_values_for_each_item: Union[List[float], List[List[TTensor]]],
    ) -> List[GroupToRank]:
        """
        Ranks groups of quantizers by their contribution to accuracy drop. Returns a list of
        ranked groups where `ranked_groups[-1]` group of quantizers has maximal ranking
        score i.e. its contribution to accuracy drop is the greatest.

        :param groups_to_rank: Groups of quantizers that should be ranked.
        :param quantized_model: Quantized model.
        :param quantized_model_graph: NNCF graph for quantized model.
        :param reference_values_for_each_item: List of reference values.
        :param approximate_values_for_each_item: List of approximate values.
        :return: List of ranked groups of quantizers.
        """
        if self._ranking_fn is None:
            self._ranking_fn = self._create_ranking_fn(get_backend(quantized_model))

        ranking_subset_indices = select_subset(
            self._ranking_subset_size,
            reference_values_for_each_item,
            approximate_values_for_each_item,
            self._ranking_fn,
        )

        with timer():
            # Calculate ranking score for groups of quantizers.
            if self._num_workers > 1:
                ranking_scores = self._multithreading_calculation_ranking_score(
                    quantized_model,
                    quantized_model_graph,
                    groups_to_rank,
                    ranking_subset_indices,
                    reference_values_for_each_item,
                )

            else:
                ranking_scores = self._sequential_calculation_ranking_score(
                    quantized_model,
                    quantized_model_graph,
                    groups_to_rank,
                    ranking_subset_indices,
                    reference_values_for_each_item,
                )

        # Rank groups.
        ranked_groups = [group for _, group in sorted(zip(ranking_scores, groups_to_rank), key=operator.itemgetter(0))]

        return ranked_groups

    def _sequential_calculation_ranking_score(
        self,
        quantized_model: TModel,
        quantized_model_graph: NNCFGraph,
        groups_to_rank: List[GroupToRank],
        ranking_subset_indices: List[int],
        reference_values_for_each_item: Union[List[float], List[List[TTensor]]],
    ):
        ranking_scores = []  # ranking_scores[i] is the ranking score for groups_to_rank[i]
        for current_group in track(groups_to_rank, description="Calculating ranking scores"):
            modified_model = revert_operations_to_floating_point_precision(
                current_group.operations,
                current_group.quantizers,
                quantized_model,
                quantized_model_graph,
                self._restore_mode,
                self._algo_backend.get_op_with_weights_metatypes(),
                self._algo_backend.is_node_with_weight,
                self._algo_backend.get_weight_tensor_port_ids,
            )

            prepared_model = self._evaluator.prepare_model(modified_model)
            ranking_score = self._calculate_ranking_score(
                prepared_model, ranking_subset_indices, reference_values_for_each_item
            )
            ranking_scores.append(float(ranking_score))

        return ranking_scores

    def _multithreading_calculation_ranking_score(
        self,
        quantized_model: TModel,
        quantized_model_graph: NNCFGraph,
        groups_to_rank: List[GroupToRank],
        ranking_subset_indices: List[int],
        reference_values_for_each_item: Union[List[float], List[List[TTensor]]],
    ):
        ranking_scores = []  # ranking_scores[i] is the ranking score for groups_to_rank[i]
        prepared_model_queue = []
        executor = ThreadPoolExecutor(max_workers=self._num_workers)
        nncf_logger.info("Calculating ranking scores")
        for idx, current_group in enumerate(groups_to_rank):
            modified_model = revert_operations_to_floating_point_precision(
                current_group.operations,
                current_group.quantizers,
                quantized_model,
                quantized_model_graph,
                self._restore_mode,
                self._algo_backend.get_op_with_weights_metatypes(),
                self._algo_backend.is_node_with_weight,
                self._algo_backend.get_weight_tensor_port_ids,
            )

            prepared_model_queue.append(executor.submit(self._evaluator.prepare_model, modified_model))

            if idx >= (self._num_workers - 1):
                prepared_model = prepared_model_queue.pop(0).result()
                ranking_score = self._calculate_ranking_score(
                    prepared_model, ranking_subset_indices, reference_values_for_each_item
                )
                ranking_scores.append(float(ranking_score))

        for future in prepared_model_queue:
            prepared_model = future.result()
            ranking_score = self._calculate_ranking_score(
                prepared_model, ranking_subset_indices, reference_values_for_each_item
            )
            ranking_scores.append(float(ranking_score))

        return ranking_scores

    def _calculate_ranking_score(
        self,
        prepared_model: TPModel,
        ranking_subset_indices: List[int],
        reference_values_for_each_item: Union[List[float], List[List[TTensor]]],
    ) -> float:
        """
        Calculates the ranking score for the current group of quantizers.

        :param modified_model: Model from which the current group of quantizers was removed.
        :param ranking_subset_indices: Indices of the `ranking_data_items` in the whole dataset.
        :param reference_values_for_each_item: List of reference values.
        :return: The ranking score for the current group of quantizers.
        """
        if self._evaluator.is_metric_mode():
            # Calculate ranking score based on metric
            ranking_score, _ = self._evaluator.validate_prepared_model(
                prepared_model, self._dataset, ranking_subset_indices
            )
        else:
            # Calculate ranking score based on differences in logits
            approximate_outputs = self._evaluator.collect_values_for_each_item_using_prepared_model(
                prepared_model, self._dataset, ranking_subset_indices
            )
            reference_outputs = [reference_values_for_each_item[i] for i in ranking_subset_indices]
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
