"""
 Copyright (c) 2023 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import operator
from typing import List, Callable, Iterable, Any, Union, Optional, TypeVar
from dataclasses import dataclass

import numpy as np

from nncf.data.dataset import Dataset
from nncf.common.factory import ModelTransformerFactory
from nncf.common.graph import NNCFNode
from nncf.common.graph import NNCFGraph
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.quantization.quantizer_removal import find_quantizer_nodes_to_cut
from nncf.quantization.algorithms.accuracy_control.backend import AccuracyControlAlgoBackend
from nncf.quantization.algorithms.accuracy_control.utils import get_logits_for_each_item


TModel = TypeVar('TModel')


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

    # TODO(andrey-churkin): Add hash


def normalized_mse(x_ref: np.ndarray, x_approx: np.ndarray) -> float:
    """
    Calculates normalized mean square error between `x_ref` and `x_approx`.
    The normalized mean square error is defined as

    NMSE(x_ref, x_approx) = MSE(x_ref, x_approx) / MSE(x_ref, 0)

    :param x_ref: A 1-D array of (N,) shape. Represents the reference values.
    :param x_approx: A 1-D array of (N,) shape. Represents the measured values.
    :return: The normalized mean square error between `x_ref` and `x_approx`.
    """
    error = x_ref - x_approx
    nmse = np.dot(error, error) / np.dot(x_ref, x_ref)
    return nmse


def get_ranking_subset_indices(errors: List[float], subset_size: int) -> List[int]:
    """
    Returns `subset_size` indices of elements in the `errors` list
    that have the biggest error value. Returned indices are sorted in
    ascending order.

    :param errors: A list of errors.
    :param subset_size: A number of returned indices.
    :return: Indices of elements in the `errors` list which have the biggest error value.
    """
    ordered_indices = [
        idx for idx, _ in sorted(enumerate(errors), key=operator.itemgetter(1), reverse=True)
    ]
    end_index = min(subset_size, len(ordered_indices))
    return sorted(ordered_indices[:end_index])


def find_groups_of_quantizers_to_rank(
        nncf_graph: NNCFGraph,
        quantizer_metatypes: List[OperatorMetatype],
        const_metatypes: List[OperatorMetatype],
        quantizable_metatypes: List[OperatorMetatype],
        quantize_agnostic_metatypes: List[OperatorMetatype],
        shape_of_metatypes: List[OperatorMetatype]) -> List[GroupToRank]:
    """
    Finds groups of quantizers to rank.

    :param nncf_graph: NNCF graph.
    :param quantizer_metatypes: List of quantizer metatypes.
    :param const_metatypes: List of constant metatypes.
    :param quantizable_metatypes: List of metatypes for operations that may be quantized.
    :param quantize_agnostic_metatypes: List of quantize agnostic metatypes.
    :param shape_of_metatypes: List of shape of metatypes.
    :return: List of groups of quantizers to rank.
    """
    groups_to_rank = []
    processed = {}
    # TODO(andrey-churkin): Set order of quantizers here.
    for quantizer_node in nncf_graph.get_nodes_by_metatypes(quantizer_metatypes):
        if processed.get(quantizer_node.node_name, False):
            continue
        quantizers, operations = find_quantizer_nodes_to_cut(nncf_graph,
                                                             quantizer_node,
                                                             quantizer_metatypes,
                                                             const_metatypes,
                                                             quantizable_metatypes,
                                                             quantize_agnostic_metatypes,
                                                             shape_of_metatypes)
        for x in quantizers:
            processed[x.node_name] = True

        groups_to_rank.append(GroupToRank(quantizers, operations))

    return groups_to_rank


def revert_operations_to_floating_point_precision(operations: List[NNCFNode],
                                                  quantizers: List[NNCFNode],
                                                  quantized_model: TModel,
                                                  nncf_graph: NNCFGraph,
                                                  algo_backend: AccuracyControlAlgoBackend) -> TModel:
    """
    Reverts provided operations to floating-point precision by removing
    quantizers. Restores original bias for operations with bias.
    Restores original weights for operations with weights.

    :param operations: List of operations to revert in floating-point precision.
    :param quantizers: List of quantizers that should be removed to revert
        operations to floating-point precision.
    :param quantized_model: Quantized model in which provided operations
        should be reverted to floating-point precision.
    :param nncf_graph: The graph which was built for `quantized_model`.
    :param algo_backend: Backend for algorithm.
    :return: The model where `operations` were reverted to floating-point precision.
    """
    transformation_layout = TransformationLayout()

    for node in quantizers:
        transformation_layout.register(algo_backend.create_command_to_remove_quantizer(node))

    for node in operations:
        original_bias = node.data.get('original_bias', None)
        if original_bias is not None:
            transformation_layout.register(algo_backend.create_command_to_update_bias(node, original_bias, nncf_graph))

        original_weight = node.data.get('original_weight', None)
        if original_weight is not None:
            transformation_layout.register(algo_backend.create_command_to_update_weight(node, original_weight))

    model_transformer = ModelTransformerFactory.create(quantized_model)
    transformed_model = model_transformer.transform(transformation_layout)

    return transformed_model


def rank_quantizers(groups_to_rank: List[GroupToRank],
                    quantized_model,
                    dataset: Dataset,
                    validation_fn: Callable[[Any, Iterable[Any]], float],
                    x_ref: Union[List[float], List[np.ndarray]],
                    x_approx: Union[List[float], List[np.ndarray]],
                    use_metric: bool,
                    ranking_subset_size: int,
                    algo_backend: AccuracyControlAlgoBackend,
                    quantized_nncf_graph: NNCFGraph,
                    excluded_groups: Optional[GroupToRank] = None) -> List[GroupToRank]:
    """
    Ranks groups of quantizers by their contribution to accuracy drop. Returns list of
    ranked groups where the `ranked_groups[-1]` group of quantizers has maximal ranking score
    i.e. its contribution to accuracy drop is the greatest.

    :param groups_to_rank: Groups of quantizers that should be ranked.
    :param quantized_model: Quantized model.
    :param dataset: Dataset for the ranking process.
    :param validation_fn: A validation function to validate the model. It should take
        two argumets:
        - `model`: model to be validate.
        - `validation_dataset`: dataset that provides data items to
              validate the provided model.
        The function should return the value of the metric with the following meaning:
        A higher value corresponds to better performance of the model.
    :param x_ref: These are reference values collected from the initial model.
        If `use_metric` is `True` then `x_ref[i]` value is a metric value for i-th
        data item. If `use_metric` is `False` then `x_ref[i]` value is the
        logits for i-th data item.
    :param x_approx: These are approximate values collected from the quantized model.
        If `use_metric` is `True` then `x_approx[i]` value is a metric value for i-th
        data item. If `use_metric` is `False` then `x_approx[i]` value is the
        logits for i-th data item.
    :param use_metric: A boolean flag. If it is `True` then the original metric will be
        used to rank quantizers. If it is `False` then NMSE metric will be used.
    :param ranking_subset_size: The number of data items that will be selected from the
        dataset to rank groups of quantizers. The `len(dataset)` data items will be
        selected if `ranking_subset_size` parameter is greater than the number of
        elements in the dataset.
    :param algo_backend: The `AccuracyControlAlgoBackend` algo backend.
    :param quantized_nncf_graph: NNCF graph for quantized model.
    :param excluded_groups: Groups that should be excluded from ranking.
    :retunr: List of ranked groups of quantizers.
    """
    # Step 1: Create a subset of data items that will be used to rank groups of quantizers.
    error_fn = operator.sub if use_metric else normalized_mse
    errors = [error_fn(a, b) for a, b in zip(x_ref, x_approx)]
    ranking_subset_indices = get_ranking_subset_indices(errors, ranking_subset_size)
    # TODO(andrey-churkin): Should we read the dataset only once here?
    if use_metric:
        ranking_dataset = dataset.get_data(ranking_subset_indices)
    else:
        ranking_dataset = dataset.get_inference_data(ranking_subset_indices)

    # Step 2: Calculate ranking score for groups of quantizers.

    # `ranking_scores[i]` value is the ranking score for `groups_to_rank[i]`.
    ranking_scores = []

    for current_group in groups_to_rank:
        if excluded_groups and current_group in excluded_groups:
            continue

        modified_model = revert_operations_to_floating_point_precision(current_group, quantized_model,
                                                                       quantized_nncf_graph, algo_backend)

        # Get the ranking score for the current group of quantizers.
        if use_metric:
            ranking_score = validation_fn(algo_backend.prepare_for_inference(modified_model), ranking_dataset)
        else:
            output_name = [x.node_name for x in quantized_nncf_graph.get_output_nodes()][0]
            x_approx_subset_current = get_logits_for_each_item(modified_model, ranking_dataset, output_name)
            x_ref_subset = (x_ref[i] for i in ranking_subset_indices)
            errors_current = [
                normalized_mse(a, b) for a, b in zip(x_ref_subset, x_approx_subset_current)
            ]
            ranking_score = sum(errors_current) / len(errors_current)

        # TODO(andrey-churkin): Why does casting require here?
        ranking_scores.append(float(ranking_score))

    # Step 3: Rank groups.
    ranked_groups = [group for _, group in sorted(zip(ranking_scores, groups_to_rank), key=operator.itemgetter(0))]

    return ranked_groups
