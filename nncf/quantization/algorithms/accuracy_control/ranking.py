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
from typing import List
from typing import Callable
from typing import Iterable
from typing import Any
from typing import Union
from typing import Optional
from typing import TypeVar
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


# TODO(andrey-churkin): To remove the `algo_backend` parameter we need to introduce
# the `CommandCreator` and `CommandCreatorFactory` classes.
def remove_group_of_quantizers_from_model(group_to_remove: GroupToRank,
                                          quantized_model: TModel,
                                          nncf_graph: NNCFGraph,
                                          algo_backend: AccuracyControlAlgoBackend) -> TModel:
    """
    Removes group of quantizers from the model.

    :param group_to_remove: Group of quantizers to remove.
    :param quantized_model: Quantized model from which quantizers should be removed.
    :param nncf_graph: The graph which was built for `quantized_model`.
    :param algo_backend: Backend for algorithm.
    :return: The model from which `group_to_remove.quantizers` were removed.
    """
    transformation_layout = TransformationLayout()

    for node in group_to_remove.quantizers:
        transformation_layout.register(algo_backend.create_command_to_remove_quantizer(node))

    for node in group_to_remove.operations:
        original_bias = node.data.get('original_bias', None)
        if original_bias is not None:
            transformation_layout.register(algo_backend.create_command_to_update_bias(node, original_bias, nncf_graph))

        original_weight = node.data.get('original_weight', None)
        if original_weight is not None:
            transformation_layout.register(algo_backend.create_command_to_update_weight(node, original_weight))

    model_transformer = ModelTransformerFactory.create(quantized_model)
    transformed_model = model_transformer.transform(transformation_layout)

    return transformed_model


# TODO(andrey-churkin): We need to introduce common metatypes to
# remove `algo_backend` from the signature of this method.
def rank_quantizers(groups_to_rank: List[GroupToRank],
                    quantized_model,
                    dataset: Dataset,
                    validation_fn: Callable[[Any, Iterable[Any]], float],
                    x_ref: Union[List[float], List[np.ndarray]],
                    x_approx: Union[List[float], List[np.ndarray]],
                    use_metric: bool,
                    ranking_subset_size: int,
                    algo_backend: AccuracyControlAlgoBackend,
                    nncf_graph: NNCFGraph,
                    excluded_groups: Optional[GroupToRank] = None) -> List[GroupToRank]:
    """
    Ranks groups of quantizers by their contribution to accuracy drop. Returns list of
    ranked groups where the `ranked_groups[-1]` group of quantizers has maximal ranking score
    i.e. its contribution to accuracy drop is the greatest.

    :param groups_to_rank:
    :param quantized_model:
    :param dataset:
    :param validation_fn:
    :param x_ref:
    :param x_approx:
    :param use_metric:
    :param ranking_subset_size:
    :param algo_backend:
    :param nncf_graph:
    :param excluded_groups:
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
        # TODO(andrey-churkin): Add a description for why this `if` statement is needed.
        if excluded_groups and current_group in excluded_groups:
            continue

        modified_model = remove_group_of_quantizers_from_model(current_group, quantized_model, nncf_graph, algo_backend)

        # Get the ranking score for the current group of quantizers.
        if use_metric:
            ranking_score = validation_fn(algo_backend.prepare_for_inference(modified_model), ranking_dataset)
        else:
            output_name = [x.node_name for x in nncf_graph.get_output_nodes()][0]
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
