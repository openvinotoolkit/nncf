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
from typing import Tuple

import numpy as np

from nncf.data.dataset import Dataset
from nncf.common.graph import NNCFNode
from nncf.common.factory import NNCFGraphFactory
from nncf.common.logging import nncf_logger
from nncf.quantization.algorithms.accuracy_control.backend import AccuracyControlAlgoBackend
from nncf.quantization.algorithms.accuracy_control.utils import remove_quantizer_from_model
from nncf.quantization.algorithms.accuracy_control.utils import get_logits_for_each_item


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


# TODO(andrey-churkin): We need to introduce common metatypes to
# remove `algo_backend` from the signature of this method.

def rank_quantizers(quantized_model,
                    dataset: Dataset,
                    validation_fn: Callable[[Any, Iterable[Any]], float],
                    x_ref: Union[List[float], List[np.ndarray]],
                    x_approx: Union[List[float], List[np.ndarray]],
                    use_metric: bool,
                    ranking_subset_size: int,
                    algo_backend: AccuracyControlAlgoBackend,
                    excluded_nodes: Optional[List[NNCFNode]] = None) -> List[NNCFNode]:
     """
     Ranks quantizers by their contribution to accuracy drop. Returns list of
     ranked quantizers where ranked_quantizers[-1] quantizer has maximal
     rank i.e. its contribution is the greatest.
     """
     # Create a subset of data items that will be used to rank quantizers.
     error_fn = operator.sub if use_metric else normalized_mse
     errors = [error_fn(a, b) for a, b in zip(x_ref, x_approx)]
     ranking_subset_indices = get_ranking_subset_indices(errors, ranking_subset_size)
     # TODO(andrey-churkin): Should we read the dataset only once here?
     if use_metric:
          ranking_dataset = dataset.get_data(ranking_subset_indices)
     else:
          ranking_dataset = dataset.get_inference_data(ranking_subset_indices)

     # Calculate ranking score for quantizers.
     quantizer_and_ranking_score: List[Tuple[NNCFNode, float]] = []
     processed_quantizers = []
     graph = NNCFGraphFactory.create(quantized_model)

     # TODO(andrey-churkin): Check the order of quantizer nodes.
     quantizer_nodes = (
          node for node in graph.topological_sort() if node.metatype in algo_backend.get_quantizer_metatypes()
     )
     for quantizer_node in quantizer_nodes:
          # TODO(andrey-churkin): Add a description for why this `if` statement is needed.
          if excluded_nodes and quantizer_node in excluded_nodes:
               continue

          # TODO(andrey-churkin): Add a description for why this `if` statement is needed.
          if quantizer_node in processed_quantizers:
               continue

          modified_model, removed_quantizers, _ = remove_quantizer_from_model(quantized_model,
                                                                              quantizer_node,
                                                                              graph,
                                                                              algo_backend.get_quantizer_metatypes(),
                                                                              algo_backend.get_const_metatypes(),
                                                                              algo_backend.get_quantizable_metatypes(),
                                                                              algo_backend.get_quantize_agnostic_metatypes(),
                                                                              algo_backend.create_command_to_remove_quantizer,
                                                                              algo_backend.create_command_to_update_bias)
          if not removed_quantizers:
               continue

          removed_names = [x.node_name for x in removed_quantizers]
          nncf_logger.info(f'Removed a block of {len(removed_names)} quantizers: {", ".join(removed_names)}')

          processed_quantizers.extend(removed_quantizers)

          # Get the ranking score for the `removed_quantizers` scope.
          if use_metric:
               ranking_score = validation_fn(algo_backend.prepare_for_inference(modified_model),
                                             ranking_dataset)
          else:
               output_name = [x.node_name for x in graph.get_output_nodes()][0]
               x_approx_subset_current = get_logits_for_each_item(modified_model, ranking_dataset, output_name)
               x_ref_subset = (x_ref[i] for i in ranking_subset_indices)
               errors_current = [
                    normalized_mse(a, b) for a, b in zip(x_ref_subset, x_approx_subset_current)
               ]
               ranking_score = sum(errors_current) / len(errors_current)

          quantizer_and_ranking_score.append((quantizer_node, ranking_score))

     # Проверить случай с метрикой, как мы должны сортировать?
     ranked_quantizers = [
          quantizer for quantizer, _ in sorted(quantizer_and_ranking_score, key=operator.itemgetter(1))
     ]

     return ranked_quantizers
