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

import sys
import operator
from typing import Callable, Any, Iterable, Optional, List

from nncf.api.compression import TModel
from nncf.data.dataset import Dataset
from nncf.parameters import IgnoredScope
from nncf.parameters import ModelType
from nncf.parameters import TargetDevice
from nncf.common.utils.backend import get_backend
from nncf.common.utils.backend import BackendType
from nncf.common.graph.utils import get_number_of_quantized_ops
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph import NNCFNode
from nncf.common.graph import NNCFGraph
from nncf.common.factory import NNCFGraphFactory
from nncf.common.logging import nncf_logger
from nncf.common.quantization.structs import QuantizationPreset
from nncf.common.quantization.quantizer_removal import revert_operations_to_floating_point_precision
from nncf.quantization.quantize import quantize
from nncf.quantization.algorithms.accuracy_control.backend import AccuracyControlAlgoBackend
from nncf.quantization.algorithms.accuracy_control.ranker import Ranker
from nncf.quantization.algorithms.accuracy_control.ranker import MetricBasedRanker
from nncf.quantization.algorithms.accuracy_control.ranker import LogitsBasedRanker
from nncf.quantization.algorithms.accuracy_control.rank_functions import normalized_mse


def get_algo_backend(backend: BackendType) -> AccuracyControlAlgoBackend:
    """
    Returns backend for accuracy control algorithm.

    :param backend: Backend.
    :return: The backend for accuracy control algorithm.
    """
    if backend == BackendType.OPENVINO:
        from nncf.quantization.algorithms.accuracy_control.openvino_backend import OVAccuracyControlAlgoBackend
        return OVAccuracyControlAlgoBackend()

    raise RuntimeError('Cannot create the backend for the accuracy control algorithm '
                       f'because {backend} is not supported.')


def quantize_with_accuracy_control(model: TModel,
                                   calibration_dataset: Dataset,
                                   validation_dataset: Dataset,
                                   validation_fn: Callable[[Any, Iterable[Any]], float],
                                   max_drop: float = 0.01,
                                   preset: QuantizationPreset = QuantizationPreset.PERFORMANCE,
                                   target_device: TargetDevice = TargetDevice.ANY,
                                   subset_size: int = 300,
                                   fast_bias_correction: bool = True,
                                   model_type: Optional[ModelType] = None,
                                   ignored_scope: Optional[IgnoredScope] = None) -> TModel:
    """
    Common implementation of the `nncf.quantize_with_accuracy_control()` method.
    """
    # Step 0: Quantize provided model.
    quantized_model = quantize(model, calibration_dataset, preset, target_device, subset_size,
                               fast_bias_correction, model_type, ignored_scope)

    # Backends
    backend = get_backend(model)
    algo_backend = get_algo_backend(backend)

    initial_metric = validation_fn(algo_backend.prepare_for_inference(model),
                                   validation_dataset.get_data())
    nncf_logger.info(f'Metric of initial model: {initial_metric}')

    quantized_metric = validation_fn(algo_backend.prepare_for_inference(quantized_model),
                                     validation_dataset.get_data())
    nncf_logger.info(f'Metric of quantized model: {quantized_metric}')

    accuracy_aware_loop = AccuracyAwareLoop(is_native=False)
    return accuracy_aware_loop.restore_accuracy(model, initial_metric,
                                                quantized_model, quantized_metric,
                                                validation_dataset, validation_fn, max_drop)


def _create_message(nodes: Iterable[NNCFNode]) -> str:
    names = [f'\t{x.node_name}' for x in nodes]
    return '\n'.join(names)


class AccuracyAwareLoop:

    def __init__(self,
                 ranking_subset_size: int = 300,
                 max_num_iterations: int = sys.maxsize,
                 is_native: bool = True):
        """
        :param ranking_subset_size:
        :param max_num_iterations:
        :param is_native:
        """
        self.ranking_subset_size = ranking_subset_size
        self.max_num_iterations = max_num_iterations
        # TODO(andrey-churkin): Should be removed when native implementation
        # will become the main one.
        self.is_native = is_native

    def restore_accuracy(self,
                         initial_model: TModel,
                         initial_metric: float,
                         quantized_model: TModel,
                         quantized_metric: float,
                         validation_dataset: Dataset,
                         validation_fn: Callable[[Any, Iterable[Any]], float],
                         max_drop: float = 0.01) -> TModel:
        """
        Restores the accuracy of the quantized model by removing the groups of quantizers
        that contribute the most to the drop in accuracy.

        :param initial_model: Initial model (not quantized).
        :param initial_metric: Metric value for initial model.
        :param quantized_model: Quantized model.
        :param quantized_metric: Metric value for quantized model.
        :param validation_dataset: A dataset for the validation process.
        :param validation_fn: A validation function to validate the model. It should take
            two argumets:
            - `model`: model to be validate.
            - `validation_dataset`: dataset that provides data items to
                validate the provided model.
            The function should return the value of the metric with the following meaning:
            A higher value corresponds to better performance of the model.
        :param max_drop: The maximum absolute accuracy drop that should be achieved.
        :return: The quantized model whose metric `final_metric` is satisfied the following condition

            initial_metric - final_metric <= max_drop.
        """
        accuracy_drop = initial_metric - quantized_metric
        nncf_logger.info(f'Accuracy drop: {accuracy_drop}')

        if accuracy_drop <= max_drop:
            return quantized_model

        nncf_logger.info('Changing the scope of quantizer nodes.')

        # Hyperparameters of algorithm
        USE_PREVIOUS_IF_DROP_INCREASE = True
        precision_change_to = 'floating-point'
        exclude_bad_nodes = False

        # Backends
        backend = get_backend(initial_model)
        algo_backend = get_algo_backend(backend)

        initial_model_graph = NNCFGraphFactory.create(initial_model)
        quantized_model_graph = NNCFGraphFactory.create(quantized_model)

        if not self.is_native:
            AccuracyAwareLoop._match_const_nodes_names(initial_model_graph,
                                                       quantized_model_graph,
                                                       algo_backend.get_const_metatypes())

        # Collect original biases and weights because these values are
        # required to undo bias correction and weight correction.
        # Store this data inside the `node.data` dictionary.
        # This data will be used in the `revert_operations_to_floating_point_precision()` method.
        AccuracyAwareLoop._collect_original_biases_and_weights(initial_model_graph, quantized_model_graph,
                                                               initial_model, algo_backend)

        # Show the number of quantized operations in the model.
        num_of_quantized_ops = get_number_of_quantized_ops(quantized_model_graph,
                                                           algo_backend.get_quantizer_metatypes(),
                                                           algo_backend.get_quantizable_metatypes())
        nncf_logger.info(f'Total number of quantized operations in the model: {num_of_quantized_ops}')

        nncf_logger.info('== Ranking groups of quantizers were started ==')
        ranker = AccuracyAwareLoop._create_ranker(initial_model, validation_fn, validation_dataset,
                                                  self.ranking_subset_size, algo_backend)
        groups_to_rank = ranker.find_groups_of_quantizers_to_rank(quantized_model_graph)
        ranked_groups = ranker.rank_groups_of_quantizers(groups_to_rank, initial_model, quantized_model,
                                                         quantized_model_graph)

        current_num_quantizers = len(
            quantized_model_graph.get_nodes_by_metatypes(algo_backend.get_quantizer_metatypes())
        )

        previous_model = quantized_model
        previous_accuracy_drop = accuracy_drop
        current_model = None
        current_accuracy_drop = None

        reached_required_drop = False
        is_step_back = True
        removed_all = False
        excluded_nodes = []
        all_removed_nodes = []
        all_reverted_ops = set()

        for iteration in range(self.max_num_iterations):
            if current_model is not None:
                previous_model = current_model

            if not ranked_groups:
                nncf_logger.info(
                        'All layers have been checked and the AccuracyAwareQuantization '
                        'will not be able to achieve the required accuracy drop')
                removed_all = True
                break

            # greedy removal of the FQ node with the highest importance score
            current_group = ranked_groups.pop()
            current_model = revert_operations_to_floating_point_precision(
                current_group.operations,
                current_group.quantizers,
                previous_model,
                quantized_model_graph,
                algo_backend.create_command_to_remove_quantizer,
                algo_backend.create_command_to_update_bias,
                algo_backend.create_command_to_update_weight)

            nncf_logger.debug(f'Removed a block of {len(current_group.quantizers)} quantizers:'
                              f'\n{_create_message(current_group.quantizers)}')
            nncf_logger.info(f'Reverted {len(current_group.operations)} operations to the {precision_change_to} '
                             f'precision: \n{_create_message(current_group.operations)}')

            current_num_quantizers = current_num_quantizers - len(current_group.quantizers)
            all_removed_nodes.extend(current_group.quantizers)
            all_reverted_ops.update(current_group.operations)

            # Calculate drop for new quantization scope.
            current_metric = validation_fn(algo_backend.prepare_for_inference(current_model),
                                           validation_dataset.get_data())
            current_accuracy_drop = initial_metric - current_metric
            nncf_logger.info('Accuracy drop with the new quantization scope is %s', float(current_accuracy_drop))

            if current_num_quantizers == 0:
                nncf_logger.info('All quantizers were removed from the model.')
                removed_all = True
                break

            # Accuracy was restored to the acceptable drop.
            if current_accuracy_drop <= max_drop:
                reached_required_drop = True
                break

            # Continue greedy quantizer remove
            if max_drop < current_accuracy_drop <= previous_accuracy_drop \
                    or (current_accuracy_drop > previous_accuracy_drop and is_step_back):
                is_step_back = False
                previous_accuracy_drop = current_accuracy_drop
                continue

            if current_accuracy_drop > previous_accuracy_drop and USE_PREVIOUS_IF_DROP_INCREASE:
                current_model = previous_model
                all_removed_nodes = all_removed_nodes[:len(all_removed_nodes)-len(current_group.quantizers)]
                all_reverted_ops.difference_update(current_group.operations)
                if exclude_bad_nodes:
                    excluded_nodes.extend(current_group.quantizers)
                    nncf_logger.debug('Quantizers were added to the excluded list: '
                                      f'{_create_message(current_group.quantizers)}')
                is_step_back = True

            previous_accuracy_drop = current_accuracy_drop

            nncf_logger.info('Re-calculating ranking scores for remaining groups')
            ranker.rank_groups_of_quantizers(ranked_groups, initial_model, current_model, quantized_model_graph)

        # Show results that were achieved.
        if removed_all or not reached_required_drop:
            nncf_logger.info('The algorithm could not achieve the required accuracy drop.', force=True)

        if iteration + 1 >= self.max_num_iterations:
            nncf_logger.info('Maximum number of iteration was reached.')

        if not removed_all:
            nncf_logger.debug(f'Quantizers that were removed:\n{_create_message(all_removed_nodes)}')
            nncf_logger.info(f'{len(all_reverted_ops)} out of {num_of_quantized_ops} '
                             f'were reverted back to the {precision_change_to} precision:'
                             f'\n{_create_message(all_reverted_ops)}')

        return current_model

    # TODO(andrey-churkin): Should be removed when native implementation will become the main one.
    @staticmethod
    def _match_const_nodes_names(initial_model_graph: NNCFGraph,
                                 quantized_model_graph: NNCFGraph,
                                 const_metatypes: List[OperatorMetatype]) -> None:
        initial_graph_const_nodes = initial_model_graph.get_nodes_by_metatypes(const_metatypes)
        quantized_graph_const_nodes = quantized_model_graph.get_nodes_by_metatypes(const_metatypes)
        for initial_graph_const_node in initial_graph_const_nodes:
            num_matches = 0
            for quantized_graph_const_node in quantized_graph_const_nodes:
                if quantized_graph_const_node.node_name.startswith(initial_graph_const_node.node_name):
                    assert quantized_graph_const_node.node_type == initial_graph_const_node.node_type
                    quantized_graph_const_node.data[NNCFGraph.NODE_NAME_ATTR] = initial_graph_const_node.node_name
                    num_matches += 1
            assert num_matches == 1

    @staticmethod
    def _collect_original_biases_and_weights(initial_model_graph: NNCFGraph,
                                             quantized_model_graph: NNCFGraph,
                                             initial_model,
                                             algo_backend: AccuracyControlAlgoBackend) -> None:
        nodes_with_bias = (node for node in quantized_model_graph.get_all_nodes() \
                           if algo_backend.is_node_with_bias(node, quantized_model_graph))
        for biased_node in nodes_with_bias:
            biased_node.data['original_bias'] = algo_backend.get_bias_value(biased_node,
                                                                            quantized_model_graph,
                                                                            initial_model)

        nodes_with_weight = (node for node in initial_model_graph.get_all_nodes() \
                             if algo_backend.is_node_with_weight(node))
        for node_with_weight in nodes_with_weight:
            node = quantized_model_graph.get_node_by_name(node_with_weight.node_name)
            node.data['original_weight'] = algo_backend.get_weight_value(node_with_weight,
                                                                         initial_model_graph,
                                                                         initial_model)

    @staticmethod
    def _create_ranker(initial_model,
                       validation_fn: Callable[[Any, Iterable[Any]], float],
                       validation_dataset: Dataset,
                       ranking_subset_size: int,
                       algo_backend: AccuracyControlAlgoBackend) -> Ranker:
        # Check whether it is possible to calculate the metric for one data item.
        # pylint: disable=W0703
        try:
            _ = validation_fn(algo_backend.prepare_for_inference(initial_model),
                              validation_dataset.get_data([0]))
            ranker = MetricBasedRanker(ranking_subset_size, operator.sub,
                                       validation_dataset, algo_backend, validation_fn)
        except Exception:
            ranker = LogitsBasedRanker(ranking_subset_size, normalized_mse,
                                       validation_dataset, algo_backend)
        nncf_logger.info(f'The {"original" if isinstance(ranker, MetricBasedRanker) else "NMSE"} '
                         'metric will be used to rank quantizers')
        return ranker
