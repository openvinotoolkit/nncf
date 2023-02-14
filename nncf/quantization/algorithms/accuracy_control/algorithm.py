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
from typing import Callable
from typing import Any
from typing import Iterable
from typing import Optional

from nncf.api.compression import TModel
from nncf.data.dataset import Dataset
from nncf.parameters import IgnoredScope
from nncf.parameters import ModelType
from nncf.parameters import TargetDevice
from nncf.common.utils.backend import get_backend
from nncf.common.utils.backend import BackendType
from nncf.common.graph.utils import get_number_of_quantized_ops
from nncf.common.graph import NNCFNode
from nncf.common.factory import NNCFGraphFactory
from nncf.common.logging import nncf_logger
from nncf.common.quantization.structs import QuantizationPreset
from nncf.quantization.quantize import quantize
from nncf.quantization.algorithms.accuracy_control.backend import AccuracyControlAlgoBackend
from nncf.quantization.algorithms.accuracy_control.utils import get_metric_for_each_item
from nncf.quantization.algorithms.accuracy_control.utils import get_logits_for_each_item
from nncf.common.quantization.quantizer_removal import remove_quantizer_from_model
from nncf.quantization.algorithms.accuracy_control.ranking import rank_quantizers


def get_algo_backend(backend: BackendType) -> AccuracyControlAlgoBackend:
    """
    Returns backend for accuracy control algorithm.

    :param backend: Backend.
    :return: The backend for accuracy control algorithm.
    """
    if backend == BackendType.OPENVINO:
        from nncf.quantization.algorithms.accuracy_control.openvino_backend import OVAccuracyControlAlgoBackend
        return OVAccuracyControlAlgoBackend()
    if backend == BackendType.ONNX:
        from nncf.quantization.algorithms.accuracy_control.onnx_backend import ONNXAccuracyControlAlgoBackend
        return ONNXAccuracyControlAlgoBackend()

    raise RuntimeError('Cannot create the backend for the accuracy control algorithm '
                       f'because {backend} is not supported.')


def native_quantize_with_accuracy_control(model: TModel,
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
    # Backends
    backend = get_backend(model)
    algo_backend = get_algo_backend(backend)

    quantized_model = quantize(model, calibration_dataset, preset, target_device, subset_size,
                               fast_bias_correction, model_type, ignored_scope)

    baseline_metric = validation_fn(algo_backend.prepare_for_inference(model),
                                    validation_dataset)
    nncf_logger.info(f'Baseline metric: {baseline_metric}')


    quantized_metric = validation_fn(algo_backend.prepare_for_inference(quantized_model))
    nncf_logger.info(f'Quantized metric: {quantized_metric}')

    accuracy_drop = baseline_metric - quantized_metric
    nncf_logger.info(f'Accuracy drop: {accuracy_drop}')

    if accuracy_drop <= max_drop:
        return quantized_model

    nncf_logger.info('Changing the scope of quantizer nodes.')
    return restore_accuracy(model, quantized_model, validation_dataset,
                            validation_fn, baseline_metric, accuracy_drop, max_drop)


def _create_message(nodes: Iterable[NNCFNode]) -> str:
    names = [f'\t{x.node_name}' for x in nodes]
    return '\n'.join(names)


def restore_accuracy(model: TModel,
                     quantized_model: TModel,
                     validation_dataset: Dataset,
                     validation_fn: Callable[[Any, Iterable[Any]], float],
                     baseline_metric: float,
                     accuracy_drop: float,
                     max_drop: float = 0.01) -> TModel:

    # Hyperparameters of algorithm
    RANKING_SUBSET_SIZE = 300
    MAX_NUM_ITERATIONS = sys.maxsize
    USE_PREVIOUS_IF_DROP_INCREASE = True

    precision_change_to = 'floating-point'
    exclude_bad_nodes = False

    # Backends
    backend = get_backend(model)
    algo_backend = get_algo_backend(backend)

    nncf_graph = NNCFGraphFactory.create(quantized_model)

    # DEBUG
    # if True:
    #     from nncf.experimental.netron import save_for_netron
    #     from nncf.common.graph import NNCFGraph
    #     graph = NNCFGraphFactory.create(model)
    #     save_for_netron(nncf_graph, 'resnet18_int8_nncf_graph.xml')
    #     save_for_netron(graph, 'resnet18_nncf_graph.xml')
    #     xs = graph.get_nodes_by_metatypes(algo_backend.get_const_metatypes())
    #     ys = nncf_graph.get_nodes_by_metatypes(algo_backend.get_const_metatypes())
    #     m = {}
    #     for x in xs:
    #         cnt = 0
    #         for y in ys:
    #             if y.node_name.startswith(x.node_name):
    #                 assert y.node_type == x.node_type
    #                 m[y.node_name] = x.node_name
    #                 y.data[NNCFGraph.NODE_NAME_ATTR] = x.node_name
    #                 cnt += 1
    #         assert cnt == 1
    # DEBUG

    # We need to collect original bias values for biased nodes.
    # It will be stored inside the `bised_node.data` dictionary.
    # The original bias will be used to undo bias correction for
    # the node if it is reverted to the original precision during
    # the quantizer removing process.
    nodes_with_bias = [node for node in nncf_graph.get_all_nodes() \
                       if algo_backend.is_node_with_bias(node, nncf_graph)]
    for biased_node in nodes_with_bias:
        biased_node.data['original_bias'] = algo_backend.get_bias_value(biased_node, nncf_graph, model)

    num_of_quantized_ops = get_number_of_quantized_ops(nncf_graph,
                                                       algo_backend.get_quantizer_metatypes(),
                                                       algo_backend.get_quantizable_metatypes())
    nncf_logger.info(f'The total number of quantized operations in the model: {num_of_quantized_ops}')

    # Check whether it is possible to calculate the metric for one data item.
    # pylint: disable=W0703
    USE_METRIC = True
    try:
        _ = validation_fn(algo_backend.prepare_for_inference(model),
                          validation_dataset.get_data([0]))
    except Exception:
        USE_METRIC = False
    nncf_logger.info(f'The {"original" if USE_METRIC else "NMSE"} metric will be used to rank quantizers')

    # TODO(andrey-churkin): We need read dataset only once here to optimize execution time.
    if USE_METRIC:
        x_ref = get_metric_for_each_item(algo_backend.prepare_for_inference(model),
                                         validation_dataset,
                                         validation_fn)

        x_approx = get_metric_for_each_item(algo_backend.prepare_for_inference(quantized_model),
                                            validation_dataset,
                                            validation_fn)
    else:
        output_name = [x.node_name for x in nncf_graph.get_output_nodes()][0]
        x_ref = get_logits_for_each_item(model, validation_dataset, output_name)
        x_approx = get_logits_for_each_item(quantized_model, validation_dataset, output_name)

    ranked_quantizers = rank_quantizers(quantized_model, validation_dataset, validation_fn, x_ref, x_approx,
                                        USE_METRIC, RANKING_SUBSET_SIZE, algo_backend)

    current_num_quantizers = len(
        nncf_graph.get_nodes_by_metatypes(algo_backend.get_quantizer_metatypes())
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

    for iteration in range(MAX_NUM_ITERATIONS):
        if current_model is not None:
            previous_model = current_model

        if not ranked_quantizers:
            nncf_logger.info(
                    'All layers have been checked and the AccuracyAwareQuantization '
                    'will not be able to achieve the required accuracy drop')
            removed_all = True
            break

        # greedy removal of the FQ node with the highest importance score
        quantizer_to_remove = ranked_quantizers.pop()
        current_model, removed_quantizers, reverted_ops = remove_quantizer_from_model(
            previous_model,
            quantizer_to_remove,
            nncf_graph,
            algo_backend.get_quantizer_metatypes(),
            algo_backend.get_const_metatypes(),
            algo_backend.get_quantizable_metatypes(),
            algo_backend.get_quantize_agnostic_metatypes(),
            algo_backend.create_command_to_remove_quantizer,
            algo_backend.create_command_to_update_bias
        )
        current_num_quantizers = current_num_quantizers - len(removed_quantizers)

        # TODO(andrey-churkin): Move to debug level.
        nncf_logger.info(f'Removed a block of {len(removed_quantizers)} quantizers:'
                         f'\n{_create_message(removed_quantizers)}')
        nncf_logger.info(f'Reverted {len(reverted_ops)} operations to the {precision_change_to} precision:'
                         f'\n{_create_message(reverted_ops)}')

        all_removed_nodes.extend(removed_quantizers)
        all_reverted_ops.update(reverted_ops)

        # Calculate drop for new quantization scope
        current_metric = validation_fn(current_model, validation_dataset.get_data())
        current_accuracy_drop = baseline_metric - current_metric
        nncf_logger.info('Accuracy drop with the new quantization scope is %s', current_accuracy_drop)

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
            all_removed_nodes = all_removed_nodes[:len(all_removed_nodes)-len(removed_quantizers)]
            all_reverted_ops.difference_update(reverted_ops)
            if exclude_bad_nodes:
                excluded_nodes.extend(removed_quantizers)
                nncf_logger.debug(f'Quantizers were added to the excluded list: {_create_message(removed_quantizers)}')
            is_step_back = True

        previous_accuracy_drop = current_accuracy_drop

        nncf_logger.info('Re-calculating node importance')
        if USE_METRIC:
            current_x_approx = get_metric_for_each_item(algo_backend.prepare_for_inference(current_model),
                                                        validation_dataset,
                                                        validation_fn)
        else:
            output_name = [x.node_name for x in nncf_graph.get_output_nodes()][0]
            current_x_approx = get_logits_for_each_item(current_model, validation_dataset, output_name)

        ranked_quantizers = rank_quantizers(current_model, validation_dataset, validation_fn, x_ref, current_x_approx,
                                            USE_METRIC, RANKING_SUBSET_SIZE, algo_backend)

    # Show results that were achieved.
    if removed_all or not reached_required_drop:
        nncf_logger.info('The algorithm could not achieve the required accuracy drop.', force=True)

    if iteration + 1 >= MAX_NUM_ITERATIONS:
        nncf_logger.info('Maximum number of iteration was reached.')

    if not removed_all:
        nncf_logger.debug(f'Quantizers that were removed:\n{_create_message(all_removed_nodes)}')
        nncf_logger.info(f'{len(all_reverted_ops)} out of {num_of_quantized_ops} '
                         f'were reverted back to the {precision_change_to} precision:'
                         f'\n{_create_message(all_reverted_ops)}')

    return current_model
