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

from typing import Callable, List, Tuple, TypeVar

from nncf.common.factory import CommandCreatorFactory
from nncf.common.factory import ModelTransformerFactory
from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.quantization.advanced_parameters import RestoreMode

TModel = TypeVar("TModel")


def find_quantizer_nodes_to_cut(
    graph: NNCFGraph,
    quantizer_node: NNCFNode,
    quantizer_metatypes: List[OperatorMetatype],
    const_metatypes: List[OperatorMetatype],
    quantizable_metatypes: List[OperatorMetatype],
    quantize_agnostic_metatypes: List[OperatorMetatype],
) -> Tuple[List[NNCFNode], List[NNCFNode]]:
    """
    Finds quantizer nodes that should be removed in addition to `quantizer_node` to get
    the correct model for inference. Returns the list of quantizer nodes (`quantizer_node` + nodes
    which were found) and the list of nodes that will be reverted to original precision if
    quantizer nodes are removed.

    :param graph: The NNCFGraph without shapeof subgraphs.
    :param quantizer_node: The quantizer node that we want to remove.
    :param quantizer_metatypes: List of quantizer metatypes.
    :param const_metatypes: List of constant metatypes.
    :param quantizable_metatypes: List of metatypes for operations
        that may be quantized.
    :param quantize_agnostic_metatypes: List of quantize agnostic metatypes.
    :return: A tuple (quantizer_nodes, ops) where
        - `quantizer_nodes` is the list of quantizer nodes
        - `ops` is the list of nodes that will be reverted to original precision
        if `quantizer_nodes` are removed.
    """

    def _parse_node_relatives(node: NNCFNode, is_parents: bool):
        relatives = graph.get_previous_nodes(node) if is_parents else graph.get_next_nodes(node)
        for relative in relatives:
            if relative.metatype in quantizable_metatypes:
                if is_parents:
                    if relative in seen_children:
                        continue
                    to_see_children.append(relative)
                else:
                    ops_to_return_in_orig_prec.add(relative)
                    if relative not in seen_parents:
                        to_see_parents.append(relative)
            elif relative.metatype in quantizer_metatypes:
                if is_parents:
                    if relative in seen_children:
                        continue
                    if relative not in to_cut:
                        to_cut.append(relative)
                    to_see_children.append(relative)
                    # We should see parents for the `relative` node here only if they are
                    # all quantizers. This covers the quantize-dequantize case, where we
                    # should see parents for the dequantize node.
                    if all(x.metatype in quantizer_metatypes for x in graph.get_previous_nodes(relative)):
                        to_see_parents.append(relative)
                elif node.metatype in quantizer_metatypes:
                    # `node` is a quantizer (quantize-dequantize case) here, and `relative`
                    # is the dequantizer. So, we should cut `relative` and look at its children.
                    if relative not in to_cut:
                        to_cut.append(relative)
                    to_see_children.append(relative)
                else:
                    seen_children.append(relative)
            elif relative.metatype not in const_metatypes:
                if relative not in seen_parents:
                    to_see_parents.append(relative)
                if relative not in seen_children and relative.metatype in quantize_agnostic_metatypes:
                    to_see_children.append(relative)

        seen_list = seen_parents if is_parents else seen_children
        seen_list.append(node)

    seen_children, seen_parents = [], []
    to_see_children, to_see_parents = [quantizer_node], []
    to_cut = [quantizer_node]
    ops_to_return_in_orig_prec = set()

    while to_see_parents or to_see_children:
        if to_see_children:
            _parse_node_relatives(to_see_children.pop(), is_parents=False)
        if to_see_parents:
            _parse_node_relatives(to_see_parents.pop(), is_parents=True)

    return to_cut, list(ops_to_return_in_orig_prec)


def revert_operations_to_floating_point_precision(
    operations: List[NNCFNode],
    quantizers: List[NNCFNode],
    quantized_model: TModel,
    quantized_model_graph: NNCFGraph,
    restore_mode: RestoreMode,
    op_with_weights_metatypes: List[OperatorMetatype],
    is_node_with_weight_fn: Callable[[NNCFNode], bool],
    get_weight_tensor_port_ids_fn: Callable[[NNCFNode], List[int]],
) -> TModel:
    """
    Reverts provided operations to floating-point precision by removing
    quantizers. Restores original bias for operations with bias.
    Restores original weights for operations with weights.

    :param operations: List of operations to revert in floating-point precision.
    :param quantizers: List of quantizers that should be removed to revert
        operations to floating-point precision.
    :param quantized_model: Quantized model in which provided operations
        should be reverted to floating-point precision.
    :param quantized_model_graph: The graph which was built for `quantized_model`.
    :param restore_mode: Restore mode.
    :param op_with_weights_metatypes: List of operation metatypes that can be reverted to representation
        with int8 weights.
    :param is_node_with_weight_fn: Checks if the node has a weight or not. Returns `True` if `node` corresponds
        to the operation with weights, `False` otherwise.
    :param get_weight_tensor_port_ids_fn: Returns node's input port indices with weights tensors.
    :return: The model where `operations` were reverted to floating-point precision.
    """
    transformation_layout = TransformationLayout()

    command_creator = CommandCreatorFactory.create(quantized_model)

    should_remove_fq = {}
    should_revert_weights_to_fp32 = {}
    if restore_mode == RestoreMode.ONLY_ACTIVATIONS:
        for node in operations:
            if (
                node.metatype in op_with_weights_metatypes
                and node.layer_attributes
                and node.layer_attributes.constant_attributes
            ):
                input_edges = quantized_model_graph.get_input_edges(node)
                for port_id in node.layer_attributes.get_const_port_ids():
                    fq_weight = input_edges[port_id].from_node
                    should_remove_fq[fq_weight.node_name] = False
                    should_revert_weights_to_fp32[node.node_name] = False

    for node in quantizers:
        if should_remove_fq.get(node.node_name, True):
            transformation_layout.register(command_creator.create_command_to_remove_quantizer(node))

    for node in operations:
        original_bias = node.attributes.get("original_bias", None)
        if original_bias is not None:
            transformation_layout.register(
                command_creator.create_command_to_update_bias(node, original_bias, quantized_model_graph)
            )

        if not should_revert_weights_to_fp32.get(node.node_name, True):
            continue

        if is_node_with_weight_fn(node):
            weight_port_ids = get_weight_tensor_port_ids_fn(node)
            for port_id in weight_port_ids:
                original_weight = node.attributes.get(f"original_weight.{port_id}", None)
                if original_weight is not None:
                    transformation_layout.register(
                        command_creator.create_command_to_update_weight(node, original_weight, port_id)
                    )

    model_transformer = ModelTransformerFactory.create(quantized_model)
    transformed_model = model_transformer.transform(transformation_layout)

    return transformed_model
