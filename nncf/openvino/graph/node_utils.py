# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import openvino.runtime as ov
import openvino.runtime.opset13 as opset

import nncf
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.layer_attributes import ConvolutionLayerAttributes
from nncf.common.graph.layer_attributes import GenericWeightedLayerAttributes
from nncf.common.graph.layer_attributes import LinearLayerAttributes
from nncf.common.graph.layer_attributes import WeightedLayerAttributes
from nncf.common.tensor_statistics.collectors import ReductionAxes
from nncf.openvino.graph.layout import OVLayoutElem
from nncf.openvino.graph.layout import get_conv_weights_layout
from nncf.openvino.graph.layout import get_conv_weights_layout_from_node
from nncf.openvino.graph.layout import get_linear_activations_layout_from_node
from nncf.openvino.graph.layout import get_linear_input_layout
from nncf.openvino.graph.layout import get_linear_weights_layout_from_node
from nncf.openvino.graph.metatypes.groups import CONV_OPERATIONS
from nncf.openvino.graph.metatypes.groups import OPERATIONS_WITH_BIAS
from nncf.openvino.graph.metatypes.groups import OPERATIONS_WITH_WEIGHTS
from nncf.openvino.graph.metatypes.openvino_metatypes import OVAddMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVConstantMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVConvertMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVConvolutionBackpropDataMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVGroupConvolutionBackpropDataMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVIfMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVMatMulMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVOpMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import get_node_metatype

InplaceInsertionFnType = Callable[[ov.Node, int], ov.Node]


def get_add_bias_node(node: NNCFNode, nncf_graph: NNCFGraph) -> Optional[NNCFNode]:
    """
    Returns Add node which stores bias for node.

    :param node: NNCFGraph node.
    :param nncf_graph: NNCFGraph.
    :return: Add node if exists.
    """
    for child in nncf_graph.get_next_nodes(node):
        if child.metatype == OVAddMetatype:
            bias_constant = get_node_with_bias_value(child, nncf_graph)
            if bias_constant is not None:
                return child
    return None


def is_node_with_bias(
    node: NNCFNode, nncf_graph: NNCFGraph, metatypes_with_bias: Optional[List[OVOpMetatype]] = None
) -> bool:
    """
    Checks if the node has a bias or not.

    :param node: The node to check.
    :param nncf_graph: NNCFGraph instance.
    :param metatypes_with_bias: List of the metatypes that contains biases.
    :return: Return `True` if `node` corresponds to the operation
        with bias (bias is added to the output tensor of that operation),
        `False` otherwise.
    """
    if metatypes_with_bias is None:
        metatypes_with_bias = OPERATIONS_WITH_BIAS

    if node.metatype not in metatypes_with_bias:
        return False

    # Since we do not verify bias constant shape, we need to verify the weight's existence at least.
    # layer_attributes adds only for nodes with weights, according to the nncf_graph_builder.py for the backend.
    if node.layer_attributes is None:
        return False

    return get_add_bias_node(node, nncf_graph) is not None


def get_number_if_op(model: ov.Model) -> int:
    """
    Returns number of If operation in a model.

    :param model: Model.
    :return: True if Model has If operation, False - otherwise.
    """

    def cnt_if_op(model: ov.Model, cnt: int) -> int:
        for op in model.get_ops():
            if get_node_metatype(op) == OVIfMetatype:
                cnt += 1
                cnt = cnt_if_op(op.get_function(0), cnt)
                cnt = cnt_if_op(op.get_function(1), cnt)
        return cnt

    return cnt_if_op(model, 0)


def get_const_value(const_node: ov.Node) -> np.ndarray:
    """
    Returns the constant tensor for the node.
    This method is applicable only for the floating-point constant data.

    :param const_node: OpenVINO node.
    :return: The constant value.
    """
    INPUT_DTYPE = os.environ.get("INPUT_DTYPE", "fp32")
    if const_node.get_element_type() == ov.Type.bf16 and INPUT_DTYPE != "bf16":
        # Fixed FP32 data type as the result for BF16 constant
        return const_node.get_data(dtype=np.float32)
    return const_node.data


def get_bias_value(node_with_bias: NNCFNode, nncf_graph: NNCFGraph, model: ov.Model) -> np.ndarray:
    """
    Returns the bias tensor for the biased node.

    :param node_with_bias: The node that corresponds to the operation with bias.
    :param nncf_graph: NNCFGraph instance.
    :param model: The model that contains this operation.
    :return: The bias value that is applied to the output tensor of the node's operation.
    """
    ops_dict = {op.get_friendly_name(): op for op in model.get_ops()}
    bias_constant = get_node_with_bias_value(get_add_bias_node(node_with_bias, nncf_graph), nncf_graph)
    ov_bias_constant = ops_dict[bias_constant.node_name]
    return get_const_value(ov_bias_constant)


def get_weight_value(node_with_weight: NNCFNode, model: ov.Model, port_id: int) -> np.ndarray:
    """
    Returns a weight value for the node with weight.

    :param node_with_weight: Node with weight.
    :param nncf_graph: NNCF graph.
    :param model: The model that contains this operation.
    :param port_id: The input port ID to get weight input.
    :return: The weight value.
    """
    const_op_friendly_name = node_with_weight.layer_attributes.constant_attributes[port_id]["name"]
    friendly_name_to_op_map = {op.get_friendly_name(): op for op in model.get_ops()}
    const_op = friendly_name_to_op_map[const_op_friendly_name]
    weight_tensor = get_const_value(const_op)
    return weight_tensor


def get_node_with_bias_value(add_node: NNCFNode, nncf_graph: NNCFGraph) -> Optional[NNCFNode]:
    """
    Returns node that represents bias constant in the NNCF graph, if it exists.

    :param add_node: NNCFNode that provides bias.
    :param nncf_graph: NNCFGraph instance.
    :return: Optional NNCFNode with bias value.
    """
    if add_node.layer_attributes is None:
        return None

    const_port_ids = add_node.layer_attributes.get_const_port_ids()
    assert len(const_port_ids) == 1
    bias_port_id = const_port_ids[0]
    bias_constant = nncf_graph.get_input_edge_by_port_id(add_node, bias_port_id).from_node

    if bias_constant.metatype == OVConvertMetatype:
        bias_constant = nncf_graph.get_input_edge_by_port_id(bias_constant, 0).from_node

    return bias_constant if bias_constant.metatype == OVConstantMetatype else None


def get_result_node_name(output_name: str, port_id: int) -> str:
    """
    Returns name of Result based on node name and its port.

    :param output_name: Node name.
    :param port_id: Node port.
    :return: Name of Result.
    """

    return f"Result_{output_name}.{port_id}"


def get_parameter_node_name(parameter_name: str, port_id: int) -> str:
    """
    Returns name of Parameter based on node name and its port.

    :param parameter_name: Node name.
    :param port_id: Node port.
    :return: Name of Parameter.
    """

    return f"Parameter_{parameter_name}.{port_id}"


def get_ov_model_reduce_node_name(output_name: str, reduce_node_name: str, port_id: int) -> str:
    """
    Returns name of reduce node based on output name, node type and port id.

    :param output_name: Target node name.
    :param node_type: Reduce node name.
    :param port_id: Target port id of the target node.
    :return: Reduce node name.
    """
    return f"{output_name}_{reduce_node_name}.{port_id}"


def get_inplace_reduce_op(
    op: Type[ov.Node], reduction_axes: Optional[ReductionAxes], use_abs: bool
) -> InplaceInsertionFnType:
    """
    Returns inplace insertion function that adds reduce node to a passed node.

    :param op: OpenVINO reduction operation type to insert.
    :param reduction_axes: Target reduction axes for the reduction node.
        Reduce along all axes in case reduction_axes are None.
    :param use_abs: Wheather reduce absolute values of input tensors or not.
    :returns: Inplace insertion function to use in ModelTransformer.
    """

    def get_reduce_op(node: ov.Node, output_port_id: int, output_node_name: str) -> ov.Node:
        reduction_axes_ = reduction_axes
        if reduction_axes_ is None:
            partial_shape = get_partial_shape_safe(node, output_port_id)
            reduction_axes_ = np.arange(partial_shape.rank.get_length()).astype(np.int64)

        if use_abs:
            op_input = opset.abs(node.output(output_port_id), name="abs_" + output_node_name)
            output_port_id = 0
        else:
            op_input = node

        return op(
            op_input.output(output_port_id),
            reduction_axes=np.array(reduction_axes_, dtype=np.int64),
            keep_dims=True,
            name=output_node_name,
        )

    return get_reduce_op


def get_inplace_min_op(reduction_axes: Optional[ReductionAxes]) -> InplaceInsertionFnType:
    """
    Returns inplace min function that adds reduce min node to a passed node.

    :param reduction_axes: Target reduction axes for the reduction node.
        Reduce along all axes in case reduction_axes are None.
    :returns: Inplace insertion function to use in ModelTransformer.
    """
    return get_inplace_reduce_op(opset.reduce_min, reduction_axes, False)


def get_inplace_max_op(reduction_axes: Optional[ReductionAxes], use_abs_max: bool) -> InplaceInsertionFnType:
    """
    Returns inplace max function that adds reduce max node to a passed node.

    :param reduction_axes: Target reduction axes for the reduction node.
        Reduce along all axes in case reduction_axes are None.
    :param use_abs: Wheather reduce absolute values of input tensors or not.
    :returns: Inplace insertion function to use in ModelTransformer.
    """
    return get_inplace_reduce_op(opset.reduce_max, reduction_axes, use_abs_max)


def get_inplace_mean_op(reduction_axes: Optional[ReductionAxes]) -> InplaceInsertionFnType:
    """
    Returns inplace mean function that adds reduce mean node to a passed node.

    :param reduction_axes: Target reduction axes for the reduction node.
        Reduce along all axes in case reduction_axes are None.
    :returns: Inplace insertion function to use in ModelTransformer.
    """
    return get_inplace_reduce_op(opset.reduce_mean, reduction_axes, False)


def get_inplace_batch_mean_op() -> InplaceInsertionFnType:
    """
    Returns inplace batch mean function that adds reduce batch mean node to a passed node.

    :returns: Inplace insertion function to use in ModelTransformer.
    """
    return get_inplace_reduce_op(opset.reduce_mean, np.array(0), False)


def get_inplace_mean_per_ch(axis: int) -> InplaceInsertionFnType:
    """
    Returns inplace mean per channel function that adds reduce mean per channel node
    to a passed node.

    :param axis: Channel axis.
    :returns: Inplace insertion function to use in ModelTransformer.
    """

    def get_reduce_op(node: ov.Node, output_port_id: int, output_node_name: str) -> ov.Node:
        input_shape = get_partial_shape_safe(node, output_port_id)
        input_shape = [dim.get_length() if dim.is_static else -1 for dim in input_shape]
        if len(input_shape) < 3:
            return opset.reduce_mean(
                node.output(output_port_id),
                reduction_axes=0,
                keep_dims=False,
                name=output_node_name,
            )

        ch_dim = 1
        if axis != ch_dim:
            transpose_dims = list(range(len(input_shape)))
            transpose_dims[axis], transpose_dims[ch_dim] = transpose_dims[ch_dim], transpose_dims[axis]
            transposed_shape = [input_shape[dim] for dim in transpose_dims]

            reshape_input_node = opset.transpose(node.output(output_port_id), transpose_dims)
            output_port_id = 0
        else:
            reshape_input_node = node
            transposed_shape = input_shape

        keeped_dims = transposed_shape[:2]
        keeped_dims = [0 if dim < 0 else dim for dim in keeped_dims]
        squized_dims = -1 if -1 in transposed_shape[2:] else np.prod(transposed_shape[2:])
        reshape_op = opset.reshape(
            reshape_input_node.output(output_port_id),
            output_shape=np.array((keeped_dims[0], keeped_dims[1], squized_dims)),
            special_zero=True,
        )
        return opset.reduce_mean(
            reshape_op,
            reduction_axes=np.array((0, 2)),
            keep_dims=False,
            name=output_node_name,
        )

    return get_reduce_op


def get_partial_shape_safe(node, port_id) -> Tuple[int, ...]:
    partial_shape = node.get_output_partial_shape(port_id)
    if partial_shape.rank.is_dynamic or not partial_shape.all_non_negative:
        raise nncf.ValidationError(
            f"Could not collect statistics for the node {node} because its output shape rank is dynamic or negative"
        )
    return partial_shape


def get_reducer_output_node_names(
    node_type, target_node_name: str, port_id: int, fn_output_port_id: int, inplace: bool
) -> List[str]:
    """
    Returns output name to feed to a reducer node.

    :param node_type: String that describes reduce node type.
    :param target_node_name: Name of the node inputs/outputs/weights of which was
        used for reduction.
    :param port_id: Target port id of the target node.
    :param fn_output_port_id: Port id of the reducer subgraph.
    :param inplace: Wheather reducer calculated inplace or not.
    :return: Output names to feed to a reducer node.
    """
    if inplace:
        target_node_name = get_ov_model_reduce_node_name(target_node_name, node_type, port_id)
        return [get_result_node_name(target_node_name, fn_output_port_id)]
    return [get_result_node_name(target_node_name, port_id)]


def get_weight_channel_axes(node: NNCFNode) -> List[int]:
    """
    Returns axes numbers of the weight tensor which correspond to its channels.

    :param node: NNCFNode with weights.
    :param weights_port_id: Weight port id of the target node.
    :return: Axes numbers of the weight tensor which correspond to its channels.
    """
    if node.metatype not in OPERATIONS_WITH_WEIGHTS:
        raise ValueError("Channel axis cannot be defined for operation without weights.")

    if node.metatype in CONV_OPERATIONS:
        weights_layout = get_conv_weights_layout_from_node(node)
        return [idx for idx, elem in enumerate(weights_layout) if elem in [OVLayoutElem.GROUPS, OVLayoutElem.C_OUT]]
    elif node.metatype == OVMatMulMetatype:
        return get_matmul_channel_axes(node)
    return node.metatype.const_channel_axis


def get_matmul_channel_axes(node: ov.Node) -> List[int]:
    """
    Calculate channel axes for the MatMul operation.

    :param node: The target node.
    :return: List of channel axes for the MatMul operation.
    """
    weights_layout = get_linear_weights_layout_from_node(node)
    return [idx for idx, elem in enumerate(weights_layout) if elem in [OVLayoutElem.SPATIAL, OVLayoutElem.C_OUT]]


def create_bias_tensor(node_without_bias: NNCFNode, graph: NNCFGraph, value: Any) -> np.ndarray:
    """
    Creates bias value constant array filled by given value.

    :param node_without_bias: NNCFNode to add bias to.
    :param graph: Target NNCFgraph.
    :param value: Value to fill bias constant array.
    :return: Bias value constant array filled by given value.
    """
    node_shape = graph.get_output_edges(node_without_bias)[0].tensor_shape
    bias_shape = [1] * len(node_shape)
    channel_axis = node_without_bias.metatype.output_channel_axis
    bias_shape[channel_axis] = node_shape[1]
    return np.full(bias_shape, value)


def get_weighted_layer_attributes(
    ov_node: ov.Node, ov_metatype: OVOpMetatype, constant_attributes: Dict[int, Any]
) -> WeightedLayerAttributes:
    """
    Function retrieves common layer attributes from the given node.

    :param ov_node: TargetOpenvino graph node instance.
    :param ov_metatype: NNCF Openvino metatype of the given node.
    :param constant_attributes: Constant attributes collected for the given node.
    :return: Weighted layer attributes for the given node.
    """
    if len(constant_attributes) != 1:
        return None

    port_id, attrs = constant_attributes.copy().popitem()
    if ov_metatype in CONV_OPERATIONS:
        node_attrs = ov_node.get_attributes()
        kwargs = {
            "weight_requires_grad": False,
            "stride": tuple(node_attrs["strides"]),
            "dilations": node_attrs["dilations"],
            "transpose": ov_metatype in [OVConvolutionBackpropDataMetatype, OVGroupConvolutionBackpropDataMetatype],
            # TODO: ticket 114378: unify pad attribute
            "padding_values": tuple(node_attrs["pads_begin"] + node_attrs["pads_end"]),
        }
        weights_shape = attrs["shape"]
        weights_layout = get_conv_weights_layout(ov_metatype=ov_metatype, weights_shape=weights_shape)
        kwargs.update(
            {
                "in_channels": weights_shape[weights_layout.index(OVLayoutElem.C_IN)],
                "out_channels": weights_shape[weights_layout.index(OVLayoutElem.C_OUT)],
                "kernel_size": tuple(
                    dim for dim, elem in zip(weights_shape, weights_layout) if elem == OVLayoutElem.SPATIAL
                ),
                "groups": (
                    weights_shape[weights_layout.index(OVLayoutElem.GROUPS)]
                    if OVLayoutElem.GROUPS in weights_layout
                    else 1
                ),
            }
        )

        return ConvolutionLayerAttributes(**kwargs)
    if ov_metatype == OVMatMulMetatype:
        weights_shape = attrs["shape"]
        weights_layout = get_linear_input_layout(
            input_shape=weights_shape, transpose=attrs["transpose"], port_id=port_id
        )

        kwargs = {
            "weight_requires_grad": False,
            "in_features": weights_shape[weights_layout.index(OVLayoutElem.C_IN)],
            "out_features": (
                weights_shape[weights_layout.index(OVLayoutElem.C_OUT)]
                if OVLayoutElem.C_OUT in weights_layout
                else None
            ),
            "with_bias": False,
        }
        return LinearLayerAttributes(**kwargs)
    return GenericWeightedLayerAttributes(weight_requires_grad=False, weight_shape=attrs.get("shape", None))


def get_activation_channel_axis(node: NNCFNode, port_id: int, input_shape: Tuple[int]) -> int:
    """
    Returns axis number of the activation tensor which correspond to it channel.

    :param node: NNCFNode instance.
    :param port_id: Port ID for input.
    :param input_shape: Shape of the input.
    :return: Channel axis number.
    """
    # In case of the OpenVINO, [N, C, ..] layout applicable for most quantizable layers.
    channel_axis = 1

    # But the MatMul layers may transpose inputs internally.
    if node.metatype == OVMatMulMetatype:
        activations_layout = get_linear_activations_layout_from_node(node, port_id, input_shape)
        channel_axis = activations_layout.index(OVLayoutElem.C_IN)

    return channel_axis
