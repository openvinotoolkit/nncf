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

from typing import Any, Callable, Optional

import numpy as np
import openvino as ov
import openvino.op as op
import openvino.opset13 as opset

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
from nncf.tensor import Tensor
from nncf.tensor import TensorBackend

InplaceInsertionFnType = Callable[[ov.Node, int, str], ov.Node]


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
    node: NNCFNode, nncf_graph: NNCFGraph, metatypes_with_bias: Optional[list[OVOpMetatype]] = None
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
        for model_op in model.get_ops():
            if get_node_metatype(model_op) == OVIfMetatype:
                cnt += 1
                cnt = cnt_if_op(model_op.get_function(0), cnt)
                cnt = cnt_if_op(model_op.get_function(1), cnt)
        return cnt

    return cnt_if_op(model, 0)


def get_const_value_as_numpy_tensor(const_node: ov.Node) -> np.ndarray:
    """
    Returns the constant tensor for the node as an instance of np.ndarray. BF16 constants will be converted to FP32.
    This method is applicable only for the floating-point constant data.

    :param const_node: OpenVINO node.
    :return: The constant value.
    """
    if const_node.get_element_type() == ov.Type.bf16:
        return const_node.get_data(dtype=np.float32)
    return const_node.data


def get_const_value_as_ov_tensor(const_node: ov.Node) -> ov.Tensor:
    """
    Returns the constant tensor for the node as an instance of openvino.Tensor which is useful when BF16 constant
    needs to be retrieved as is.

    :param const_node: OpenVINO node.
    :return: The constant value as openvino.Tensor.
    """
    return ov.Tensor(const_node.data, const_node.get_output_shape(0), const_node.get_element_type())


def get_bias_value(
    node_with_bias: NNCFNode, nncf_graph: NNCFGraph, model: ov.Model, node_mapping: dict[str, ov.Node] = None
) -> np.ndarray:
    """
    Returns the bias tensor for the biased node.

    :param node_with_bias: The node that corresponds to the operation with bias.
    :param nncf_graph: NNCFGraph instance.
    :param model: The model that contains this operation.
    :param node_mapping: Original nodes mapping cache.
    :return: The bias value that is applied to the output tensor of the node's operation.
    """
    if node_mapping is None:
        node_mapping = {op.get_friendly_name(): op for op in model.get_ops()}
    bias_constant = get_node_with_bias_value(get_add_bias_node(node_with_bias, nncf_graph), nncf_graph)
    ov_bias_constant = node_mapping[bias_constant.node_name]
    return get_const_value_as_numpy_tensor(ov_bias_constant)


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
    weight_tensor = get_const_value_as_numpy_tensor(const_op)
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
    op: type[ov.Node],
    reduction_axes: Optional[ReductionAxes],
    use_abs: bool,
    keep_dims: bool = True,
) -> InplaceInsertionFnType:
    """
    Returns inplace insertion function that adds reduce node to a passed node.

    :param op: OpenVINO reduction operation type to insert.
    :param reduction_axes: Target reduction axes for the reduction node.
        Reduce along all axes in case reduction_axes are None.
    :param use_abs: Whether reduce absolute values of input tensors or not.
    :param keep_dims: Whether to keep the original dimension length or return result as a scalar.
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
            keep_dims=keep_dims,
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


def get_inplace_max_op(
    reduction_axes: Optional[ReductionAxes], use_abs_max: bool, keep_dims: bool = True
) -> InplaceInsertionFnType:
    """
    Returns inplace max function that adds reduce max node to a passed node.

    :param reduction_axes: Target reduction axes for the reduction node.
        Reduce along all axes in case reduction_axes are None.
    :param use_abs_max: Whether reduce absolute values of input tensors or not.
    :param keep_dims: Whether to keep the original dimension length or return result as a scalar.
    :returns: Inplace insertion function to use in ModelTransformer.
    """
    return get_inplace_reduce_op(opset.reduce_max, reduction_axes, use_abs_max, keep_dims)


def get_inplace_mean_op(reduction_axes: Optional[ReductionAxes]) -> InplaceInsertionFnType:
    """
    Returns inplace mean function that adds reduce mean node to a passed node.

    :param reduction_axes: Target reduction axes for the reduction node.
        Reduce along all axes in case reduction_axes are None.
    :returns: Inplace insertion function to use in ModelTransformer.
    """
    return get_inplace_reduce_op(opset.reduce_mean, reduction_axes, False)


def var_op(
    op_input: ov.Output, output_node_name: str, reduction_axes: Optional[np.ndarray] = None, keep_dims: bool = True
) -> ov.Node:
    """
    Return a subgraph computing variance on a given output.

    :param op_input: An output to compute variance for.
    :param output_node_name: Variance output name.
    :param keep_dims: Whether to keep the original dimension length or return result as a scalar.
    :param reduction_axes: Axes along which to compute variance.
    """
    mean = opset.reduce_mean(
        op_input,
        reduction_axes=reduction_axes,
        keep_dims=True,
        name=f"{output_node_name}/mean",
    )
    diff = opset.squared_difference(mean, op_input, name=f"{output_node_name}/squared_diff")
    variance = opset.reduce_mean(
        diff,
        reduction_axes=reduction_axes,
        keep_dims=keep_dims,
        name=output_node_name,
    )
    return variance


def get_inplace_mean_var_op(reduction_axes: Optional[ReductionAxes] = None) -> InplaceInsertionFnType:
    """
    Return an operation getter function that computes variance across given axes and then mean-reduces the result across
    the remaining axes.

    :param reduction_axes: Axes along which to compute variance.
    """

    def get_mean_var_reduce_op(node: ov.Node, output_port_id: int, output_node_name: str) -> ov.Node:
        partial_shape = get_partial_shape_safe(node, output_port_id)
        all_axes = np.arange(partial_shape.rank.get_length()).astype(np.int64)
        reduction_axes_ = np.array(all_axes if reduction_axes is None else reduction_axes, dtype=np.int64)

        reduce_all = np.array_equal(reduction_axes_, all_axes)
        var_op_name = output_node_name if reduce_all else f"{output_node_name}/var"
        result = var_op(node.output(output_port_id), var_op_name, reduction_axes_, keep_dims=not reduce_all)
        if not reduce_all:
            result = opset.reduce_mean(
                result,
                reduction_axes=all_axes,
                keep_dims=False,
                name=output_node_name,
            )

        return result

    return get_mean_var_reduce_op


def get_inplace_max_var_op(reduction_axes: Optional[ReductionAxes] = None) -> InplaceInsertionFnType:
    """
    Return an operation getter function that computes variance across given axes and then max-reduces the result across
    the remaining axes.

    :param reduction_axes: Axes along which to compute variance.
    """

    def get_max_var_reduce_op(node: ov.Node, output_port_id: int, output_node_name: str) -> ov.Node:
        partial_shape = get_partial_shape_safe(node, output_port_id)
        all_axes = np.arange(partial_shape.rank.get_length()).astype(np.int64)
        reduction_axes_ = np.array(all_axes if reduction_axes is None else reduction_axes, dtype=np.int64)

        reduce_all = np.array_equal(reduction_axes_, all_axes)
        var_op_name = output_node_name if reduce_all else f"{output_node_name}/var"
        result = var_op(node.output(output_port_id), var_op_name, reduction_axes_, keep_dims=not reduce_all)
        if not reduce_all:
            result = opset.reduce_max(
                result,
                reduction_axes=all_axes,
                keep_dims=False,
                name=output_node_name,
            )

        return result

    return get_max_var_reduce_op


def get_inplace_mean_max_op(reduction_axes: Optional[ReductionAxes], use_abs_max: bool) -> InplaceInsertionFnType:
    """
    Return an operation getter function that computes maximum across given axes and then mean-reduces the result across
    the remaining axes.

    :param reduction_axes: Axes to compute maximum across.
    :param use_abs_max: Whether to apply abs() operation before the max operation.
    """

    def get_mean_max_reduce_op(node: ov.Node, output_port_id: int, output_node_name: str) -> ov.Node:
        partial_shape = get_partial_shape_safe(node, output_port_id)
        all_axes = np.arange(partial_shape.rank.get_length()).astype(np.int64)
        reduction_axes_ = np.array(all_axes if reduction_axes is None else reduction_axes, dtype=np.int64)

        reduce_all = np.array_equal(reduction_axes_, all_axes)
        max_op_name = output_node_name if reduce_all else f"{output_node_name}/max"
        result = get_inplace_max_op(reduction_axes, use_abs_max, keep_dims=not reduce_all)(
            node, output_port_id, max_op_name
        )
        if not reduce_all:
            result = opset.reduce_mean(
                result,
                reduction_axes=all_axes,
                keep_dims=False,
                name=output_node_name,
            )

        return result

    return get_mean_max_reduce_op


def get_inplace_shape_op() -> InplaceInsertionFnType:
    """
    Return an operation returning a shape on the given output.
    """

    def get_shape_op(node: ov.Node, output_port_id: int, output_node_name: str) -> ov.Node:
        result = opset.shape_of(node.output(output_port_id), name=output_node_name)
        return result

    return get_shape_op


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

        kept_dims = transposed_shape[:2]
        kept_dims = [0 if dim < 0 else dim for dim in kept_dims]
        squeezed_dims = -1 if -1 in transposed_shape[2:] else np.prod(transposed_shape[2:])
        reshape_op = opset.reshape(
            reshape_input_node.output(output_port_id),
            output_shape=np.array((kept_dims[0], kept_dims[1], squeezed_dims)),
            special_zero=True,
        )
        return opset.reduce_mean(
            reshape_op,
            reduction_axes=np.array((0, 2)),
            keep_dims=False,
            name=output_node_name,
        )

    return get_reduce_op


def get_partial_shape_safe(node, port_id) -> tuple[int, ...]:
    partial_shape = node.get_output_partial_shape(port_id)
    if partial_shape.rank.is_dynamic or not partial_shape.all_non_negative:
        msg = f"Could not collect statistics for the node {node} because its output shape rank is dynamic or negative"
        raise nncf.ValidationError(msg)
    return partial_shape


def get_reducer_output_node_names(
    node_type, target_node_name: str, port_id: int, fn_output_port_id: int, inplace: bool
) -> list[str]:
    """
    Returns output name to feed to a reducer node.

    :param node_type: String that describes reduce node type.
    :param target_node_name: Name of the node inputs/outputs/weights of which was
        used for reduction.
    :param port_id: Target port id of the target node.
    :param fn_output_port_id: Port id of the reducer subgraph.
    :param inplace: Whether reducer calculated inplace or not.
    :return: Output names to feed to a reducer node.
    """
    if inplace:
        target_node_name = get_ov_model_reduce_node_name(target_node_name, node_type, port_id)
        return [get_result_node_name(target_node_name, fn_output_port_id)]
    return [get_result_node_name(target_node_name, port_id)]


def get_weight_channel_axes(node: NNCFNode) -> list[int]:
    """
    Returns axes numbers of the weight tensor which correspond to its channels.

    :param node: NNCFNode with weights.
    :param weights_port_id: Weight port id of the target node.
    :return: Axes numbers of the weight tensor which correspond to its channels.
    """
    if node.metatype not in OPERATIONS_WITH_WEIGHTS:
        msg = "Channel axis cannot be defined for operation without weights."
        raise ValueError(msg)

    if node.metatype in CONV_OPERATIONS:
        weights_layout = get_conv_weights_layout_from_node(node)
        return [idx for idx, elem in enumerate(weights_layout) if elem in [OVLayoutElem.GROUPS, OVLayoutElem.C_OUT]]
    elif node.metatype == OVMatMulMetatype:
        return get_matmul_channel_axes(node)
    return node.metatype.const_channel_axis


def get_matmul_channel_axes(node: ov.Node) -> list[int]:
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
    ov_node: ov.Node, ov_metatype: OVOpMetatype, constant_attributes: dict[int, Any]
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


def get_activation_channel_axis(node: NNCFNode, port_id: int, input_shape: tuple[int]) -> int:
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


def convert_op(node: ov.Node, target_dtype: ov.Type) -> ov.Node:
    """
    Return a subgraph which converts the given node output to the target data type. If the output is already in the
    target data type then the given node is returned.

    :param node: The input node to convert.
    :param target_dtype: The target data type to convert the input node to.
    :return: The converted node.
    """
    if node.get_element_type() == target_dtype:
        return node
    return opset.convert(node, target_dtype)


def non_convertable_divide_op(a: ov.Node, b: ov.Node) -> ov.Node:
    """
    Creates a "non-convertable" divide operation. It won't be converted to a*(1/b).
    """
    divide_node = a / b
    divide_node.get_rt_info()["nonconvertable_divide_0"] = True
    return divide_node


def create_ov_const_from_tensor(x: Tensor, dtype: ov.Type, name: Optional[str] = None) -> op.Constant:
    """
    Create an OpenVINO Constant node from the given tensor.
    :param x: Data tensor. Supports NumPy and OV tensor backends. If x backend is OV, the constant node is created
        directly from underlying OV tensor.
    :param dtype: Data type of the constant.
    :param name: Optional name of the constant.
    :return: OpenVINO Constant node.
    """
    if x.backend == TensorBackend.ov:
        assert x.data.get_element_type() == dtype
        return opset.constant(x.data, name=name, shared_memory=True)
    const = opset.constant(x.data, dtype=dtype, name=name)
    return const


def create_ov_codebook_subgraph(
    codebook: Tensor, indexes: Tensor, dtype: ov.Type, name: Optional[str] = None
) -> op.Constant:
    """
    Create an OpenVINO subgraph with gather from the given codebook and indexes tensors.

    :param codebook: Codebook tensor.
    :param indexes: Indexes tensor.
    :param dtype: Data type of the indexes.
    :param name: Optional name of the constant.
    :return: OpenVINO subgraph.
    """
    codebook_const = opset.constant(codebook.data)
    if codebook.dtype != ov.Type.f16:
        codebook_const = opset.convert(codebook_const, destination_type=ov.Type.f16)

    codebook_indexes = opset.constant(indexes.data, dtype=dtype)
    if dtype == ov.Type.u4:
        codebook_indexes = opset.convert(codebook_indexes, destination_type=ov.Type.u8)

    const = opset.gather(codebook_const, codebook_indexes, 0, name=name)
    return const
