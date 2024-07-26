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

from typing import Callable, List, Optional

import torch
import torch.fx
from torch.ao.quantization.fx.utils import create_getattr_from_value
from torch.ao.quantization.pt2e.utils import _fuse_conv_bn_
from torch.ao.quantization.pt2e.utils import _get_tensor_constant_from_node
from torch.quantization.fake_quantize import FakeQuantize

import nncf
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.transformations.commands import TargetType
from nncf.experimental.torch.fx.node_utils import get_graph_node_by_name
from nncf.torch.graph.transformations.commands import PTTargetPoint

TransformationFNType = Callable[[torch.fx.GraphModule], None]


def leaf_module_insertion_transformation_builder(
    module_to_insert: torch.nn.Module, target_points: List[PTTargetPoint]
) -> TransformationFNType:
    """
    Returns transformation which inserts given module to a target model
    and calls given module after each target points.

    :param module_to_insert: Given torch.nn.Module to insert.
    :param target_points: Target points to insert the target module.
    :returns: Transformation which which inserts given module to a target model
        and calls given module after each target points.
    """

    def leaf_module_insertion_transformation(model: torch.fx.GraphModule):
        module_attr_name = _set_module_to_the_graph_module(model, module_to_insert, target_points)
        # Insert call_module nodes to the model
        graph = model.graph
        for target_point in target_points:
            target_node = _get_target_node(graph, target_point)
            _insert_call_module(graph, target_node, module_attr_name)

    return leaf_module_insertion_transformation


def bias_update_transformation_builder(node: NNCFNode, value: torch.Tensor) -> TransformationFNType:
    """
    Return transformation which updates constant of the given bias node to the given value.

    :param node: Bias node which requires bias constant update.
    :param value: New value to use as the bias constant.
    :return: Transformation which updates constant of the given bias node to the given value.
    """

    def bias_update_transformation(model: torch.fx.GraphModule):
        graph = model.graph
        target_node_name = node.node_name
        graph_node = get_graph_node_by_name(graph, target_node_name)
        if len(graph_node.users) != 1:
            raise nncf.InternalError(f"Node with bias have {len(graph_node.users)} users, 1 expected.")

        bias_node = next(iter(graph_node.users))
        with graph.inserting_before(bias_node):
            new_constant = create_getattr_from_value(model, graph, target_node_name + "_shifted_bias", value)

        args = list(bias_node.args)
        # A bias node suppose to have constant on the second input port.
        args[1] = new_constant
        bias_node.args = tuple(args)
        graph.eliminate_dead_code()

    return bias_update_transformation


def qdq_insertion_tranformation_builder(
    quantizer: FakeQuantize, target_points: List[PTTargetPoint]
) -> TransformationFNType:
    """
    Returns transformation which inserts quantize-dequantize operations with parameters
    inherited from the given quantizer to each given target point.

    :param quantizer: Quantizer module to inherit quantization parameters from.
    :param target_points: List of target point used to insert quantize-dequantize pairs.
    :return: Transformation which inserts quantize-dequantize operations with parameters
        inherited from the given quantizer to each given target point.
    """

    def qdq_insertion_tranformation(model: torch.fx.GraphModule):
        if any(tp.target_type != TargetType.OPERATION_WITH_WEIGHTS for tp in target_points) and len(target_points) > 1:
            raise RuntimeError(
                "Insertion of shared qdq pair for the weights is not supported."
                " Please use non shared qdq pairs for the weights quantization."
            )
        for target_point in target_points:
            target_node = _get_target_node(model.graph, target_point)
            insert_one_qdq_after_node(model, target_node, quantizer)

    return qdq_insertion_tranformation


def insert_one_qdq_after_node(model: torch.fx.GraphModule, target_node: torch.fx.Node, quantizer: FakeQuantize):
    """
    Inserts quantize-dequantize after the target node to the target model.

    :param model: Target model.
    :param target_node: Target node, quantizer-dequantizer pair is inserted just after the
        target node.
    :param quantizer: Quantizer module to inherit quantization parameters from.
    """

    # Copied from torch.ao.quantization.quantize_pt2e.convert_pt2e
    # 1. extract information for inserting q/dq node from activation_post_process
    node_type = "call_function"
    quantize_op: Optional[Callable] = None

    dtype = torch.int8 if quantizer.quant_min < 0 else torch.uint8
    if quantizer.is_per_channel:
        qparams = {
            "_scale_": quantizer.scale,
            "_zero_point_": quantizer.zero_point,
            "_axis_": quantizer.ch_axis,
            "_quant_min_": quantizer.quant_min,
            "_quant_max_": quantizer.quant_max,
            "_dtype_": dtype,
        }
        quantize_op = torch.ops.quantized_decomposed.quantize_per_channel.default
        dequantize_op = torch.ops.quantized_decomposed.dequantize_per_channel.default
    else:
        qparams = {
            "_scale_": float(quantizer.scale),
            "_zero_point_": int(quantizer.zero_point),
            "_quant_min_": quantizer.quant_min,
            "_quant_max_": quantizer.quant_max,
            "_dtype_": dtype,
        }
        quantize_op = torch.ops.quantized_decomposed.quantize_per_tensor.default
        dequantize_op = torch.ops.quantized_decomposed.dequantize_per_tensor.default

    # 2. replace activation_post_process node with quantize and dequantize
    graph = model.graph
    # TODO(dlyakhov): use metatype to get correct input_port_id
    # Do not quantize already quantized nodes
    # inserting_before handle only order in the graph generated code.
    # so, inserting quantize-dequantize and all constant nodes before the usage of the nodes
    with graph.inserting_before(target_node):
        quantize_op_inputs = [target_node]
        for key, value_or_node in qparams.items():
            # TODO(dlyakhov): we can add the information of whether a value needs to
            # be registered as an attribute in qparams dict itself
            if key in ["_scale_", "_zero_point_"] and (not isinstance(value_or_node, (float, int))):
                # For scale and zero_point values we register them as buffers in the root module.
                # However, note that when the values are not tensors, as in the case of
                # per_tensor quantization, they will be treated as literals.
                # However, registering them as a node seems to cause issue with dynamo
                # tracing where it may consider tensor overload as opposed to default.
                # With extra check of scale and zero_point being scalar, it makes
                # sure that the default overload can be used.
                # TODO(dlaykhov): maybe need more complex attr name here
                qparam_node = create_getattr_from_value(model, graph, target_node.name + key, value_or_node)
                quantize_op_inputs.append(qparam_node)
            else:
                # for qparams that are not scale/zero_point (like axis, dtype) we store
                # them as literals in the graph.
                quantize_op_inputs.append(value_or_node)
    with graph.inserting_after(target_node):
        quantized_node = graph.create_node(node_type, quantize_op, tuple(quantize_op_inputs), {})
        # use the same qparams from quantize op
    dq_inputs = [quantized_node] + quantize_op_inputs[1:]
    user_dq_nodes = []
    with graph.inserting_after(quantized_node):
        for user in target_node.users:
            if user is quantized_node:
                continue
            user_dq_nodes.append((user, graph.call_function(dequantize_op, tuple(dq_inputs), {})))

    for user, dq_node in user_dq_nodes:
        user.replace_input_with(target_node, dq_node)


def _insert_call_module(graph: torch.fx.Graph, target_node: torch.fx.Node, module_attr_name: str):
    """
    Inserts module call node to the graph after the target node.

    :param graph: Graph to insert module call node.
    :param target_node: Target node, module call node is being iserted just after the target node.
    :param module_attr_name: The name of the graph attribute which keeps the target module.
    """
    with graph.inserting_after(target_node):
        return graph.create_node(
            "call_module", module_attr_name, (target_node,), {}, name=module_attr_name + "_graph_node"
        )


def _get_target_node(graph: torch.fx.Graph, target_point: PTTargetPoint) -> torch.fx.Node:
    """
    Returns TorchFX graph node correspondent to the target point.

    :param graph: Target torch.fx.Graph.
    :param target_point: A target point to find the target node.
    :return: TorchFX graph node correspondent to the target point.
    """
    # TODO(dlyakhov): Support node insertion on a specific input port id.
    target_type = target_point.target_type
    target_node = get_graph_node_by_name(graph, target_point.target_node_name)
    if target_type in [TargetType.OPERATOR_PRE_HOOK, TargetType.OPERATION_WITH_WEIGHTS]:
        target_node = target_node.all_input_nodes[target_point.input_port_id]
    elif target_type != TargetType.OPERATOR_POST_HOOK:
        raise RuntimeError(f"Unsupported target type: {target_type} for target_point: {target_point}")
    return target_node


def _set_module_to_the_graph_module(
    model: torch.fx.GraphModule, module_to_insert: torch.nn.Module, target_points: List[PTTargetPoint]
) -> str:
    """
    Sets given module to the given torch.fx.GraphModule with unique name.

    :param graph: Target torch.fx.Graph.
    :param module_to_insert: Module to insert to the target graph.
    :param target_points: Target points which will be used to insert target module
        to the graph.
    :return: A graph module attribute name which keep given module.
    """
    module_to_insert = module_to_insert
    # TODO(dlyakhov) Make module name human readable.
    module_name_in_model = (
        "__".join(
            "_".join((tp.target_node_name, str(tp.input_port_id), str(tp.target_type.value))) for tp in target_points
        )
        + "_"
        + str(id(module_to_insert))
    )
    assert not hasattr(model, module_name_in_model)
    setattr(model, module_name_in_model, module_to_insert)
    return module_name_in_model


def apply_quantization_transformations(model: torch.fx.GraphModule) -> None:
    """
    Applies quantization transformations to the model.
    :param model: Model to apply transformations to.
    """
    # BatchNorm operations have 3 output ports,
    # to make it easier for alorithms to work
    # with the target graph BatchNorm operations
    # are being fused
    _fuse_conv_bn_(model)
    separate_conv_and_bias(model)
    separate_linear_and_bias(model)


def revert_quantization_transformations(model: torch.fx.GraphModule) -> None:
    """
    Reverts quantization transformations from the model.
    :param model: Model to revert transformations from.
    """
    merge_conv_and_bias(model)
    merge_linear_and_bias(model)


def _is_linear(n: torch.fx.Node) -> bool:
    """
    Return whether the node refers to an aten linear op.

    :param n: The given node.
    :return: True if given node is a linear node, else False.
    """
    return n.op == "call_function" and n.target in (torch.ops.aten.linear.default,)


def _is_conv(n: torch.fx.Node):
    """
    Return whether the node refers to an aten conv op.
    """
    return n.op == "call_function" and n.target in (
        torch.ops.aten.conv1d.default,
        torch.ops.aten.conv2d.default,
    )


def separate_linear_and_bias(model: torch.fx.GraphModule):
    """
    Separates one joined linear+bias node to two nodes: conv and bias.
    Needed as nncf does not expect joined conv

    :param model: Target model.
    """
    add_node_target = torch.ops.aten.add_.Tensor
    for n in model.graph.nodes:
        if not _is_linear(n):
            continue
        if len(n.args) < 3 or n.args[2] is None:
            continue
        linear_node = n
        linear_bias_node = linear_node.args[2]
        while linear_bias_node.op != "get_attr":
            # Assume zero argument is on a path to the constant
            linear_bias_node = linear_bias_node.args[0]
        linear_bias_value = _get_tensor_constant_from_node(linear_bias_node, model)
        args = list(n.args)
        args[2] = None
        linear_node.args = tuple(args)
        with model.graph.inserting_after(linear_node):
            new_linear_bias_node = create_getattr_from_value(
                model,
                model.graph,
                linear_bias_node.name + "_",
                linear_bias_value,
            )
        with model.graph.inserting_after(new_linear_bias_node):
            add_node = model.graph.create_node(
                "call_function", add_node_target, (linear_node, new_linear_bias_node), {}
            )
        for user in list(linear_node.users):
            if user is add_node:
                continue
            user.replace_input_with(linear_node, add_node)
        if "val" in linear_node.meta:
            add_node.meta["val"] = linear_node.meta["val"]
    model.graph.eliminate_dead_code()
    model.recompile()


def separate_conv_and_bias(model: torch.fx.GraphModule):
    """
    Separates one joined conv+bias node to two nodes: conv and bias.
    Needed as nncf does not expect joined conv

    :param model: Target model.
    """
    add_node_target = torch.ops.aten.add_.Tensor
    for n in model.graph.nodes:
        if not _is_conv(n):
            continue
        if len(n.args) < 3 or n.args[2] is None:
            continue
        conv_node = n
        dims = len(_get_tensor_constant_from_node(conv_node.args[1], model).shape)
        conv_bias_node = conv_node.args[2]
        conv_bias_value = _get_tensor_constant_from_node(conv_bias_node, model)
        args = list(n.args)
        args[2] = None
        conv_node.args = tuple(args)
        with model.graph.inserting_after(conv_node):
            new_conv_bias_node = create_getattr_from_value(
                model, model.graph, conv_bias_node.name + "_", conv_bias_value.reshape((1, -1) + (1,) * (dims - 2))
            )
        with model.graph.inserting_after(new_conv_bias_node):
            add_node = model.graph.create_node("call_function", add_node_target, (conv_node, new_conv_bias_node), {})
        for user in list(conv_node.users):
            if user is add_node:
                continue
            user.replace_input_with(conv_node, add_node)

        if "val" in conv_node.meta:
            add_node.meta["val"] = conv_node.meta["val"]
    model.graph.eliminate_dead_code()
    model.recompile()


def merge_conv_and_bias(model: torch.fx.GraphModule):
    """
    Merges two separate conv and bias nodes to a one node: conv+bias.
    Needed as nncf does not expect joined conv

    :param model: Target model.
    """
    _merge_node_and_bias(model, _is_conv)


def merge_linear_and_bias(model: torch.fx.GraphModule):
    """
    Merges two separate linear and bias nodes to a one node: linear+bias.

    :param model: Target model.
    """
    _merge_node_and_bias(model, _is_linear)


def _merge_node_and_bias(model: torch.fx.GraphModule, is_target_node: Callable[[torch.fx.Node], bool]):
    """
    Merges two separate node and bias node to a one node: node+bias.
    Check which node should be merged by the given `is_target_node` predicate.

    :param model: Target model.
    :param is_target_node: Predicate to specify nodes which shoudld be merged with the bias
    """
    add_node_targets = (torch.ops.aten.add_.Tensor,)
    for n in model.graph.nodes:
        if not is_target_node(n):
            continue
        if len(n.args) > 2 and n.args[2] is not None:
            continue
        bias_node = next(iter(n.users))
        if len(n.users) > 1 or bias_node.target not in add_node_targets:
            continue
        conv_node = n
        const_node = None
        for node in bias_node.all_input_nodes:
            if node is not conv_node:
                const_node = node
                break
        assert const_node is not None
        bias_value = _get_tensor_constant_from_node(const_node, model).squeeze()
        with model.graph.inserting_before(conv_node):
            new_bias_node = create_getattr_from_value(model, model.graph, const_node.name + "_", bias_value)
        args = list(conv_node.args)
        args[2] = new_bias_node
        conv_node.args = tuple(args)
        for user in list(bias_node.users):
            user.replace_input_with(bias_node, conv_node)

    model.graph.eliminate_dead_code()
    model.recompile()
