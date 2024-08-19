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
from torch.ao.quantization.pt2e.utils import fold_bn_weights_into_conv_node
from torch.quantization.fake_quantize import FakeQuantize

import nncf
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.transformations.commands import TargetType
from nncf.experimental.torch.fx.node_utils import get_graph_node_by_name
from nncf.experimental.torch.fx.node_utils import get_tensor_constant_from_node
from nncf.torch.graph.transformations.commands import PTTargetPoint

TransformationFNType = Callable[[torch.fx.GraphModule], None]


def _set_new_node_meta(new_node: torch.fx.Node, prev_node: torch.fx.Node, target_module: torch.nn.Module):
    """
    Sets correct meta \"val\" value to the new node.

    :param new_node: The new node.
    :param prev_node: Input node of the new node.
        New node expected to have only one input node.
    :param target_module: Module which is being called by the new node.
    """
    val = prev_node.meta["val"]
    val = val if isinstance(val, tuple) else (val,)
    retval = []
    for t in val:
        retval.append(torch.ones(t.shape))

    with torch.no_grad():
        new_node.meta["val"] = target_module(*val)


def module_insertion_transformation_builder(
    module_to_insert: torch.nn.Module, target_points: List[PTTargetPoint], target_module_name: str
) -> TransformationFNType:
    """
    Returns transformation which inserts given module to a target model
    and calls given module after each target points replacing inputs/outputs
    of the target node.

    :param module_to_insert: Given torch.nn.Module to insert.
    :param target_points: Target points to insert the target module.
    :param target_module_name: Target model attribute name for the module_to_insert.
    :returns: Transformation which which inserts given module to a target model
        and calls given module after each target points.
    """

    def module_insertion_transformation(model: torch.fx.GraphModule):
        module_attr_name = _set_module_to_the_graph_module(model, module_to_insert, target_module_name)
        # Insert call_module nodes to the model
        graph = model.graph
        for idx, target_point in enumerate(target_points):
            new_node = _insert_call_module(graph, target_point, module_attr_name, f"{module_attr_name}_{idx}")
            target_node = get_graph_node_by_name(graph, target_point.target_node_name)

            if target_point.target_type == TargetType.OPERATOR_POST_HOOK:
                _set_new_node_meta(new_node, target_node, module_to_insert)
                with graph.inserting_after(target_node):
                    for user in target_node.users:
                        if user is new_node:
                            continue
                        user.replace_input_with(target_node, new_node)

            else:
                prev_node = target_node.args[target_point.input_port_id]
                _set_new_node_meta(new_node, prev_node, module_to_insert)
                target_node.replace_input_with(prev_node, new_node)

    return module_insertion_transformation


def leaf_module_insertion_transformation_builder(
    module_to_insert: torch.nn.Module, target_points: List[PTTargetPoint], target_module_name: str
) -> TransformationFNType:
    """
    Returns transformation which inserts given module to a target model
    and calls given module after each target points.

    :param module_to_insert: Given torch.nn.Module to insert.
    :param target_points: Target points to insert the target module.
    :param target_module_name: Target model attribute name for the module_to_insert.
    :returns: Transformation which which inserts given module to a target model
        and calls given module after each target points.
    """

    def leaf_module_insertion_transformation(model: torch.fx.GraphModule):
        module_attr_name = _set_module_to_the_graph_module(model, module_to_insert, target_module_name)
        # Insert call_module nodes to the model
        graph = model.graph
        for idx, target_point in enumerate(target_points):
            _insert_call_module(graph, target_point, module_attr_name, f"{module_attr_name}_{idx}")

    return leaf_module_insertion_transformation


def bias_update_transformation_builder(node: NNCFNode, value: torch.Tensor) -> TransformationFNType:
    """
    Return transformation which updates constant of the given node with bias to the given value.

    :param node: Node with bias which requires bias constant update.
    :param value: New value to use as the bias constant.
    :return: Transformation which updates constant of the given node with bias to the given value.
    """

    def bias_update_transformation(model: torch.fx.GraphModule):
        graph = model.graph
        target_node_name = node.node_name
        graph_node = get_graph_node_by_name(graph, target_node_name)
        if len(graph_node.users) != 1:
            raise nncf.InternalError(f"Node with bias have {len(graph_node.users)} users, 1 expected.")

        bias_node = next(iter(graph_node.users))
        constant_update_fn(model, bias_node, value, input_port_id=1)

    return bias_update_transformation


def constant_update_transformation_builder(node: NNCFNode, value: torch.Tensor) -> TransformationFNType:
    """
    Return transformation which updates constant of the given node to the given value.

    :param node: Node which requires bias constant update.
    :param value: New value to use as the node constant.
    :return: Transformation which updates constant of the given node to the given value.
    """

    def constant_update_transformation(model: torch.fx.GraphModule):
        constant_update_fn(model, get_graph_node_by_name(model.graph, node.node_name), value, input_port_id=1)

    return constant_update_transformation


def constant_update_fn(model: torch.fx.GraphModule, node: torch.fx.Node, value: torch.Tensor, input_port_id: int = 1):
    """
    Updates constant of given node on the given input port id with given value.

    :param model: Target torch GraphModule.
    :param node: Given graph node.
    :param value: New value to use as the node constant.
    :param input_port_id: Target constant input port id.
    """
    graph = model.graph
    with graph.inserting_before(node):
        new_constant = create_getattr_from_value(model, graph, node.name + "_updated_constant", value)

    args = list(node.args)
    # A bias node suppose to have constant on the second input port.
    if args[input_port_id].op != "get_attr":
        raise nncf.InternalError(
            f"Constant on input port {input_port_id} for {node} is expected,"
            f" but node {args[input_port_id]} is present."
        )
    args[input_port_id] = new_constant
    node.args = tuple(args)
    graph.eliminate_dead_code()


def qdq_insertion_transformation_builder(
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

    def qdq_insertion_transformation(model: torch.fx.GraphModule):
        if any(tp.target_type != TargetType.OPERATION_WITH_WEIGHTS for tp in target_points) and len(target_points) > 1:
            raise RuntimeError(
                "Insertion of shared qdq pair for the weights is not supported."
                " Please use non shared qdq pairs for the weights quantization."
            )
        for target_point in target_points:
            insert_one_qdq(model, target_point, quantizer)

    return qdq_insertion_transformation


def insert_one_qdq(model: torch.fx.GraphModule, target_point: PTTargetPoint, quantizer: FakeQuantize):
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
    target_node = get_graph_node_by_name(graph, target_point.target_node_name)
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

    input_node = get_input_node(target_point, target_node)
    quantize_op_inputs[0] = input_node

    ctx_manager = get_ctx_manager(graph, target_point)
    with ctx_manager(target_node):
        quantized_node = graph.create_node(node_type, quantize_op, tuple(quantize_op_inputs), {})

    # use the same qparams from quantize op
    dq_inputs = [quantized_node] + quantize_op_inputs[1:]
    if target_point.target_type == TargetType.OPERATOR_POST_HOOK:
        user_dq_nodes = []
        with graph.inserting_after(quantized_node):
            for user in target_node.users:
                if user is quantized_node:
                    continue
                user_dq_nodes.append((user, graph.call_function(dequantize_op, tuple(dq_inputs), {})))

        for user, dq_node in user_dq_nodes:
            user.replace_input_with(target_node, dq_node)
    elif target_point.target_type in [TargetType.OPERATOR_PRE_HOOK, TargetType.OPERATION_WITH_WEIGHTS]:
        with graph.inserting_after(quantized_node):
            dq_node = graph.call_function(dequantize_op, tuple(dq_inputs), {})

        args = list(target_node.args)
        args[target_point.input_port_id] = dq_node
        target_node.args = tuple(args)
    else:
        raise nncf.InternalError(f"Unexpected target type: {target_point.target_type}")


def _insert_call_module(
    graph: torch.fx.Graph, target_point: PTTargetPoint, module_attr_name: str, graph_node_name: str
):
    """
    Inserts module call node to the graph after the target node.

    :param graph: Graph to insert module call node.
    :param target_node: Target node, module call node is being iserted just after the target node.
    :param module_attr_name: The name of the graph attribute which keeps the target module.
    :param graph_node_name: Target name for module call node.
    :return: Inserted module call node.
    """
    target_node = get_graph_node_by_name(graph, target_point.target_node_name)
    input_node = get_input_node(target_point, target_node)
    ctx_manager = get_ctx_manager(graph, target_point)
    with ctx_manager(target_node):
        return graph.create_node("call_module", module_attr_name, (input_node,), {}, name=graph_node_name)


def get_input_node(target_point: PTTargetPoint, target_node: torch.fx.Node) -> torch.fx.Node:
    """
    Returns an input node according to the given target point.

    :param target_point: Given target point.
    :param target_node: The target node of the given target point.
    :return: An input node according to the given target point.
    """
    target_type = target_point.target_type
    if target_type not in [
        TargetType.OPERATOR_PRE_HOOK,
        TargetType.OPERATOR_POST_HOOK,
        TargetType.OPERATION_WITH_WEIGHTS,
    ]:
        raise nncf.InternalError(f"Unexpected target type: {target_type}")
    if target_type == TargetType.OPERATOR_POST_HOOK:
        return target_node
    return target_node.args[target_point.input_port_id]


def get_ctx_manager(graph: torch.fx.Graph, target_point: PTTargetPoint) -> Callable:
    """
    Return insertion context manager according to the given target point.
    An insertion context manager sets the point at which create_node and
    companion methods will insert into the torch.fx.Graph.

    :param graph: torch.fx.Graph instance.
    :param target_point: Given target point.
    :return: Insertion context manager according to the given target point.
    """
    if target_point.target_type not in [
        TargetType.OPERATOR_PRE_HOOK,
        TargetType.OPERATOR_POST_HOOK,
        TargetType.OPERATION_WITH_WEIGHTS,
    ]:
        raise nncf.InternalError(f"Unexpected target type: {target_point.target_type}")

    if target_point.target_type == TargetType.OPERATOR_POST_HOOK:
        return graph.inserting_after
    return graph.inserting_before


def _set_module_to_the_graph_module(
    model: torch.fx.GraphModule,
    module_to_insert: torch.nn.Module,
    module_name_in_model: str,
) -> str:
    """
    Sets given module to the given torch.fx.GraphModule with unique name.

    :param graph: Target torch.fx.Graph.
    :param module_to_insert: Module to insert to the target graph.
    :param module_name_in_model: Target model attribute name for the module_to_insert.
    :return: A graph module attribute name which keep given module.
    """
    assert not hasattr(model, module_name_in_model)
    setattr(model, module_name_in_model, module_to_insert)
    return module_name_in_model


def _is_supported_batch_norm_for_training(node: torch.fx.Node):
    """
    Return True if the given node refers to an aten batch norm op QAT supports.
    """
    supported_ops = [
        torch.ops.aten._native_batch_norm_legit.default,
        torch.ops.aten.cudnn_batch_norm.default,
        torch.ops.aten.miopen_batch_norm.default,
    ]
    return node.target in supported_ops


def _is_bn_node(node: torch.fx.Node):
    return (
        _is_supported_batch_norm_for_training(node)
        or node.target == torch.ops.aten._native_batch_norm_legit_no_training.default
    )


def fuse_conv_bn(model: torch.fx.GraphModule) -> None:
    """
    BatchNorm operations have 3 output ports, to make it easier for algorithms to work with
    the target graph BatchNorm operations are being fused

    :param model: Model to apply transformations to.
    """
    has_bn = any(_is_bn_node(node) for node in model.graph.nodes)
    if not has_bn:
        return

    for node in model.graph.nodes:
        if node.op != "call_function" or not _is_bn_node(node):
            continue
        bn_node = node

        node = bn_node.args[0]
        if not _is_conv(node):
            continue
        conv_node = node
        conv_weight_node = conv_node.args[1]
        conv_bias_node = conv_node.args[2] if len(conv_node.args) > 2 else None
        fold_bn_weights_into_conv_node(conv_node, conv_weight_node, conv_bias_node, bn_node, model)

    model.graph.eliminate_dead_code()
    model.recompile()


def apply_quantization_transformations(model: torch.fx.GraphModule) -> None:
    """
    Applies quantization transformations to the model.
    :param model: Model to apply transformations to.
    """
    # BatchNorm operations have 3 output ports,
    # to make it easier for algorithms to work
    # with the target graph BatchNorm operations
    # are being fused
    fuse_conv_bn(model)
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
        linear_bias_value = get_tensor_constant_from_node(linear_bias_node, model)
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
        dims = len(get_tensor_constant_from_node(conv_node.args[1], model).shape)
        conv_bias_node = conv_node.args[2]
        conv_bias_value = get_tensor_constant_from_node(conv_bias_node, model)
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
    :param is_target_node: Predicate to specify nodes which should be merged with the bias
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
        bias_value = get_tensor_constant_from_node(const_node, model).squeeze()
        with model.graph.inserting_before(conv_node):
            new_bias_node = create_getattr_from_value(model, model.graph, const_node.name + "_", bias_value)
        args = list(conv_node.args)
        args[2] = new_bias_node
        conv_node.args = tuple(args)
        for user in list(bias_node.users):
            user.replace_input_with(bias_node, conv_node)

    model.graph.eliminate_dead_code()
    model.recompile()
