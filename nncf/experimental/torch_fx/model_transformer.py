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

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.fx
from torch.ao.quantization.fx.utils import create_getattr_from_value
from torch.ao.quantization.pt2e.duplicate_dq_pass import DuplicateDQPass
from torch.ao.quantization.pt2e.port_metadata_pass import PortNodeMetaForQDQ
from torch.ao.quantization.pt2e.qat_utils import _fold_conv_bn_qat
from torch.ao.quantization.pt2e.utils import _disallow_eval_train
from torch.ao.quantization.pt2e.utils import _fuse_conv_bn_
from torch.fx import GraphModule
from torch.fx.passes.infra.pass_manager import PassManager


@dataclass
class QPARAMSPerTensor:
    scale: float
    zero_point: int
    quant_min: int
    quant_max: int
    dtype: torch.dtype


@dataclass
class QPARAMPerChannel:
    scales: torch.Tensor
    zero_points: Optional[torch.Tensor]
    axis: int
    quant_min: int
    quant_max: int
    dtype: torch.dtype


def insert_qdq_to_model(model: torch.fx.GraphModule, qsetup) -> torch.fx.GraphModule:
    # from prepare
    _fuse_conv_bn_(model)

    # from convert
    original_graph_meta = model.meta
    _insert_qdq_to_model(model, qsetup)

    # Magic. Without this call compiled model
    # is not preformant
    model = GraphModule(model, model.graph)

    model = _fold_conv_bn_qat(model)
    pm = PassManager([DuplicateDQPass()])

    model = pm(model).graph_module
    pm = PassManager([PortNodeMetaForQDQ()])
    model = pm(model).graph_module

    model.meta.update(original_graph_meta)
    model = _disallow_eval_train(model)
    return model


def _insert_qdq_to_model(model: torch.fx.GraphModule, qsetup) -> torch.fx.GraphModule:
    # qsetup_after_node = dict()
    # for node in list(model.graph.nodes):
    #    for node_name, setup in qsetup.items():
    #        if node.name != node_name:
    #            continue
    #        qsetup_after_node[node.all_input_nodes[0].name] = setup["activations"]
    #        if "weights" in setup:
    #            qsetup_after_node[node.all_input_nodes[1].name] = setup["weights"]
    #        # Case for nodes with two activations
    #        #if len(node.all_input_nodes) > 1 and node.all_input_nodes[1].op not in ["placeholder", "get_attr"]:
    #        #    qsetup_after_node[node.all_input_nodes[1].name] = setup["activations"]
    #        break
    # qsetup = qsetup_after_node

    for idx, node in enumerate(list(model.graph.nodes)):
        if node.name not in qsetup:
            continue
        # 1. extract information for inserting q/dq node from activation_post_process
        params = qsetup[node.name]
        node_type = "call_function"
        quantize_op: Optional[Callable] = None
        # scale, zero_point = activation_post_process.calculate_qparams()  # type: ignore[attr-defined, operator]
        if isinstance(params, QPARAMPerChannel):
            quantize_op = torch.ops.quantized_decomposed.quantize_per_channel.default
            dequantize_op = torch.ops.quantized_decomposed.dequantize_per_channel.default
            qparams = {
                "_scale_": params.scales,
                "_zero_point_": params.zero_points,
                "_axis_": params.axis,
                "_quant_min_": params.quant_min,
                "_quant_max_": params.quant_max,
                "_dtype_": params.dtype,
            }
        elif isinstance(params, QPARAMSPerTensor):
            quantize_op = torch.ops.quantized_decomposed.quantize_per_tensor.default
            dequantize_op = torch.ops.quantized_decomposed.dequantize_per_tensor.default
            qparams = {
                "_scale_": params.scale,
                "_zero_point_": params.zero_point,
                "_quant_min_": params.quant_min,
                "_quant_max_": params.quant_max,
                "_dtype_": params.dtype,
            }

        else:
            raise RuntimeError(f"params {params} are unknown")
        # 2. replace activation_post_process node with quantize and dequantize
        graph = model.graph

        # TODO: use metatype to get correct input_port_id
        # Do not quantize already quantized nodes
        # inserting_before handle only order in the graph generated code.
        # so, inserting quantize-dequantize and all constant nodes before the usage of the nodes
        with graph.inserting_before(node):
            quantize_op_inputs = [node]
            for key, value_or_node in qparams.items():
                # TODO: we can add the information of whether a value needs to
                # be registered as an attribute in qparams dict itself
                if key in ["_scale_", "_zero_point_"] and (not isinstance(value_or_node, (float, int))):
                    # For scale and zero_point values we register them as buffers in the root module.
                    # However, note that when the values are not tensors, as in the case of
                    # per_tensor quantization, they will be treated as literals.
                    # However, registering them as a node seems to cause issue with dynamo
                    # tracing where it may consider tensor overload as opposed to default.
                    # With extra check of scale and zero_point being scalar, it makes
                    # sure that the default overload can be used.
                    # TODO: maybe need more complex attr name here
                    qparam_node = create_getattr_from_value(model, graph, str(idx) + key, value_or_node)
                    quantize_op_inputs.append(qparam_node)
                else:
                    # for qparams that are not scale/zero_point (like axis, dtype) we store
                    # them as literals in the graph.
                    quantize_op_inputs.append(value_or_node)

        with graph.inserting_after(node):
            quantized_node = graph.create_node(node_type, quantize_op, tuple(quantize_op_inputs), {})
            # use the same qparams from quantize op
        dq_inputs = [quantized_node] + quantize_op_inputs[1:]
        user_dq_nodes = []
        with graph.inserting_after(quantized_node):
            for user in node.users:
                if user is quantized_node:
                    continue
                user_dq_nodes.append((user, graph.call_function(dequantize_op, tuple(dq_inputs), {})))

        for user, dq_node in user_dq_nodes:
            user.replace_input_with(node, dq_node)

        # node.replace_all_uses_with(dequantized_node)
        # graph.erase_node(node)
        from torch.fx.passes.graph_drawer import FxGraphDrawer

        g = FxGraphDrawer(model, "model_after_qdq_insertion")
        g.get_dot_graph().write_svg("model_after_qdq_insertion.svg")
