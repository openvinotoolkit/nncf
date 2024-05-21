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


def insert_qdq_to_model(model: torch.fx.GraphModule, qsetup):
    for node in list(model.graph.nodes):
        if node.name not in qsetup:
            continue
        # 1. extract information for inserting q/dq node from activation_post_process
        for idx, (place, params) in enumerate(qsetup[node.name].items()):
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
            with graph.inserting_before(node):
                # TODO: use metatype to get correct input_port_id
                source_node = node.args[1] if place == "weights" else node.args[0]
                quantize_op_inputs = [source_node]
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

                quantized_node = graph.create_node(node_type, quantize_op, tuple(quantize_op_inputs), {})
                # use the same qparams from quantize op
                dq_inputs = [quantized_node] + quantize_op_inputs[1:]
                dequantized_node = graph.call_function(dequantize_op, tuple(dq_inputs), {})
            orig_users = list(source_node.users.keys())
            for user_node in orig_users:
                if user_node is quantized_node:
                    continue
                user_node.replace_input_with(source_node, dequantized_node)
            # node.replace_all_uses_with(dequantized_node)
            # graph.erase_node(node)
