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

from typing import Callable, Optional

import torch
import torch.fx
from torch.ao.quantization.fx.utils import create_getattr_from_value

from nncf.quantization.fake_quantize import FakeQuantizeParameters
from nncf.torch.quantization.layers import PTQuantizerSpec


def quantizer_insertion_tranformation_builder(
    qspec: PTQuantizerSpec, fq_params: FakeQuantizeParameters, axis: int, eps=1e-5
):
    # signed = bool(torch.any(fq_params.input_low.data < 0))
    # Subtract eps from the scale to make quantizer parameters equal to
    # original parameters on the forward call.
    scale = (fq_params.input_high.data - eps).reshape(qspec.scale_shape)

    def quantizer_insertion_tranformation(model: torch.fx.GraphModule, node: torch.fx.Node):
        # 1. extract information for inserting q/dq node from activation_post_process
        node_type = "call_function"
        quantize_op: Optional[Callable] = None
        # scale, zero_point = activation_post_process.calculate_qparams()  # type: ignore[attr-defined, operator]
        if qspec.per_channel:
            quantize_op = torch.ops.quantized_decomposed.quantize_per_channel.default
            dequantize_op = torch.ops.quantized_decomposed.dequantize_per_channel.default
        else:
            quantize_op = torch.ops.quantized_decomposed.quantize_per_tensor.default
            dequantize_op = torch.ops.quantized_decomposed.dequantize_per_tensor.default
        # TODO: map FakeQuantizePramaeters to qparams for quantize/dequantize
        qparams = {
            "_scale_": scale,
            "_zero_point_": 0,
            "_axis_": axis,
            "_quant_min_": 0,
            "_quant_max_": 2**qspec.num_bits - 1,
            "_dtype_": torch.int8,
        }
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
                    qparam_node = create_getattr_from_value(model, graph, node.name + key, value_or_node)
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

    return quantizer_insertion_tranformation
