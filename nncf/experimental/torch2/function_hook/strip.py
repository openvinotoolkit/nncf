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


from typing import Any, TypeVar

import torch
from torch import nn

import nncf
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.layer_attributes import ConstantLayerAttributes
from nncf.experimental.torch2.function_hook.nncf_graph.nncf_graph_builder import build_nncf_graph
from nncf.experimental.torch2.function_hook.wrapper import get_hook_storage
from nncf.parameters import StripFormat
from nncf.torch.model_graph_manager import get_const_data
from nncf.torch.model_graph_manager import get_const_node
from nncf.torch.model_graph_manager import get_module_by_name
from nncf.torch.model_graph_manager import split_const_name
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.quantization.layers import BaseQuantizer
from nncf.torch.quantization.strip import convert_to_torch_fakequantizer

TModel = TypeVar("TModel", bound=nn.Module)


def strip_quantized_model(
    model: NNCFNetwork, example_input: Any, strip_format: StripFormat = StripFormat.NATIVE
) -> NNCFNetwork:
    """
    Removes auxiliary layers and operations added during the quantization process,
    resulting in a clean quantized model ready for deployment. The functionality of the model object is still preserved
    as a compressed model.

    :param model: Compressed model.
    :param example_input: An example input tensor to be used for tracing the model.
    :param strip_format: Describes the format in which model is saved after strip.
    :return: The modified NNCF network.
    """
    graph = build_nncf_graph(model, example_input)

    if strip_format == StripFormat.DQ:
        pass
        # model = replace_with_decompressors(model)
    elif strip_format == StripFormat.NATIVE:
        model = replace_quantizer_to_torch_native_module(model, graph)
    else:
        msg = f"Unsupported strip format: {strip_format}"
        raise nncf.ParameterNotSupportedError(msg)
    return model


def replace_quantizer_to_torch_native_module(model: TModel, graph: NNCFGraph) -> TModel:
    """
    Replace NNCF quantizer modules to PyTorch FakeQuantizer module and remove unused quantizer operators.

    :param model: Target model.
    :return: The modified NNCF network.
    """
    hook_storage = get_hook_storage(model)
    for name, module in hook_storage.named_hooks():
        if isinstance(module, BaseQuantizer):
            new_fq = convert_to_torch_fakequantizer(module)
            hook_storage.set_submodule(name, new_fq)

            if name.startswith("pre_hooks") and (module.is_half_range or module.narrow_range):
                key = name.split(".")[1]
                node_name = "__".join(key.split("__")[:-1])
                port_id = int(key.split("__")[-1])
                op_node = graph.get_node_by_name(node_name)
                const_node = get_const_node(op_node, port_id, graph)
                if const_node is None:
                    continue
                data = get_const_data(const_node, model)
                with torch.no_grad():
                    # Half range and narrow_range require to clamp weights of module
                    # Note: Half range and narrow_range used only for weight.
                    input_low, input_high = module.get_input_low_input_high()  # type: ignore

                    data = torch.min(torch.max(data, input_low), input_high)
                    data = module.quantize(data, execute_traced_op_as_identity=False)

                if not isinstance(const_node.layer_attributes, ConstantLayerAttributes):
                    msg = f"Unexpected layer attributes type {type(const_node.layer_attributes)}"
                    raise nncf.InternalError(msg)

                module_name, weight_attr_name = split_const_name(const_node.layer_attributes.name)
                module = get_module_by_name(module_name, model)
                weight_param = getattr(module, weight_attr_name)
                weight_param.data = data
    return model
