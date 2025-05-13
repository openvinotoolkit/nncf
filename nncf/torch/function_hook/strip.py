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
from nncf.parameters import StripFormat
from nncf.torch.function_hook.hook_storage import decode_hook_name
from nncf.torch.function_hook.nncf_graph.nncf_graph_builder import build_nncf_graph
from nncf.torch.function_hook.wrapper import get_hook_storage
from nncf.torch.model_graph_manager import get_const_data
from nncf.torch.model_graph_manager import get_const_node
from nncf.torch.model_graph_manager import get_module_by_name
from nncf.torch.model_graph_manager import split_const_name
from nncf.torch.quantization.layers import AsymmetricQuantizer
from nncf.torch.quantization.layers import BaseQuantizer
from nncf.torch.quantization.layers import SymmetricQuantizer
from nncf.torch.quantization.strip import asym_fq_to_decompressor
from nncf.torch.quantization.strip import convert_to_torch_fakequantizer
from nncf.torch.quantization.strip import sym_fq_to_decompressor

TModel = TypeVar("TModel", bound=nn.Module)


def strip_quantized_model(model: TModel, example_input: Any, strip_format: StripFormat = StripFormat.NATIVE) -> TModel:
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

    if strip_format == StripFormat.NATIVE:
        model = replace_quantizer_to_torch_native_module(model, graph)
    elif strip_format == StripFormat.DQ:
        model = replace_quantizer_to_compressed_weight_with_decompressor(model, graph)
    elif strip_format == StripFormat.IN_PLACE:
        model = apply_compression_in_place(model, graph)
    else:
        msg = f"Unsupported strip format: {strip_format}"
        raise nncf.ParameterNotSupportedError(msg)
    return model


def replace_quantizer_to_torch_native_module(model: TModel, graph: NNCFGraph) -> TModel:
    """
    Replaces NNCF's BaseQuantizer modules with PyTorch's FakeQuantize ones.

    :param model: Target model.
    :param graph: The model graph.
    :return: The modified NNCF network.
    """
    hook_storage = get_hook_storage(model)
    for name, module in hook_storage.named_hooks():
        if not isinstance(module, BaseQuantizer):
            continue

        # Replace nncf fake quantizer to torch fake quantizer
        new_fq = convert_to_torch_fakequantizer(module)
        hook_storage.set_submodule(name, new_fq)

        # Update the weights of the module
        hook_type, op_name, port_id = decode_hook_name(name)
        if hook_type == "pre_hooks" and (module.is_half_range or module.narrow_range):
            op_node = graph.get_node_by_name(op_name)
            const_node = get_const_node(op_node, port_id, graph)
            if const_node is None:
                continue
            data = get_const_data(const_node, model)
            with torch.no_grad():
                # Half range and narrow_range require to clamp weights of the module
                # Note: Half range and narrow_range are used only for weights.
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


def replace_quantizer_to_compressed_weight_with_decompressor(model: TModel, graph: NNCFGraph) -> TModel:
    """
    Performs transformation from fake quantize format (FQ) to dequantization one (DQ):
        (weights + FQ) -> (compressed_weights + DQ)

    :param model: Compressed model
    :param graph: The model graph.
    :return: The modified NNCF network.
    """
    hook_storage = get_hook_storage(model)

    for name, module in hook_storage.named_hooks():
        if not isinstance(module, (SymmetricQuantizer, AsymmetricQuantizer)):
            continue
        msg = ""
        if module._qspec.half_range or module._qspec.narrow_range:
            msg += "Unexpected parameters of quantizers on strip: half_range and narrow_range should be False.\n"
        if module.num_bits not in [4, 8]:
            msg += f"Unsupported number of bits {module.num_bits} for the quantizer {module}.\n"
        if msg:
            raise nncf.ValidationError(msg)

        _, op_name, _ = decode_hook_name(name)
        weight_node = graph.get_node_by_name(op_name)

        if weight_node is None:
            msg = "FQ is not assigned to weight. Strip to DQ format is not supported for FQ on activation."
            raise nncf.UnsupportedModelError(msg)

        if not isinstance(weight_node.layer_attributes, ConstantLayerAttributes):
            msg = f"Unexpected layer attributes type {type(weight_node.layer_attributes)}"
            raise nncf.InternalError(msg)

        weight = get_const_data(weight_node, model)

        convert_fn = asym_fq_to_decompressor if isinstance(module, AsymmetricQuantizer) else sym_fq_to_decompressor
        decompressor, q_weight = convert_fn(module, weight)  # type: ignore[operator]
        packed_tensor = decompressor.pack_weight(q_weight)

        module_name, weight_attr_name = split_const_name(weight_node.layer_attributes.name)
        module = get_module_by_name(module_name, model)
        weight_param = getattr(module, weight_attr_name)

        weight_param.requires_grad = False
        weight_param.data = packed_tensor

        hook_storage.set_submodule(name, decompressor)
    return model


def apply_compression_in_place(model: TModel, graph: NNCFGraph) -> TModel:
    """
    Applies fake quantizers in-place to the weights:
        (weights + FQ) -> (fake quantized weights)

    :param model: Compressed model
    :param graph: The model graph.
    :return: The modified NNCF network.
    """
    hook_storage = get_hook_storage(model)

    hooks_to_delete = []
    for name, hook in hook_storage.named_hooks():
        if not isinstance(hook, (SymmetricQuantizer, AsymmetricQuantizer)):
            continue
        _, op_name, _ = decode_hook_name(name)
        weight_node = graph.get_node_by_name(op_name)

        if weight_node is None:
            msg = "FQ is not assigned to weight. In-place strip is not supported for FQ on activation."
            raise nncf.UnsupportedModelError(msg)

        if not isinstance(weight_node.layer_attributes, ConstantLayerAttributes):
            msg = f"Unexpected layer attributes type {type(weight_node.layer_attributes)}"
            raise nncf.InternalError(msg)

        weight = get_const_data(weight_node, model)
        fq_weight = hook.quantize(weight)

        module_name, weight_attr_name = split_const_name(weight_node.layer_attributes.name)
        module = get_module_by_name(module_name, model)
        weight_param = getattr(module, weight_attr_name)

        weight_param.requires_grad = False
        weight_param.data = fq_weight

        hooks_to_delete.append(name)

    for hook_name in hooks_to_delete:
        hook_storage.delete_hook(hook_name)

    return model
