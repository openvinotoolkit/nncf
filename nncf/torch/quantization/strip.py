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


from typing import List

import numpy as np
import torch
from torch.quantization.fake_quantize import FakeQuantize

import nncf
from nncf.common.graph.transformations.commands import Command
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.experimental.common.check_feature import is_experimental_torch_tracing_enabled
from nncf.experimental.torch2.commands import PT2InsertionCommand
from nncf.torch.dynamic_graph.scope import Scope
from nncf.torch.graph.transformations.commands import ExtraCompressionModuleType
from nncf.torch.graph.transformations.commands import PTSharedFnInsertionCommand
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.model_graph_manager import get_const_node
from nncf.torch.model_graph_manager import get_module_by_name
from nncf.torch.model_graph_manager import split_const_name
from nncf.torch.model_transformer import PTModelTransformer
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.quantization.layers import AsymmetricLoraQuantizer
from nncf.torch.quantization.layers import AsymmetricQuantizer
from nncf.torch.quantization.layers import BaseQuantizer
from nncf.torch.quantization.layers import INT4AsymmetricWeightsDecompressor
from nncf.torch.quantization.layers import INT4SymmetricWeightsDecompressor
from nncf.torch.quantization.layers import INT8AsymmetricWeightsDecompressor
from nncf.torch.quantization.layers import INT8SymmetricWeightsDecompressor
from nncf.torch.quantization.layers import SymmetricLoraQuantizer
from nncf.torch.quantization.layers import SymmetricQuantizer
from nncf.torch.quantization.quantize_functions import TuneRange

SUPPORTED_NUM_BITS_FOR_STRIP_MODEL = [8]


def replace_quantizer_to_torch_native_module(model: NNCFNetwork) -> NNCFNetwork:
    """
    Replace NNCF quantizer modules to PyTorch FakeQuantizer module and remove unused quantizer operators.

    :param model: Target model.
    :return: The modified NNCF network.
    """
    compression_module_type = ExtraCompressionModuleType.EXTERNAL_QUANTIZER
    if model.nncf.is_compression_module_registered(compression_module_type):
        external_quantizers = model.nncf.get_compression_modules_by_type(compression_module_type)
        for key in external_quantizers:
            if external_quantizers[key].is_enabled_quantization():
                external_quantizers[key] = convert_to_torch_fakequantizer(external_quantizers[key])

    for node in model.nncf.get_original_graph().get_all_nodes():
        if node.node_type in ["nncf_model_input", "nncf_model_output"]:
            continue

        nncf_module = model.nncf.get_containing_module(node.node_name)

        if hasattr(nncf_module, "pre_ops"):
            for key in list(nncf_module.pre_ops.keys()):
                op = nncf_module.get_pre_op(key)
                if isinstance(op.op, BaseQuantizer) and op.op.is_enabled_quantization():
                    if op.op.is_half_range or op.op.narrow_range:
                        # Half range and narrow_range require to clamp weights of module
                        # Note: Half range and narrow_range used only for weight.
                        input_low, input_high = op.op.get_input_low_input_high()

                        data = nncf_module.weight.data
                        data = torch.min(torch.max(data, input_low), input_high)
                        data = op.op.quantize(data, execute_traced_op_as_identity=False)
                        nncf_module.weight.data = data
                    op.op = convert_to_torch_fakequantizer(op.op)

        if hasattr(nncf_module, "post_ops"):
            for key in list(nncf_module.post_ops.keys()):
                op = nncf_module.get_post_ops(key)
                if isinstance(op.op, BaseQuantizer) and op.op.is_enabled_quantization():
                    op.op = convert_to_torch_fakequantizer(op.op)

    return model


def convert_to_torch_fakequantizer(nncf_quantizer: BaseQuantizer) -> FakeQuantize:
    """
    Convert BaseQuantizer module to FakeQuantize.

    :param quantizer: NNCF Quantizer module.
    :return: Instance of FakeQuantize similar to the input quantizer.
    """
    # Call set_ranges in case the basic parameters impacting levels had changed
    nncf_quantizer.set_levels()

    if nncf_quantizer.num_bits not in SUPPORTED_NUM_BITS_FOR_STRIP_MODEL:
        msg = (
            "Converting nncf quantizer module to torch native only supports "
            f"for num_bits in {SUPPORTED_NUM_BITS_FOR_STRIP_MODEL}."
        )
        raise nncf.InternalError(msg)
    per_channel = nncf_quantizer.per_channel
    scale_shape = nncf_quantizer.scale_shape
    ch_axis = int(np.argmax(scale_shape))
    dtype = torch.qint8 if nncf_quantizer.level_low < 0 else torch.quint8

    if per_channel:
        observer = torch.ao.quantization.observer.PerChannelMinMaxObserver
    else:
        observer = torch.ao.quantization.observer.MinMaxObserver

    if isinstance(nncf_quantizer, SymmetricQuantizer):
        qscheme = torch.per_channel_symmetric if per_channel else torch.per_tensor_symmetric
    elif isinstance(nncf_quantizer, AsymmetricQuantizer):
        qscheme = torch.per_channel_affine if per_channel else torch.per_tensor_affine

    quant_min, quant_max, scale, zero_point = nncf_quantizer.get_parameters_for_torch_fq()

    fakequantizer = FakeQuantize(
        observer=observer,
        quant_max=quant_max,
        quant_min=quant_min,
        dtype=dtype,
        qscheme=qscheme,
        eps=nncf_quantizer.eps,
    )

    if not per_channel:
        scale = scale.squeeze()
        zero_point = zero_point.squeeze()

    fakequantizer.scale = scale
    fakequantizer.ch_axis = ch_axis
    fakequantizer.zero_point = zero_point

    # Disable observer to save parameters
    fakequantizer.disable_observer()

    return fakequantizer


def remove_disabled_quantizers(model: NNCFNetwork) -> NNCFNetwork:
    """
    Remove all unused quantizer operators from the model.

    :param model: Compressed model.
    :return: The modified NNCF network.
    """
    compression_module_type = ExtraCompressionModuleType.EXTERNAL_QUANTIZER
    if model.nncf.is_compression_module_registered(compression_module_type):
        external_quantizers = model.nncf.get_compression_modules_by_type(compression_module_type)
        for key in list(external_quantizers.keys()):
            op = external_quantizers[key]
            if isinstance(op, BaseQuantizer) and not op.is_enabled_quantization():
                external_quantizers.pop(key)

    if not model.nncf.replace_modules:
        return model

    for node in model.nncf.get_original_graph().get_all_nodes():
        if node.node_type in ["nncf_model_input", "nncf_model_output"]:
            continue

        nncf_module = model.nncf.get_containing_module(node.node_name)

        if hasattr(nncf_module, "pre_ops"):
            for key in list(nncf_module.pre_ops.keys()):
                op = nncf_module.get_pre_op(key)
                if isinstance(op, BaseQuantizer) and not op.is_enabled_quantization():
                    nncf_module.remove_pre_forward_operation(key)

        if hasattr(nncf_module, "post_ops"):
            for key in list(nncf_module.post_ops.keys()):
                op = nncf_module.post_ops(key)
                if isinstance(op, BaseQuantizer) and not op.is_enabled_quantization():
                    nncf_module.remove_post_forward_operation(key)

    return model


def strip_quantized_model(model: NNCFNetwork):
    """
    Returns the model with as much custom NNCF additions as possible removed
    while still preserving the functioning of the model object as a compressed model.

    :param model: Compressed model.
    :return: The modified NNCF network.
    """
    model_layout = model.nncf.transformation_layout()
    transformations = model_layout.transformations
    if any([type(q.fn) in [AsymmetricLoraQuantizer, SymmetricLoraQuantizer] for q in transformations]):
        model = replace_with_decompressors(model, transformations)
    else:
        model = replace_quantizer_to_torch_native_module(model)
        model = remove_disabled_quantizers(model)
    return model


def replace_with_decompressors(model: NNCFNetwork, transformations: List[Command]) -> NNCFNetwork:
    """
    Performs transformation from fake quantize format (FQ) to dequantization one (DQ).
    The former takes floating-point input, quantizes and dequantizes, and returns a floating-point value,
    while the latter takes a quantized integer representation, dequantizes it, and outputs a floating-point result.

    Mathematically, both methods lead to the same outcome, but due to differences in the order of operations and
    rounding errors, the actual results may differ. In particular, this error can occur for values
    that are located in the midpoint between two quantized values ("quants").

    The FQ format may round these values to one "quant", while the DQ format rounds them to another "quant".
    To avoid these issues, the compressed representation should be provided not by directly quantizing the input,
    but by quantizing a pre-processed, fake-quantized, floating-point representation.

    :param model: Compressed model with Decompressors.
    :return: The modified NNCF network.
    """
    transformation_layout = TransformationLayout()
    model = model.nncf.get_clean_shallow_copy()
    graph = model.nncf.get_graph()

    for command in transformations:
        quantizer = command.fn

        if len(command.target_points) > 1:
            msg = "Command contains more than one target point!"
            raise nncf.ValidationError(msg)

        tp = command.target_points[0]
        node_with_weight = graph.get_node_by_name(tp.target_node_name)
        weight_node = get_const_node(node_with_weight, tp.input_port_id, graph)

        module_name, weight_attr_name = split_const_name(weight_node.layer_attributes.name)
        module = get_module_by_name(module_name, model)
        original_weight = getattr(module, weight_attr_name)

        original_dtype = original_weight.dtype
        original_shape = original_weight.shape
        original_eps = torch.finfo(original_dtype).eps

        qdq_weight = quantizer.quantize(original_weight)
        if hasattr(quantizer, "_lspec"):
            # Special reshape for LoRA-grouped output
            qdq_weight = qdq_weight.reshape(quantizer._lspec.weight_shape)
        qdq_weight = qdq_weight.to(original_dtype)

        if isinstance(quantizer, AsymmetricQuantizer):
            input_range_safe = abs(quantizer.input_range) + quantizer.eps
            input_low, input_range = TuneRange.apply(quantizer.input_low, input_range_safe, quantizer.levels)

            integer_dtype = torch.uint8

            input_low = input_low.to(original_dtype)
            input_range = input_range.to(original_dtype)

            scale = input_range / quantizer.level_high
            scale = torch.where(torch.abs(scale) < original_eps, original_eps, scale)
            scale = scale.to(original_dtype)

            zero_point = quantizer.level_low - torch.round(input_low / scale)
            zero_point = torch.clip(zero_point, quantizer.level_low, quantizer.level_high)
            zero_point = zero_point.to(integer_dtype)

            q_weight = qdq_weight / scale
            q_weight = q_weight + zero_point
            q_weight = torch.round(q_weight)
            q_weight = torch.clip(q_weight, quantizer.level_low, quantizer.level_high)
            q_weight = q_weight.to(integer_dtype)

            if quantizer.num_bits == 8:
                decompressor = INT8AsymmetricWeightsDecompressor(
                    scale=scale, zero_point=zero_point, result_dtype=original_dtype
                )
            else:
                decompressor = INT4AsymmetricWeightsDecompressor(
                    scale=scale,
                    zero_point=zero_point,
                    compressed_weight_shape=q_weight.shape,
                    result_shape=original_shape,
                    result_dtype=original_dtype,
                )

        elif isinstance(quantizer, SymmetricQuantizer):
            integer_dtype = torch.int8

            scale = quantizer.scale / abs(quantizer.level_low)
            scale = torch.where(torch.abs(scale) < original_eps, original_eps, scale)
            scale = scale.to(original_dtype)

            q_weight = qdq_weight / scale
            q_weight = torch.round(q_weight)
            q_weight = torch.clip(q_weight, quantizer.level_low, quantizer.level_high)
            q_weight = q_weight.to(integer_dtype)

            if quantizer.num_bits == 8:
                decompressor = INT8SymmetricWeightsDecompressor(scale=scale, result_dtype=original_dtype)
            else:
                decompressor = INT4SymmetricWeightsDecompressor(
                    scale=scale,
                    compressed_weight_shape=q_weight.shape,
                    result_shape=original_shape,
                    result_dtype=original_dtype,
                )

        packed_tensor = decompressor.pack_weight(q_weight)

        # sets compressed tensor
        compressed_parameter = torch.nn.Parameter(packed_tensor, requires_grad=False)
        setattr(module, weight_attr_name, compressed_parameter)

        consumer_nodes = graph.get_next_nodes(weight_node)
        if len(consumer_nodes) > 1:
            for consumer_node in consumer_nodes:
                consumer_module = model.nncf.get_module_by_scope(Scope.from_str(consumer_node.layer_name))
                for name, param in consumer_module.named_parameters(recurse=False, remove_duplicate=False):
                    if id(param) == id(original_weight):
                        setattr(consumer_module, name, compressed_parameter)

        if is_experimental_torch_tracing_enabled():
            transformation_layout.register(
                PT2InsertionCommand(
                    [
                        PTTargetPoint(
                            TargetType.OPERATOR_POST_HOOK, target_node_name=weight_node.node_name.replace(".", ":")
                        )
                    ],
                    decompressor,
                )
            )
        else:
            decompressor_name = f"weights_decompressor_{weight_node.node_name.replace('.', '_')}"
            transformation_layout.register(
                PTSharedFnInsertionCommand(
                    [PTTargetPoint(TargetType.OPERATOR_POST_HOOK, target_node_name=weight_node.node_name)],
                    decompressor,
                    decompressor_name,
                )
            )

    return PTModelTransformer(model).transform(transformation_layout)
