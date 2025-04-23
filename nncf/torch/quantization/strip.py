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


from typing import Union

import numpy as np
import torch
from torch.quantization.fake_quantize import FakeQuantize

import nncf
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.parameters import StripFormat
from nncf.torch.graph.transformations.commands import ExtraCompressionModuleType
from nncf.torch.graph.transformations.commands import PTSharedFnInsertionCommand
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.model_graph_manager import get_const_data
from nncf.torch.model_graph_manager import get_module_by_name
from nncf.torch.model_graph_manager import split_const_name
from nncf.torch.model_transformer import PTModelTransformer
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.quantization.layers import AsymmetricQuantizer
from nncf.torch.quantization.layers import BaseQuantizer
from nncf.torch.quantization.layers import INT4AsymmetricWeightsDecompressor
from nncf.torch.quantization.layers import INT4SymmetricWeightsDecompressor
from nncf.torch.quantization.layers import INT8AsymmetricWeightsDecompressor
from nncf.torch.quantization.layers import INT8SymmetricWeightsDecompressor
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


def strip_quantized_model(model: NNCFNetwork, strip_format: StripFormat = StripFormat.NATIVE):
    """
    Removes auxiliary layers and operations added during the quantization process,
    resulting in a clean quantized model ready for deployment. The functionality of the model object is still preserved
    as a compressed model.

    :param model: Compressed model.
    :param strip format: Describes the format in which model is saved after strip.
    :return: The modified NNCF network.
    """
    if strip_format == StripFormat.DQ:
        model = replace_with_decompressors(model)
    elif strip_format == StripFormat.NATIVE:
        model = replace_quantizer_to_torch_native_module(model)
        model = remove_disabled_quantizers(model)
    else:
        msg = f"Unsupported strip format: {strip_format}"
        raise nncf.ParameterNotSupportedError(msg)
    return model


def asym_fq_to_decompressor(
    quantizer: AsymmetricQuantizer, weight: torch.Tensor
) -> tuple[Union[INT8AsymmetricWeightsDecompressor, INT4AsymmetricWeightsDecompressor], torch.Tensor]:
    """
    Converts an asymmetric quantizer and original weight tensor to a decompressor and quantized weight tensor.

    :param quantizer: The asymmetric quantizer instance.
    :param weight: The weight tensor to be compressed and used in decompressor.
    :return: The decompressor and quantized weight corresponding to the given quantizer and original weight.
    """
    assert isinstance(quantizer, AsymmetricQuantizer)
    weight_dtype = weight.dtype
    weight_shape = weight.shape
    float_dtype = torch.float32
    integer_dtype = torch.uint8

    eps = torch.finfo(float_dtype).eps
    qdq_weight = quantizer.quantize(weight)
    if hasattr(quantizer, "_lspec"):
        # Reshape for group-wise quantization, implemented for classes with lora spec only
        qdq_weight = qdq_weight.reshape(quantizer._lspec.weight_shape)
    qdq_weight = qdq_weight.to(float_dtype)

    input_range_safe = abs(quantizer.input_range) + quantizer.eps
    input_low, input_range = TuneRange.apply(quantizer.input_low, input_range_safe, quantizer.levels)

    input_low = input_low.to(float_dtype)
    input_range = input_range.to(float_dtype)

    scale = input_range / quantizer.level_high
    scale = torch.where(torch.abs(scale) < eps, eps, scale)
    scale = scale.to(float_dtype)

    zero_point = quantizer.level_low - torch.round(input_low / scale)
    zero_point = torch.clip(zero_point, quantizer.level_low, quantizer.level_high)
    zero_point = zero_point.to(float_dtype)

    q_weight = qdq_weight / scale
    q_weight = q_weight + zero_point
    q_weight = torch.round(q_weight)
    q_weight = torch.clip(q_weight, quantizer.level_low, quantizer.level_high)

    q_weight = q_weight.to(integer_dtype)
    zero_point = zero_point.data.to(integer_dtype)

    if quantizer.num_bits == 8:
        decompressor = INT8AsymmetricWeightsDecompressor(scale=scale, zero_point=zero_point, result_dtype=weight_dtype)
    else:
        decompressor = INT4AsymmetricWeightsDecompressor(
            scale=scale,
            zero_point=zero_point,
            compressed_weight_shape=q_weight.shape,
            result_shape=weight_shape,
            result_dtype=weight_dtype,
        )
    return decompressor, q_weight


def sym_fq_to_decompressor(
    quantizer: SymmetricQuantizer, weight: torch.Tensor
) -> tuple[Union[INT8SymmetricWeightsDecompressor, INT4SymmetricWeightsDecompressor], torch.Tensor]:
    """
    Converts an asymmetric quantizer and original weight tensor to a decompressor and quantized weight tensor.

    :param quantizer: The asymmetric quantizer instance.
    :param weight: The weight tensor to be compressed and used in decompressor.
    :return: The decompressor and quantized weight corresponding to the given quantizer and original weight.
    """
    assert isinstance(quantizer, SymmetricQuantizer)
    weight_dtype = weight.dtype
    weight_shape = weight.shape
    float_dtype = torch.float32
    integer_dtype = torch.int8

    eps = torch.finfo(float_dtype).eps
    qdq_weight = quantizer.quantize(weight)
    if hasattr(quantizer, "_lspec"):
        # Reshape for group-wise quantization, implemented for classes with lora spec only
        qdq_weight = qdq_weight.reshape(quantizer._lspec.weight_shape)
    qdq_weight = qdq_weight.to(float_dtype)

    scale = quantizer.scale.to(float_dtype) / abs(quantizer.level_low)
    scale = torch.where(torch.abs(scale) < eps, eps, scale)
    scale = scale.to(float_dtype)

    q_weight = qdq_weight / scale
    q_weight = torch.round(q_weight)
    q_weight = torch.clip(q_weight, quantizer.level_low, quantizer.level_high)

    q_weight = q_weight.to(integer_dtype)

    if quantizer.num_bits == 8:
        decompressor = INT8SymmetricWeightsDecompressor(scale=scale, result_dtype=weight_dtype)
    else:
        decompressor = INT4SymmetricWeightsDecompressor(
            scale=scale,
            compressed_weight_shape=q_weight.shape,
            result_shape=weight_shape,
            result_dtype=weight_dtype,
        )
    return decompressor, q_weight


def replace_with_decompressors(model: NNCFNetwork) -> NNCFNetwork:
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
    transformations = model.nncf.transformation_layout().transformations
    model = model.nncf.get_clean_shallow_copy()
    graph = model.nncf.get_graph()
    for command in transformations:
        quantizer = command.fn
        if not isinstance(quantizer, (SymmetricQuantizer, AsymmetricQuantizer)):
            # strip is only applied to Fake Quantizers, skip all other modules, e.g. SQMultiply for AWQ
            transformation_layout.register(command)
            continue

        msg = ""
        if quantizer._qspec.half_range or quantizer._qspec.narrow_range:
            msg += "Unexpected parameters of quantizers on strip: half_range and narrow_range should be False.\n"
        if quantizer.num_bits not in [4, 8]:
            msg += f"Unsupported number of bits {quantizer.num_bits} for the quantizer {quantizer}.\n"
        if len(command.target_points) > 1:
            msg += "Command contains more than one target point."
        if msg:
            raise nncf.ValidationError(msg)

        tp = command.target_points[0]
        weight_node = graph.get_node_by_name(tp.target_node_name)
        if weight_node is None:
            msg = "FQ is not assigned to weight. Strip to DQ format is not supported for FQ on activation."
            raise nncf.UnsupportedModelError(msg)
        weight_name = weight_node.layer_attributes.name
        weight = get_const_data(weight_node, model)

        convert_fn = asym_fq_to_decompressor if isinstance(quantizer, AsymmetricQuantizer) else sym_fq_to_decompressor
        decompressor, q_weight = convert_fn(quantizer, weight)

        packed_tensor = decompressor.pack_weight(q_weight)

        # sets compressed tensor
        # TODO:(AlexanderDokuchaev): update set_const_data
        module_name, weight_attr_name = split_const_name(weight_name)
        module = get_module_by_name(module_name, model)
        weight = getattr(module, weight_attr_name)

        if not isinstance(weight, torch.nn.Parameter):
            msg = f"Weight is not a torch.nn.Parameter in the model by name {weight_name}."
            raise nncf.InternalError(msg)

        weight.requires_grad = False
        weight.data = packed_tensor

        decompressor_name = f"weights_decompressor_{weight_node.node_name.replace('.', '_')}"
        transformation_layout.register(
            PTSharedFnInsertionCommand(
                [PTTargetPoint(TargetType.OPERATOR_POST_HOOK, target_node_name=weight_node.node_name)],
                decompressor,
                decompressor_name,
            )
        )

    return PTModelTransformer(model).transform(transformation_layout)
