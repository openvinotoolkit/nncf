"""
 Copyright (c) 2019-2023 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import torch
from numpy import argmax
from torch.quantization.fake_quantize import FakeQuantize

from nncf.api.compression import CompressionAlgorithmController
from nncf.common.logging import nncf_logger
from nncf.torch.composite_compression import CompositeCompressionAlgorithmController
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.pruning.operations import PT_PRUNING_OPERATOR_METATYPES
from nncf.torch.pruning.operations import ModelPruner
from nncf.torch.quantization.layers import AsymmetricQuantizer
from nncf.torch.quantization.layers import BaseQuantizer
from nncf.torch.quantization.layers import SymmetricQuantizer
from nncf.torch.quantization.quantize_functions import TuneRange

SUPPORTED_ALGORITHMS = ["quantization", "filter_prunning"]


def prepare_for_inference(
    compressed_model: NNCFNetwork, compressed_ctrl: CompressionAlgorithmController
) -> NNCFNetwork:
    """Prepare NNCFNetwork for inference.
    Replace NNCF quantizers modules to torch.FakeQuantize and fill zeroes prunned masks.

    Args:
        compressed_model (NNCFNetwork): Compressed model.
        compressed_ctrl (CompressionAlgorithmController): Compression controller.

    Returns:
        NNCFNetwork: Model for inference.
    """

    # TODO: may be create copy of model?
    compressed_model.train(False)

    if isinstance(compressed_ctrl, CompositeCompressionAlgorithmController):
        ctrls = compressed_ctrl.child_ctrls
    else:
        ctrls = [compressed_ctrl]

    # Check supported algorithms
    for controller in ctrls:
        if controller.name not in SUPPORTED_ALGORITHMS:
            raise RuntimeError(f"Function prepare_for_inference supprots only {SUPPORTED_ALGORITHMS} algorithms.")

    # Strip model
    inference_model = compressed_model
    for controller in ctrls:
        inference_model = controller.strip_model(inference_model)

    for controller in ctrls:
        if controller.name == "filter_prunning":
            graph = inference_model.get_original_graph()
            ModelPruner(inference_model, graph, PT_PRUNING_OPERATOR_METATYPES).mask_propagation()

    inference_model = replace_quantizer_to_native_module(inference_model)

    remove_nncf_prunner_operators(inference_model)

    return inference_model


def remove_nncf_prunner_operators(model: NNCFNetwork) -> None:
    """_summary_

    Args:
        model (NNCFNetwork): _description_
    """
    for node in model.get_original_graph().get_all_nodes():
        if node.node_type in ["nncf_model_input", "nncf_model_output"]:
            continue

        nncf_module = model.get_containing_module(node.node_name)

        # TODO: change condition to remove and raise error if exists not expected operations
        if hasattr(nncf_module, "pre_ops"):
            for key in nncf_module.pre_ops.keys():
                op = nncf_module.get_pre_op(key)
                if not isinstance(op.op, FakeQuantize):
                    nncf_module.remove_pre_forward_operation(key)

        if hasattr(nncf_module, "post_ops"):
            for key in nncf_module.post_ops.keys():
                op = nncf_module.get_pre_op(key)
                if not isinstance(op.op, FakeQuantize):
                    nncf_module.remove_post_forward_operation(key)


def replace_quantizer_to_native_module(model: NNCFNetwork) -> NNCFNetwork:
    """Replace NNCF quantizer modules to PyTorch FakeQuantizer module.

    Args:
        model (NNCFNetwork): Target model.

    Returns:
        NNCFNetwork: Model with replaced quanizer modules.
    """

    for key in model.external_quantizers.keys():
        nncf_op = model.external_quantizers[key]
        model.external_quantizers[key] = convert_to_fakequantizer(model.external_quantizers[key])
        model.external_quantizers[key].nncf_op = nncf_op

    for node in model.get_original_graph().get_all_nodes():
        if node.node_type in ["nncf_model_input", "nncf_model_output"]:
            continue

        nncf_module = model.get_containing_module(node.node_name)

        if hasattr(nncf_module, "pre_ops"):
            for key in nncf_module.pre_ops.keys():
                op = nncf_module.get_pre_op(key)
                if isinstance(op.op, BaseQuantizer):
                    op.nncf_op = op.op  # TODO: remove
                    op.op = convert_to_fakequantizer(op.op)

        if hasattr(nncf_module, "post_ops"):
            for key in nncf_module.post_ops.keys():
                op = nncf_module.get_post_ops(key)
                if isinstance(op.op, BaseQuantizer):
                    op.nncf_op = op.op  # TODO: remove
                    op.op = convert_to_fakequantizer(op.op)

    return model


def convert_to_fakequantizer(quantizer: BaseQuantizer) -> FakeQuantize:
    """Convert BaseQuantizer module to torch.FakeQuantize.

    Args:
        quantizer (BaseQuantizer): NNCF Quantizer module that will be converted.

    Returns:
        FakeQuantize: _description_
    """
    fakequantizer = None

    num_bits = quantizer.num_bits
    assert num_bits == 8, "Support only 8bit quantisation."

    signed = quantizer.signed
    if quantizer.level_low == 0 and signed:
        # TODO: rethink...
        nncf_logger.warning("Incorrect signed flag")
        signed = False

    dtype = torch.qint8 if signed else torch.quint8
    per_channel = quantizer.per_channel
    ch_axis = argmax(quantizer.scale_shape)

    if per_channel:
        observer = torch.ao.quantization.observer.PerChannelMinMaxObserver
    else:
        # TODO: try to use FixedQParamsObserver only for per_tensor
        observer = torch.ao.quantization.observer.MinMaxObserver

    # reduce_range # TODO

    if isinstance(quantizer, SymmetricQuantizer):
        qscheme = torch.per_channel_symmetric if per_channel else torch.per_tensor_symmetric
        quant_max, quant_min, scale, zero_point = convert_symmetric_parameters(
            level_high=quantizer.level_high,
            level_low=quantizer.level_low,
            scale=quantizer.scale.data,
            eps=quantizer.eps,
            zero_point=quantizer.zero_point if hasattr(quantizer, "zero_point") else None,
        )
    elif isinstance(quantizer, AsymmetricQuantizer):
        qscheme = torch.per_channel_affine if per_channel else torch.per_tensor_affine
        quant_max, quant_min, scale, zero_point = convert_asymmetric_parameters(
            level_high=quantizer.level_high,
            level_low=quantizer.level_low,
            input_low=quantizer.input_low.data,
            input_range=quantizer.input_range.data,
            levels=quantizer.levels,
            eps=quantizer.eps,
        )
    else:
        raise RuntimeError(f"Unknow class of quntizer: {quantizer}")

    fakequantizer = FakeQuantize(
        observer=observer,
        quant_max=quant_max,
        quant_min=quant_min,
        dtype=dtype,
        qscheme=qscheme,
        eps=quantizer.eps,
    )

    fakequantizer.scale = scale
    fakequantizer.ch_axis = ch_axis
    fakequantizer.activation_post_process.ch_axis = ch_axis
    fakequantizer.zero_point = zero_point

    # Disable observer to save parameters
    fakequantizer.disable_observer()

    return fakequantizer


def convert_asymmetric_parameters(level_high, level_low, input_low, input_range, levels, eps):
    """Convert parameters for asymmetric quantisation.

    Args:
        level_high (_type_): _description_
        level_low (_type_): _description_
        input_low (_type_): _description_
        input_range (_type_): _description_
        levels (_type_): _description_
        eps (_type_): _description_

    Returns:
        _type_: _description_
    """
    quant_max = level_high
    quant_min = level_low

    input_low = torch.reshape(input_low, (-1,))
    input_range = torch.reshape(input_range, (-1,))

    input_range_safe = abs(input_range) + eps
    input_low_tuned, input_range_tuned = TuneRange.apply(input_low, input_range_safe, levels)

    min_val_neg = input_low_tuned
    max_val_pos = input_low_tuned + input_range_tuned

    scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
    scale = torch.max(scale, torch.tensor([eps]))
    zero_point = quant_min - torch.round(min_val_neg / scale).to(torch.int)
    zero_point = torch.clamp(zero_point, quant_min, quant_max)

    return quant_max, quant_min, scale, zero_point


def convert_symmetric_parameters(level_high, level_low, scale, eps, zero_point=None):
    """Convert parameters for symmetric quantisation.

    Args:
        level_high (_type_): _description_
        level_low (_type_): _description_
        input_low (_type_): _description_
        input_range (_type_): _description_
        levels (_type_): _description_
        eps (_type_): _description_

    Returns:
        _type_: _description_
    """
    quant_max = level_high
    quant_min = level_low

    scale = torch.reshape(scale, (-1,)) + torch.tensor([eps])
    scale = abs(scale / quant_max)

    if zero_point:
        zero_point = torch.reshape(zero_point, (-1,))
    else:
        zero_point = torch.zeros_like(scale, dtype=torch.int32)  # TODO: gpu?

    return quant_max, quant_min, scale, zero_point
