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

from typing import Optional
from typing import Tuple

import torch
from numpy import argmax
from torch.quantization.fake_quantize import FakeQuantize

from nncf.api.compression import CompressionAlgorithmController
from nncf.torch.composite_compression import CompositeCompressionAlgorithmController
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.pruning.filter_pruning.layers import FilterPruningMask
from nncf.torch.pruning.operations import PT_PRUNING_OPERATOR_METATYPES
from nncf.torch.pruning.operations import ModelPruner
from nncf.torch.pruning.operations import PrunType
from nncf.torch.quantization.layers import AsymmetricQuantizer
from nncf.torch.quantization.layers import BaseQuantizer
from nncf.torch.quantization.layers import SymmetricQuantizer
from nncf.torch.quantization.quantize_functions import TuneRange

SUPPORTED_ALGORITHMS = ["quantization", "filter_pruning"]


def prepare_for_inference(
    compressed_model: NNCFNetwork,
    compressed_ctrl: CompressionAlgorithmController,
) -> None:
    """
    Prepare NNCFNetwork for inference:
      - for quantisation algorithm replace Replace NNCF quantizers modules to FakeQuantize.
      - for pruning_filter algorithm prune module by filling weights and bias with zeros.

    :param compressed_model: Compressed model.
    :param compressed_ctrl: Compression controller.
    """
    compressed_model.train(False)

    if isinstance(compressed_ctrl, CompositeCompressionAlgorithmController):
        ctrls = compressed_ctrl.child_ctrls
    else:
        ctrls = [compressed_ctrl]

    # Check supported algorithms
    for controller in ctrls:
        if controller.name not in SUPPORTED_ALGORITHMS:
            raise RuntimeError(f"Function prepare_for_inference supprots only {SUPPORTED_ALGORITHMS} algorithms.")

    # Strip the model
    for controller in ctrls:
        compressed_model = controller.strip_model(compressed_model)

    # Prepare the model for inference
    for controller in ctrls:
        if controller.name == "quantization":
            replace_quantizer_to_native_module(compressed_model)
        if controller.name == "filter_pruning":
            graph = compressed_model.get_original_graph()
            ModelPruner(compressed_model, graph, PT_PRUNING_OPERATOR_METATYPES, PrunType.FILL_ZEROS).prune_model()


def remove_nncf_prunner_operators(model: NNCFNetwork) -> None:
    """
    Remove all FilterPruningMask operators from the model.

    :param model: Compressed model.
    """
    for node in model.get_original_graph().get_all_nodes():
        if node.node_type in ["nncf_model_input", "nncf_model_output"]:
            continue

        nncf_module = model.get_containing_module(node.node_name)

        if hasattr(nncf_module, "pre_ops"):
            for key in list(nncf_module.pre_ops.keys()):
                op = nncf_module.get_pre_op(key)
                if isinstance(op.op, FilterPruningMask):
                    nncf_module.remove_pre_forward_operation(key)

        if hasattr(nncf_module, "post_ops"):
            for key in list(nncf_module.post_ops.keys()):
                op = nncf_module.get_pre_op(key)
                if isinstance(op.op, FilterPruningMask):
                    nncf_module.remove_post_forward_operation(key)


def replace_quantizer_to_native_module(model: NNCFNetwork) -> None:
    """
    Replace NNCF quantizer modules to PyTorch FakeQuantizer module.

    :param model: Target model.
    """

    for key in model.external_quantizers.keys():
        model.external_quantizers[key] = convert_to_fakequantizer(model.external_quantizers[key])

    for node in model.get_original_graph().get_all_nodes():
        if node.node_type in ["nncf_model_input", "nncf_model_output"]:
            continue

        nncf_module = model.get_containing_module(node.node_name)

        if hasattr(nncf_module, "pre_ops"):
            for key in nncf_module.pre_ops.keys():
                op = nncf_module.get_pre_op(key)
                if isinstance(op.op, BaseQuantizer):
                    op.op = convert_to_fakequantizer(op.op)

        if hasattr(nncf_module, "post_ops"):
            for key in nncf_module.post_ops.keys():
                op = nncf_module.get_post_ops(key)
                if isinstance(op.op, BaseQuantizer):
                    op.op = convert_to_fakequantizer(op.op)


def convert_to_fakequantizer(quantizer: BaseQuantizer) -> FakeQuantize:
    """
    Convert BaseQuantizer module to FakeQuantize.

    :param quantizer: NNCF Quantizer module.

    :return: Instance of FakeQuantize similar to the input quantizer.
    """
    fakequantizer = None

    num_bits = quantizer.num_bits
    assert num_bits == 8, "Support only 8bit quantisation."

    dtype = torch.qint8 if quantizer.level_low < 0 else torch.quint8
    per_channel = quantizer.per_channel
    ch_axis = argmax(quantizer.scale_shape)

    if per_channel:
        observer = torch.ao.quantization.observer.PerChannelMinMaxObserver
    else:
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
    fakequantizer.zero_point = zero_point

    # Disable observer to save parameters
    fakequantizer.disable_observer()

    return fakequantizer


def convert_asymmetric_parameters(
    level_high: int, level_low: int, input_low: torch.Tensor, input_range: torch.Tensor, levels: int, eps: float
) -> Tuple[int, int, torch.Tensor, torch.Tensor]:
    """
    Convert parameters for asymmetric quantisation.

    :param level_high: fixed the low quant number
    :param level_low: fixed the high quant number
    :param input_low: minimum limit for input value.
    :param input_range: range limit for input value.
    :param levels: Number of quantization levels.
    :param eps: Correction coefficient.

    :return: A Tuple
        quant_max - Fixed the low quant number.
        quant_min - Fixed the high quant number.
        scale - Quantizer scale.
        zero_point - Quantizer zero point.
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
    zero_point = quant_min - torch.round(min_val_neg / scale).to(torch.int)
    zero_point = torch.clamp(zero_point, quant_min, quant_max)

    return quant_max, quant_min, scale, zero_point


def convert_symmetric_parameters(
    level_high: int, level_low: int, scale: torch.Tensor, eps: float, zero_point: Optional[torch.Tensor] = None
) -> Tuple[int, int, torch.Tensor, torch.Tensor]:
    """
    Convert parameters for symmetric quantisation.

    :param level_high: Fixed the low quant number.
    :param level_low: Fixed the high quant number.
    :param scale: Quantizer scale.
    :param eps: Correction coefficient.
    :param zero_point: Quantizer zero point.

    :return: A Tuple
        quant_max - Fixed the low quant number.
        quant_min - Fixed the high quant number.
        scale - Quantizer scale.
        zero_point - Quantizer zero point.
    """
    quant_max = level_high
    quant_min = level_low

    scale = torch.reshape(scale, (-1,)) + torch.tensor([eps])
    scale = abs(scale / quant_max)

    if zero_point is not None:
        zero_point = torch.reshape(zero_point, (-1,))
    else:
        zero_point = torch.zeros_like(scale, dtype=torch.int32)

    return quant_max, quant_min, scale, zero_point
