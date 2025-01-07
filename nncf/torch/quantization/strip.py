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


import numpy as np
import torch
from torch.quantization.fake_quantize import FakeQuantize

import nncf
from nncf.torch.graph.transformations.commands import ExtraCompressionModuleType
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.quantization.layers import AsymmetricQuantizer
from nncf.torch.quantization.layers import BaseQuantizer
from nncf.torch.quantization.layers import SymmetricQuantizer

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
        raise nncf.InternalError(
            "Converting nncf quantizer module to torch native only supports "
            f"for num_bits in {SUPPORTED_NUM_BITS_FOR_STRIP_MODEL}."
        )
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
    model = replace_quantizer_to_torch_native_module(model)
    model = remove_disabled_quantizers(model)
    return model
