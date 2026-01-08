# Copyright (c) 2026 Intel Corporation
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
from nncf.torch.quantization.layers import AsymmetricQuantizer
from nncf.torch.quantization.layers import BaseQuantizer
from nncf.torch.quantization.layers import INT4AsymmetricWeightsDecompressor
from nncf.torch.quantization.layers import INT4SymmetricWeightsDecompressor
from nncf.torch.quantization.layers import INT8AsymmetricWeightsDecompressor
from nncf.torch.quantization.layers import INT8SymmetricWeightsDecompressor
from nncf.torch.quantization.layers import SymmetricQuantizer
from nncf.torch.quantization.quantize_functions import TuneRange

SUPPORTED_NUM_BITS_FOR_STRIP_MODEL = [8]


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
