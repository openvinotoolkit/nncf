from typing import Optional

import torch
from torch import nn

from nncf.torch.layers import NNCFEmbedding
from nncf.torch.layers import NNCFLinear
from nncf.torch.quantization.quantize_functions import get_scale_zp_from_input_low_input_high


class WeightsCompressor(nn.Module):
    """Class provided fake quantize operation or dequantization of compressed weights in forward pass

    Atributes:
        zero_point: zero point in quantization scheme
        scale: scale in quantizatin scheme
    """

    def __init__(self, zero_point, scale):
        super().__init__()
        self.zero_point = zero_point
        self.scale = scale

    def forward(self, input, op_arg):
        if input.weight.dtype is torch.uint8:
            w = input.weight.type(dtype=self.scale.dtype)
            input.weight = (w - self.zero_point) * self.scale
        else:
            axis = 0 if input.weight.shape[0] == self.scale.shape[0] else 1
            input.weight = torch.fake_quantize_per_channel_affine(
                input.weight, self.scale, self.zero_point, axis, 0, 255
            )


def insert_pre_compression_operations(module: nn.Module, compress_weights=False) -> Optional[nn.Module]:
    """
    Insets weights compression with dequantization or quantization pre operation for Linear and Embedding layers.

    :param module: The module to insert the weights compression.
    :param compress_weights: Enables real compression of weights in Linear and Embedding layers.
        If False inserts pytorch torch.fake_quantize_per_channel_affine(),
        else compress weights to int8 and inserts custom dequantization.
    :return: The module with inserted operations. The module is not trainable if compress_weights is True.
    """
    q_dims = {NNCFEmbedding: 0, NNCFLinear: 1}
    allowed_types = [NNCFEmbedding, NNCFLinear]
    for _, layer in module.named_children():
        if not type(layer) in allowed_types:
            insert_pre_compression_operations(layer, compress_weights)
            continue
        q_dim = q_dims[type(layer)]
        input_low = torch.min(layer.weight, dim=q_dim)[0].detach()
        input_high = torch.max(layer.weight, dim=q_dim)[0].detach()
        scale, zero_point = get_scale_zp_from_input_low_input_high(0, 255, input_low, input_high)

        if compress_weights:
            scale = scale.unsqueeze(q_dim)
            zero_point = zero_point.unsqueeze(q_dim)
            layer.register_pre_forward_operation(WeightsCompressor(zero_point, scale))

            compressed_weight = layer.weight.data / scale + zero_point
            compressed_weight = torch.clamp(torch.round(compressed_weight), 0, 255)

            layer.weight.requires_grad = False
            layer.weight.data = compressed_weight.type(dtype=torch.uint8)
        else:
            zero_point = zero_point.type(dtype=torch.int32)
            layer.register_pre_forward_operation(WeightsCompressor(zero_point, scale))
