# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional

import torch
from torch.nn import nn

from nncf.common.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.utils.backend import BackendType
from nncf.parameters import CompressWeightsMode
from nncf.quantization.algorithms.smooth_quant.backend import ALGO_BACKENDS
from nncf.quantization.algorithms.weight_compression.backend import WeightCompressionAlgoBackend
from nncf.torch.graph.operator_metatypes import PTModuleEmbeddingMetatype
from nncf.torch.graph.operator_metatypes import PTModuleLinearMetatype
from nncf.torch.layers import NNCF_WRAPPED_USER_MODULES_DICT
from nncf.torch.nncf_module_replacement import replace_modules_by_nncf_modules
from nncf.torch.quantization.quantize_functions import get_scale_zp_from_input_low_input_high


class WeightsDecompressor(nn.Module):
    """
    Applies decompression of compressed weights on the forward pass.

    Attributes:
        zero_point: zero point in quantization scheme
        scale: scale in quantization scheme
    """

    def __init__(self, zero_point, scale):
        super().__init__()
        self.zero_point = zero_point
        self.scale = scale

    def forward(self, layer, op_arg):
        w = layer.weight.type(dtype=self.scale.dtype)
        layer.weight = (w - self.zero_point) * self.scale


@ALGO_BACKENDS.register(BackendType.TORCH)
class PTWeightCompressionAlgoBackend(WeightCompressionAlgoBackend):
    @property
    def weighted_metatypes(self) -> List[OperatorMetatype]:
        return [PTModuleLinearMetatype, PTModuleEmbeddingMetatype]

    @staticmethod
    def is_node_with_weights(_: NNCFNode) -> bool:
        return True

    @staticmethod
    def do_compression(
        model: nn.Module,
        nodes_to_compress: List[NNCFNode],
        mode: CompressWeightsMode,
        ratio: float = None,
        group_size: int = None,
    ) -> nn.Module:
        """
        Compress weights of Linear and Embedding layers to 8-bit integer.

        :param model: The Torch model for applying weight compression.
        :param nodes_to_compress: List of nodes in the model's graph,
            corresponding to the layers for weight compression.
        :param mode: Defines a mode for weight compression.
            INT8 stands for 8-bit integer quantization of all weights.
            NF4 stands for a mixed-precision weights quantization to NF4 data type. The first and last layers
            are always compressed to a backup precision which is 8-bit integer by default. All others are quantized
            whether to NF4 or to a backup precision depending on criteria and the given ratio.
        :param ratio: the ratio between baseline and backup precisions (e.g. 0.9 means 90% of layers quantized to NF4
            and the rest to INT8).
        :param group_size: number of weights (e.g. 128) in the channel dimension
            that share quantization parameters (scale). The value -1 means no grouping.
        :return: The non-trainable module with inserted operations.
        """
        model, _ = replace_modules_by_nncf_modules(model)

        bits = 8
        level_high = 2**bits - 1
        assert level_high < 256

        user_types = list(NNCF_WRAPPED_USER_MODULES_DICT.values())

        if compression_hist is None:
            compression_hist = {}
        for node in nodes_to_compress:
            layer = model.nncf.get_containing_module(node.node_name)

            if not type(layer) in user_types:
                continue

            if layer.weight.dtype in [torch.uint8, torch.int8]:
                if layer.weight in compression_hist:
                    layer.register_pre_forward_operation(compression_hist[layer.weight])
                continue

            target_dim = layer.target_weight_dim_for_compression
            stat_dim = (target_dim + 1) % 2
            input_low = torch.min(layer.weight, dim=stat_dim).values.detach()
            input_high = torch.max(layer.weight, dim=stat_dim).values.detach()
            scale, zero_point = get_scale_zp_from_input_low_input_high(0, level_high, input_low, input_high)

            scale = scale.unsqueeze(stat_dim)
            zero_point = zero_point.unsqueeze(stat_dim)
            key = layer.register_pre_forward_operation(WeightsDecompressor(zero_point, scale))

            compressed_weight = layer.weight.data / scale + zero_point
            compressed_weight = torch.clamp(torch.round(compressed_weight), 0, level_high)

            layer.weight.requires_grad = False
            layer.weight.data = compressed_weight.type(dtype=torch.uint8)

            compression_hist[layer.weight] = layer.get_pre_op(key)

        return model

    @staticmethod
    def validate_params(mode: CompressWeightsMode) -> None:
        if mode != CompressWeightsMode.INT8:
            raise AttributeError(f"Torch backend supports only INT8 mode for weight compression, but given {mode} mode")
