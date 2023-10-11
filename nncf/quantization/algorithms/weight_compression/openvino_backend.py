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

from dataclasses import dataclass
from typing import List, Optional, Tuple, TypeVar, Union

import numpy as np
import openvino.runtime as ov
from openvino.runtime import opset9 as opset

from nncf.common.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.logging import nncf_logger
from nncf.common.utils.backend import BackendType
from nncf.common.utils.helpers import create_table
from nncf.openvino.graph.metatypes.openvino_metatypes import OVEmbeddingMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVMatMulMetatype
from nncf.openvino.graph.node_utils import get_channel_agnostic_reduction_axes
from nncf.openvino.graph.node_utils import get_const_value
from nncf.openvino.graph.node_utils import get_weight_channel_axes
from nncf.parameters import CompressWeightsMode
from nncf.quantization.algorithms.weight_compression.backend import ALGO_BACKENDS
from nncf.quantization.algorithms.weight_compression.backend import WeightCompressionAlgoBackend
from nncf.quantization.fake_quantize import calculate_scale_zero_point


@ALGO_BACKENDS.register(BackendType.OPENVINO)
class OVWeightCompressionAlgoBackend(WeightCompressionAlgoBackend):
    @property
    def weighted_metatypes(self) -> List[OperatorMetatype]:
        return [OVMatMulMetatype, OVEmbeddingMetatype]

    @staticmethod
    def is_node_with_weights(node: NNCFNode) -> bool:
        return node.layer_attributes and node.layer_attributes.constant_attributes

    @staticmethod
    def validate_params(mode: CompressWeightsMode) -> None:
        pass

    @staticmethod
    def do_compression(
        model: ov.Model,
        nodes_to_compress: List[NNCFNode],
        mode: CompressWeightsMode,
        ratio: float = None,
        group_size: int = None,
    ) -> ov.Model:
        all_weight_params: List[WeightNodeParams] = []
        quantized_nodes_ids = set()

        friendly_name_to_op_map = {op.get_friendly_name(): op for op in model.get_ops()}

        for nncf_node in nodes_to_compress:
            weight_port_ids = nncf_node.layer_attributes.get_const_port_ids()
            for weight_port_id in weight_port_ids:
                weight_op_friendly_name = nncf_node.layer_attributes.constant_attributes[weight_port_id]["name"]
                weight_node = friendly_name_to_op_map[weight_op_friendly_name]
                if weight_node is None:
                    continue
                if id(weight_node) in quantized_nodes_ids:
                    continue
                weight_output = weight_node.output(0)

                original_weight_dtype = weight_output.get_element_type().to_dtype()
                if original_weight_dtype not in [np.float32, np.float16, np.float64]:
                    continue
                const_shape = nncf_node.layer_attributes.constant_attributes[weight_port_id]["shape"]
                channel_axes = get_weight_channel_axes(nncf_node, weight_port_id)
                axes = get_channel_agnostic_reduction_axes(channel_axes, const_shape)
                fq_name = f"{weight_op_friendly_name}/fq_weights_{weight_port_id}"
                num_weights = np.prod(const_shape)
                weight_params = WeightNodeParams(axes, num_weights, fq_name, weight_node, original_weight_dtype)
                all_weight_params.append(weight_params)
                quantized_nodes_ids.add(id(weight_node))

        if mode == CompressWeightsMode.NF4:
            _assign_mixed_precision(all_weight_params, ratio, group_size)

        for wp in all_weight_params:
            weight_node = wp.weight_node
            original_weight_dtype = wp.original_weight_dtype

            weight_output = weight_node.output(0)
            weight_name = weight_node.get_friendly_name()
            target_inputs = weight_output.get_target_inputs()

            weight = get_const_value(weight_node)
            config = wp.compression_config

            if config.is_nf4:
                original_shape = weight.shape
                norm_weight, scale = _get_norm_weight_and_nf4_scale(weight, wp.reduction_axes, group_size)
                compressed_const = opset.constant(norm_weight, dtype=ov.Type.nf4, name=weight_name)
                convert = opset.convert(compressed_const, original_weight_dtype)
                mul = opset.multiply(convert, scale.astype(original_weight_dtype), name=wp.fq_name)
                if config.group_size != -1:
                    mul = opset.reshape(mul, output_shape=original_shape, special_zero=False)
                last_output = mul.output(0)
            else:
                compressed_weights, scale, zero_point = _int8_compress(weight, wp.reduction_axes)
                compressed_const = opset.constant(compressed_weights, dtype=np.uint8, name=weight_name)
                convert = opset.convert(compressed_const, original_weight_dtype)
                sub = opset.subtract(convert, zero_point.astype(original_weight_dtype))
                mul = opset.multiply(sub, scale.astype(original_weight_dtype), name=wp.fq_name)
                last_output = mul.output(0)

            for target_input in target_inputs:
                target_input.replace_source_output(last_output)
        return model


TWeightType = TypeVar("TWeightType")

NF4_QUANTILES = np.array(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ]
)

CENTER_OF_NF4_QUANTILES = (NF4_QUANTILES[1:] + NF4_QUANTILES[:-1]) / 2


@dataclass
class WeightCompressionConfig:
    """
    Information on how to compress (quantize) a specific weight.

    :param num_bits: number of bits for storing a single quantized value. 8, by default.
    :param is_nf4: is NF4 format used for quantization. False, by default.
    :param group_size: number of weights (e.g. 128) in the channel dimension that share quantization parameters (scale).
        The value -1 means no grouping. Defaults to -1.
    """

    num_bits: Optional[int] = 8
    is_nf4: Optional[bool] = False
    group_size: Optional[int] = -1


@dataclass
class WeightNodeParams:
    """
    Information about weight node in the ov.Model that is useful for weight compression.

    :param reduction_axes: Axis or axes along which to reduce (collect) different statistics (e.g. min, max).
    :param num_weights: number of elements in the weight array.
    :param fq_name: name for the inserted weight compression operation.
    :param weight_node: the weight node itself.
    :param original_weight_dtype: type of elements in the weight array.
    :param compression_config: configuration of weight compression for the weight node.
    """

    reduction_axes: Union[int, Tuple[int]]
    num_weights: int
    fq_name: str
    weight_node: ov.Node
    original_weight_dtype: TWeightType
    compression_config = WeightCompressionConfig()


def _int8_compress(
    weight: np.ndarray, reduction_axes: Union[int, Tuple[int]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Do unsigned int8 asymmetric weight compression - quantization to [0, 255] range.

    :param weight: Weight array to compress
    :param reduction_axes: Axis or axes along which to reduce (collect) different statistics (e.g. min, max).
    :return: compressed weights in unsigned int8, scale and zero point that was used for its quantization.
    """
    num_bits = 8
    level_low = 0
    level_high = 2**num_bits - 1

    min_values = np.min(weight, axis=reduction_axes, keepdims=True)
    max_values = np.max(weight, axis=reduction_axes, keepdims=True)

    scale, zero_point = calculate_scale_zero_point(min_values, max_values, level_low, level_high, narrow_range=False)

    compressed_weights = np.round(weight / scale + zero_point)
    compressed_weights = np.clip(compressed_weights, level_low, level_high).astype(np.uint8)
    return compressed_weights, scale, zero_point


def _get_int8_err(weight: np.ndarray, reduction_axes: Union[int, Tuple[int]]) -> float:
    """
    Calculates a quantity characterizing the difference between floating point weights and its int8 fake quantized
    (compressed and decompressed) version.

    :param weight: Weight array to compress.
    :param reduction_axes: Axis or axes along which to reduce (collect) different statistics (e.g. min, max).
    :return: The quantity characterizing the int8 error.
    """
    compressed_weights, scale, zero_point = _int8_compress(weight, reduction_axes)

    decompressed_weight = compressed_weights.astype(dtype=scale.dtype)
    decompressed_weight = (compressed_weights - zero_point) * scale

    diff = (decompressed_weight - weight) ** 2
    layer_err = np.mean(diff, axis=reduction_axes)
    val = np.max(layer_err)
    return val


def _calculate_scale_per_group(
    weight: np.ndarray, reduction_axes: Union[int, Tuple[int]], group_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates scale and reshapes weights for group-wise quantization.
    Having weights with shapes [c_out, c_in] and group size = 128, the shape of scale is [c_out, c_in // 128, 1], and
    shape of weights is [c_out, c_in // 128, 128].

    :param weight: Weight array to compress.
    :param reduction_axes: Axis or axes along which to reduce (collect) different statistics (e.g. min, max).
    :param group_size: number of weights (e.g. 128) in the channel dimension that share quantization parameters (scale).
    :return: Scale and reshaped weights.
    """
    assert group_size != -1
    if isinstance(reduction_axes, tuple) and len(reduction_axes) != 1:
        raise RuntimeError(
            f"group-quantization is supported for a single reduction axes, but got {len(reduction_axes)}"
        )
    reduction_axis = reduction_axes[0] if isinstance(reduction_axes, tuple) else reduction_axes
    channel_size = weight.shape[reduction_axis]
    if channel_size % group_size != 0:
        raise RuntimeError(f"Channel size {channel_size} should be divisible by size of group {group_size}")

    num_groups_per_channel = channel_size // group_size
    shape = list(weight.shape)  # [a1, r, a2] - "r" refers to number of channels along reduction axis
    shape[reduction_axis : reduction_axis + 1] = (num_groups_per_channel, group_size)
    reshaped_weight = weight.reshape(shape)  # [a1, r, a2] -> [a1, r//gs, gs, a2], when "gs" is group size
    scale = np.max(np.abs(reshaped_weight), axis=reduction_axis + 1, keepdims=True)  # [a1, r//gs, 1, a2]
    return scale, reshaped_weight


def _get_norm_weight_and_nf4_scale(
    weight: np.ndarray, reduction_axes: Tuple[int], group_size: int = -1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates scale for nf4 quantization and normalizes weights by the scale.
    Weights are reshaped in case of positive value of group size.

    :param weight: Weight array to compress.
    :param reduction_axes: Axis or axes along which to reduce (collect) different statistics (e.g. min, max).
    :param group_size: number of weights (e.g. 128) in the channel dimension that share quantization parameters (scale).
        The value -1 means no grouping. Defaults to -1.
    :return: Normalized weights and nf4 scale.
    """
    if group_size != -1:
        # shape of scale : [a1, r//gs, 1, a2], scale of weight: [a1, r//gs, r, a2]
        scale, weight = _calculate_scale_per_group(weight, reduction_axes, group_size)
    else:
        scale = np.max(np.abs(weight), axis=reduction_axes, keepdims=True)  # [a1, 1, a2]
    eps = np.finfo(weight.dtype).eps
    # NOTE: adding machine epsilon to avoid division by zero
    scale[np.abs(scale) < eps] = eps
    norm_weight = weight / scale
    return norm_weight, scale


def _get_nf4_error(weight: np.ndarray, reduction_axes: Tuple[int], group_size: int = -1) -> float:
    """
    Calculates a quantity characterizing the difference between floating point weights and its nf4 fake quantized
    (compressed and decompressed) version.

    :param weight: Weight array to compress.
    :param reduction_axes: Axis or axes along which to reduce (collect) different statistics (e.g. min, max).
    :return: The quantity characterizing the nf4 error.
    """
    original_shape = weight.shape

    norm_weight, scale = _get_norm_weight_and_nf4_scale(weight, reduction_axes, group_size)

    index_of_quantile = np.searchsorted(CENTER_OF_NF4_QUANTILES, norm_weight)
    nf4_rounded = NF4_QUANTILES[index_of_quantile]

    decompressed_weight = nf4_rounded * scale
    decompressed_weight = decompressed_weight.reshape(original_shape)
    diff = (decompressed_weight - weight) ** 2
    layer_err = np.mean(diff, axis=reduction_axes)
    val = np.max(layer_err)
    return val


def _proportion_str(num_weights_list: List[int], total_num_weights: int, total_num_params: int) -> str:
    percentage = sum(num_weights_list) / max(total_num_weights, 1) * 100
    return f"{percentage:.0f}% ({len(num_weights_list)} / {total_num_params})"


def _get_bitwidth_distribution_str(all_weight_params: List[WeightNodeParams]) -> str:
    """
    Generates a table that shows the ratio of weights quantized to different number of bits.

    :param all_weight_params: List of information about each weight node.
    :return: A string containing the table.
    """
    total_num_weights = sum(ws.num_weights for ws in all_weight_params)
    num_internal_weights = 0
    num_params = len(all_weight_params)
    num_internal_params = 0
    if num_params > 2:
        num_internal_params = num_params - 2
    num_bits_vs_num_weights_map = {}
    for i, data in enumerate(all_weight_params):
        num_bits = data.compression_config.num_bits
        n_total, n_internal = num_bits_vs_num_weights_map.get(num_bits, ([], []))
        if i not in (0, num_params - 1):
            n_internal.append(data.num_weights)
            num_internal_weights += data.num_weights
        n_total.append(data.num_weights)
        num_bits_vs_num_weights_map[num_bits] = (n_total, n_internal)

    # Table creation
    header = ["Num bits (N)", "% all weight", "% internal weights"]
    rows = []
    for bitwidth, (n_total, n_internal) in num_bits_vs_num_weights_map.items():
        rows.append(
            [
                bitwidth,
                _proportion_str(n_total, total_num_weights, num_params),
                _proportion_str(n_internal, num_internal_weights, num_internal_params),
            ]
        )

    table = create_table(header, rows)
    pretty_string = f"Statistics of the bitwidth distribution:\n{table}"
    return pretty_string


def _assign_mixed_precision(all_weight_params: List[WeightNodeParams], ratio: float, group_size: int) -> None:
    """
    Assigns mixed quantization scheme (e.g. uniform int8 or non-uniform nf4) for weights based on some criteria.

    :param all_weight_params: List of information about each weight node. The quantization scheme is added to this info.
    :param ratio: the ratio between baseline and backup precisions (e.g. 0.9 means 90% of layers quantized to NF4
        and the rest to INT8).
    :param group_size: number of weights (e.g. 128) in the channel dimension that share quantization parameters (scale).
        The value -1 means no grouping.
    """
    nf4_config = WeightCompressionConfig(num_bits=4, is_nf4=True, group_size=group_size)
    if ratio != 1:
        # NOTE: first and last layer is always in 8 bit.
        errors = []
        num_internal_weights = 0
        for weight_param in all_weight_params[1:-1]:
            weight = get_const_value(weight_param.weight_node)
            axes = weight_param.reduction_axes
            nf4_error = _get_nf4_error(weight, axes, group_size)
            int8_error = _get_int8_err(weight, axes)
            eps = np.finfo(weight.dtype).eps
            error = nf4_error / (int8_error + eps)
            errors.append(error)
            num_internal_weights += weight_param.num_weights
        # NOTE: index is defined in the array of all weight params by taking into account that errors were not
        # calculated for first and last layers.
        indexes_of_layers_in_ascending_order_of_errors = [
            i[0] + 1 for i in sorted(enumerate(errors), reverse=False, key=lambda x: x[1])
        ]
        num_weights_in_4bit = 0
        for index in indexes_of_layers_in_ascending_order_of_errors:
            weight_param = all_weight_params[index]
            current_ratio = (num_weights_in_4bit + weight_param.num_weights) / num_internal_weights
            if current_ratio >= ratio:
                break
            weight_param.compression_config = nf4_config
            num_weights_in_4bit += weight_param.num_weights

    else:
        for weight_param in all_weight_params[1:-1]:
            weight_param.compression_config = nf4_config
    nncf_logger.info(_get_bitwidth_distribution_str(all_weight_params))
