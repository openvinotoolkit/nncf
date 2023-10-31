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
from nncf.common.logging.track_progress import track
from nncf.common.utils.helpers import create_table
from nncf.openvino.graph.metatypes.openvino_metatypes import OVEmbeddingMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVMatMulMetatype
from nncf.openvino.graph.node_utils import get_channel_agnostic_reduction_axes
from nncf.openvino.graph.node_utils import get_const_value
from nncf.openvino.graph.node_utils import get_weight_channel_axes
from nncf.parameters import CompressWeightsMode
from nncf.quantization.algorithms.weight_compression.backend import WeightCompressionAlgoBackend
from nncf.quantization.fake_quantize import calculate_scale_zero_point
from nncf.scopes import IgnoredScope


class OVWeightCompressionAlgoBackend(WeightCompressionAlgoBackend):
    @property
    def weighted_metatypes(self) -> List[OperatorMetatype]:
        return [OVMatMulMetatype, OVEmbeddingMetatype]

    @staticmethod
    def is_node_with_weights(node: NNCFNode) -> bool:
        return node.layer_attributes and node.layer_attributes.constant_attributes

    @staticmethod
    def validate_params(mode: CompressWeightsMode, ignored_scope: Optional[IgnoredScope] = None) -> None:
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

        if mode != CompressWeightsMode.INT8:
            primary_config = WeightCompressionConfig(mode=mode, group_size=group_size)
            _assign_mixed_precision(all_weight_params, ratio, primary_config)

        nncf_logger.info(_get_bitwidth_distribution_str(all_weight_params))

        for wp in track(all_weight_params, description="Applying Weight Compression"):
            weight_node = wp.weight_node
            original_weight_dtype = wp.original_weight_dtype

            weight_output = weight_node.output(0)
            weight_name = weight_node.get_friendly_name()
            target_inputs = weight_output.get_target_inputs()

            weight = get_const_value(weight_node)
            config = wp.compression_config
            if config.mode == CompressWeightsMode.NF4:
                original_shape = weight.shape
                norm_weight, scale = _get_norm_weight_and_nf4_scale(weight, wp.reduction_axes, group_size)
                compressed_const = opset.constant(norm_weight, dtype=ov.Type.nf4, name=weight_name)
                convert = opset.convert(compressed_const, original_weight_dtype)
                mul = opset.multiply(convert, scale.astype(original_weight_dtype), name=wp.fq_name)
                if config.group_size != -1:
                    mul = opset.reshape(mul, output_shape=original_shape, special_zero=False)
                last_output = mul.output(0)
            else:
                original_shape = weight.shape
                compressed_weights, scale, zero_point = _do_integer_quantization(weight, wp.reduction_axes, config)
                compression_type = np.uint8 if config.num_bits == 8 else ov.Type.u4
                compressed_weights_node = opset.constant(compressed_weights, dtype=compression_type, name=weight_name)
                convert_weights_node = opset.convert(compressed_weights_node, original_weight_dtype)
                zero_point_node = opset.constant(zero_point, dtype=compression_type, name=f"{weight_name}/ZP")
                convert_zp_node = opset.convert(zero_point_node, original_weight_dtype)
                sub = opset.subtract(convert_weights_node, convert_zp_node)
                mul = opset.multiply(sub, scale.astype(original_weight_dtype), name=wp.fq_name)
                if config.group_size != -1:
                    mul = opset.reshape(mul, output_shape=original_shape, special_zero=False)
                last_output = mul.output(0)

            for target_input in target_inputs:
                target_input.replace_source_output(last_output)
        return model


TWeightType = TypeVar("TWeightType")


@dataclass
class WeightCompressionConfig:
    """
    Information on how to compress (quantize) a specific weight.

    :param mode: Defines a mode for weight compression. Defaults to INT8 mode.
    :param group_size: Number of weights (e.g. 128) in the channel dimension that share quantization parameters (scale).
        The value -1 means no grouping. Defaults to -1.
    """

    mode: Optional[CompressWeightsMode] = CompressWeightsMode.INT8
    group_size: Optional[int] = -1

    @property
    def num_bits(self):
        """
        :return: number of bits that is used for storing a single quantized value in the given mode.
        """
        return 8 if self.mode == CompressWeightsMode.INT8 else 4


@dataclass
class WeightNodeParams:
    """
    Information about weight node in the ov.Model that is useful for weight compression.

    :param reduction_axes: Axis or axes along which to reduce (collect) different statistics (e.g. min, max).
    :param num_weights: Number of elements in the weight array.
    :param fq_name: Name for the inserted weight compression operation.
    :param weight_node: The weight node itself.
    :param original_weight_dtype: Type of elements in the weight array.
    :param compression_config: Configuration of weight compression for the weight node.
    """

    reduction_axes: Union[int, Tuple[int]]
    num_weights: int
    fq_name: str
    weight_node: ov.Node
    original_weight_dtype: TWeightType
    compression_config = WeightCompressionConfig()


def _do_integer_quantization(
    weight: np.ndarray, reduction_axes: Union[int, Tuple[int]], config: WeightCompressionConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The method quantizes the given weights to integer data type in accordance with the compression config.
    The config defines a quantization mode:
        INT8 mode refers to unsigned int8 asymmetric weight compression - quantization to [0, 255] range.
        INT4_ASYM mode refers to unsigned int4 asymmetric weight compression with a typical non-fixed zero-point -
            quantization to [0, 15] range.
        INT4_SYM mode refers to unsigned int4 symmetric weight compression with a fixed zero point equals to 8 -
            quantization to [0, 15] range.
        NF4 mode requires a dedicated procedure and it is not supported in this method.
    One of the parameter of compression config is a group size. Quantization is per-channel, if group size equals to -1,
    otherwise it's per-group, i.e. group size number of weights in the channel dimension share quantization parameters
    (scales).

    :param weight: Weight array to compress.
    :param reduction_axes: Axis or axes along which to reduce (collect) different statistics (e.g. min, max).
    :param config: Information on how to compress (quantize) a specific weight.
    :return: The compressed weights, scale and zero point that was used for its quantization.
    """
    mode = config.mode
    assert mode != CompressWeightsMode.NF4, "The function supports integer quantization only"
    group_size = config.group_size
    num_bits = config.num_bits

    level_low = 0
    level_high = 2**num_bits - 1

    if group_size != -1:
        # weights are reshaped from [a1, r, a2] to [a1, r//gs, gs, a2]
        weight, reduction_axes = _reshape_weights_for_grouped_quantization(weight, reduction_axes, group_size)

    if mode in [CompressWeightsMode.INT8, CompressWeightsMode.INT4_ASYM]:
        min_values = np.min(weight, axis=reduction_axes, keepdims=True)  # [a1, r, a2] -> [a1, 1, a2]
        max_values = np.max(weight, axis=reduction_axes, keepdims=True)  # [a1, r, a2] -> [a1, 1, a2]
        scale, zero_point = calculate_scale_zero_point(
            min_values, max_values, level_low, level_high, narrow_range=False
        )
    else:
        scale = np.max(np.abs(weight), axis=reduction_axes, keepdims=True)  # [a1, r//gs, 1, a2]
        level_low_sym = -(2 ** (num_bits - 1))
        level_high_sym = 2 ** (num_bits - 1) - 1
        scale = scale / level_high_sym
        zero_point = np.array([-level_low_sym])

    eps = np.finfo(weight.dtype).eps
    # NOTE: adding machine epsilon to avoid division by zero
    scale[np.abs(scale) < eps] = eps
    compressed_weights = np.round(weight / scale + zero_point)
    compressed_weights = np.clip(compressed_weights, level_low, level_high).astype(np.uint8)
    return compressed_weights, scale, zero_point


def _get_integer_quantization_error(
    weight: np.ndarray, reduction_axes: Union[int, Tuple[int]], config: WeightCompressionConfig
) -> float:
    """
    Calculates a quantity characterizing the difference between floating point weights and fake quantized
    (compressed and decompressed) to integer ones.

    :param weight: Weight array to compress.
    :param reduction_axes: Axis or axes along which to reduce (collect) different statistics (e.g. min, max).
    :param config: Information on how to compress (quantize) a specific weight.
    :return: The quantity characterizing the error of integer quantization.
    """
    orig_shape = weight.shape
    compressed_weights, scale, zero_point = _do_integer_quantization(weight, reduction_axes, config)

    decompressed_weight = compressed_weights.astype(dtype=scale.dtype)
    decompressed_weight = (compressed_weights - zero_point) * scale

    decompressed_weight = decompressed_weight.reshape(orig_shape)
    diff = (decompressed_weight - weight) ** 2
    layer_err = np.mean(diff, axis=reduction_axes)
    val = np.max(layer_err)
    return val


def _reshape_weights_for_grouped_quantization(
    weight: np.ndarray, reduction_axes: Union[int, Tuple[int]], group_size: int
) -> Tuple[np.ndarray, int]:
    """
    Reshapes weights for group-wise quantization and return a new reduction axis for collecting statistics per group
    dimension. Having weights with shapes [c_out, c_in] and group size = 128, shape of reshaped weights is
    [c_out, c_in // 128, 128].

    :param weight: Weight array to compress.
    :param reduction_axes: Axis or axes along which to reduce (collect) different statistics (e.g. min, max).
    :param group_size: Number of weights (e.g. 128) in the channel dimension that share quantization parameters (scale).
    :return: reshaped weights and new reduction axis.
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
    reshaped_weight = weight.reshape(shape)
    reduction_axis += 1
    return reshaped_weight, reduction_axis


def _get_norm_weight_and_nf4_scale(
    weight: np.ndarray, reduction_axes: Tuple[int], group_size: int = -1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates scale for nf4 quantization and normalizes weights by the scale.
    Weights are reshaped in case of positive value of group size.

    :param weight: Weight array to compress.
    :param reduction_axes: Axis or axes along which to reduce (collect) different statistics (e.g. min, max).
    :param group_size: Number of weights (e.g. 128) in the channel dimension that share quantization parameters (scale).
        The value -1 means no grouping. Defaults to -1.
    :return: Normalized weights and nf4 scale.
    """
    if group_size != -1:
        # weights are reshaped: [a1, r, a2] -> [a1, r//gs, gs, a2]
        weight, reduction_axis = _reshape_weights_for_grouped_quantization(weight, reduction_axes, group_size)
        scale = np.max(np.abs(weight), axis=reduction_axis, keepdims=True)  # [a1, r//gs, 1, a2]
    else:
        scale = np.max(np.abs(weight), axis=reduction_axes, keepdims=True)  # [a1, 1, a2]
    eps = np.finfo(weight.dtype).eps
    # NOTE: adding machine epsilon to avoid division by zero
    scale[np.abs(scale) < eps] = eps
    norm_weight = weight / scale
    return norm_weight, scale


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


def _assign_mixed_precision(
    all_weight_params: List[WeightNodeParams], ratio: float, primary_config: WeightCompressionConfig
) -> None:
    """
    Assigns mixed quantization scheme (e.g. uniform int8 or non-uniform nf4) for weights based on some criteria.

    :param all_weight_params: List of information about each weight node. The quantization scheme is added to this info.
    :param ratio: The ratio between primary and backup precisions (e.g. 0.9 means 90% of layers quantized to NF4
        and the rest to INT8).
    :param primary_config: Information on how to compress (quantize) weights to primary precision.
    :return: None.
    """
    if ratio == 1:
        for weight_param in all_weight_params[1:-1]:
            weight_param.compression_config = primary_config
        return
    errors = []
    num_internal_weights = 0
    # NOTE: first and last layers are always in 8 bit: no need to calculate error for them
    for weight_param in track(all_weight_params[1:-1], description="Searching for Mixed-Precision Configuration"):
        weight = get_const_value(weight_param.weight_node)
        backup_config = weight_param.compression_config
        reduction_axes = weight_param.reduction_axes
        backup_error = _get_integer_quantization_error(weight, reduction_axes, backup_config)
        eps = np.finfo(weight.dtype).eps
        error = 1 / (backup_error + eps)
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
        weight_param.compression_config = primary_config
        num_weights_in_4bit += weight_param.num_weights
