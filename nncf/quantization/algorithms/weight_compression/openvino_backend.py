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
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple, TypeVar

import numpy as np
import openvino.runtime as ov
from openvino.runtime import opset13 as opset

from nncf.common.factory import ModelTransformerFactory
from nncf.common.graph import NNCFNode
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph_matching import find_subgraphs_matching_pattern
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.logging import nncf_logger
from nncf.common.logging.track_progress import track
from nncf.common.tensor_statistics.statistic_point import StatisticPoint
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.utils.helpers import create_table
from nncf.data import Dataset
from nncf.experimental.common.tensor_statistics.collectors import NoopAggregator
from nncf.openvino.graph.metatypes.openvino_metatypes import OVEmbeddingMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVMatMulMetatype
from nncf.openvino.graph.node_utils import get_channel_agnostic_reduction_axes
from nncf.openvino.graph.node_utils import get_const_value
from nncf.openvino.graph.node_utils import get_weight_channel_axes
from nncf.openvino.graph.transformations.commands import OVTargetPoint
from nncf.openvino.graph.transformations.commands import TargetType
from nncf.openvino.rt_info import dump_parameters
from nncf.openvino.statistics.aggregator import OVStatisticsAggregator
from nncf.openvino.statistics.collectors import OVMeanPerChanelReducer
from nncf.openvino.statistics.collectors import OVMeanTensorStatistic
from nncf.openvino.statistics.collectors import OVRawTensorStatistic
from nncf.openvino.statistics.collectors import TensorCollector
from nncf.parameters import CompressWeightsMode
from nncf.quantization.algorithms.smooth_quant.openvino_backend import OVSmoothQuantAlgoBackend
from nncf.quantization.algorithms.weight_compression.awq_patterns import get_awq_patterns
from nncf.quantization.algorithms.weight_compression.backend import WeightCompressionAlgoBackend
from nncf.quantization.fake_quantize import calculate_scale_zero_point
from nncf.quantization.passes import transform_to_inference_graph
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
        all_layers: Optional[bool] = False,
        graph: NNCFGraph = None,
        dataset: Dataset = None,
    ) -> ov.Model:
        all_weight_params: List[WeightNodeParams] = []
        quantized_nodes_ids = set()

        friendly_name_to_op_map = {op.get_friendly_name(): op for op in model.get_ops()}

        is_last_layer_shared = False
        n = len(nodes_to_compress)
        for i, nncf_node in enumerate(nodes_to_compress):
            weight_port_ids = nncf_node.layer_attributes.get_const_port_ids()
            for weight_port_id in weight_port_ids:
                weight_op_friendly_name = nncf_node.layer_attributes.constant_attributes[weight_port_id]["name"]
                weight_node = friendly_name_to_op_map[weight_op_friendly_name]
                if weight_node is None:
                    continue
                if id(weight_node) in quantized_nodes_ids:
                    if i == n - 1:
                        is_last_layer_shared = True
                    continue
                weight_output = weight_node.output(0)

                original_weight_dtype = weight_output.get_element_type().to_dtype()
                if original_weight_dtype not in [np.float32, np.float16, np.float64]:
                    continue
                const_shape = nncf_node.layer_attributes.constant_attributes[weight_port_id]["shape"]
                channel_axes = get_weight_channel_axes(nncf_node)
                reduction_axes = get_channel_agnostic_reduction_axes(channel_axes, const_shape)
                if isinstance(reduction_axes, tuple) and len(reduction_axes) != 1:
                    nncf_logger.warning(
                        f"Weight compression expects a single reduction axes, but given {len(reduction_axes)}. "
                        f"Weight shape: {const_shape}, reduction axes: {reduction_axes}, "
                        f"node name: {nncf_node.node_name}. The node won't be quantized."
                    )
                    continue
                reduction_axis = reduction_axes[0] if isinstance(reduction_axes, tuple) else reduction_axes

                fq_name = f"{weight_op_friendly_name}/fq_weights_{weight_port_id}"
                num_weights = np.prod(const_shape)
                weight_params = WeightNodeParams(
                    reduction_axis,
                    num_weights,
                    fq_name,
                    weight_node,
                    original_weight_dtype,
                    metatype=nncf_node.metatype,
                )
                all_weight_params.append(weight_params)
                quantized_nodes_ids.add(id(weight_node))

        internal_weight_params = _get_internal_weight_params(all_weight_params, mode, is_last_layer_shared, all_layers)
        _set_weight_compression_config(internal_weight_params, mode, ratio, group_size)
        nncf_logger.info(_get_bitwidth_distribution_str(all_weight_params, internal_weight_params))

        if dataset is not None:
            model = _apply_AWQ(model, graph, all_weight_params, nodes_to_compress, dataset)

        for wp in track(all_weight_params, description="Applying Weight Compression"):
            weight_node = wp.weight_node
            original_weight_dtype = wp.original_weight_dtype

            weight_output = weight_node.output(0)
            weight_name = weight_node.get_friendly_name()
            target_inputs = weight_output.get_target_inputs()

            weight = get_const_value(weight_node)
            config = wp.compression_config
            original_shape = weight.shape
            if config.mode == CompressWeightsMode.NF4:
                norm_weight, scale = _get_norm_weight_and_nf4_scale(weight, wp.reduction_axis, group_size)
                compressed_const = opset.constant(norm_weight, dtype=ov.Type.nf4, name=weight_name)
                convert = opset.convert(compressed_const, original_weight_dtype)
                mul = opset.multiply(convert, scale.astype(original_weight_dtype), name=wp.fq_name)
            else:
                compressed_weights, scale, zero_point = _do_integer_quantization(weight, wp.reduction_axis, config)
                compression_type = ov.Type.u8 if config.num_bits == 8 else ov.Type.u4
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

        dump_parameters(
            model,
            parameters={
                "mode": mode.value,
                "group_size": group_size,
                "ratio": ratio,
            },
            algo_name="weight_compression",
        )
        return model


TWeightType = TypeVar("TWeightType")


@dataclass
class WeightCompressionConfig:
    """
    Information on how to compress (quantize) a specific weight.

    :param mode: Defines a mode for weight compression. Defaults to INT8_ASYM mode.
    :param group_size: Number of weights (e.g. 128) in the channel dimension that share quantization parameters (scale).
        The value -1 means no grouping. Defaults to -1.
    """

    mode: Optional[CompressWeightsMode] = CompressWeightsMode.INT8_ASYM
    group_size: Optional[int] = -1

    @property
    def num_bits(self):
        """
        :return: number of bits that is used for storing a single quantized value in the given mode.
        """
        return 8 if self.mode in [CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM] else 4


@dataclass
class WeightNodeParams:
    """
    Information about weight node in the ov.Model that is useful for weight compression.

    :param reduction_axis: Axis, along which to reduce (collect) different statistics (e.g. min, max).
    :param num_weights: Number of elements in the weight array.
    :param fq_name: Name for the inserted weight compression operation.
    :param weight_node: The weight node itself.
    :param original_weight_dtype: Type of elements in the weight array.
    :param compression_config: Configuration of weight compression for the weight node.
    :param metatype: Metatype of the corresponding operation with weight.
    """

    reduction_axis: int
    num_weights: int
    fq_name: str
    weight_node: ov.Node
    original_weight_dtype: TWeightType
    compression_config = WeightCompressionConfig()
    metatype: OperatorMetatype = None


def _do_integer_quantization(
    weight: np.ndarray, reduction_axis: int, config: WeightCompressionConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The method quantizes the given weights to integer data type in accordance with the compression config.
    The config defines a quantization mode:
        INT8_SYM mode refers to unsigned int8 symmetric weight compression with a fixed zero point equals to 128 -
            quantization to [0, 255] range.
        INT8_ASYM mode refers to unsigned int8 asymmetric weight compression with a typical non-fixed zero-point -
            quantization to [0, 255] range.
        INT4_ASYM mode refers to unsigned int4 asymmetric weight compression with a typical non-fixed zero-point -
            quantization to [0, 15] range.
        INT4_SYM mode refers to unsigned int4 symmetric weight compression with a fixed zero point equals to 8 -
            quantization to [0, 15] range.
        NF4 mode requires a dedicated procedure and it is not supported in this method.
    One of the parameter of compression config is a group size. Quantization is per-channel, if group size equals to -1,
    otherwise it's per-group, i.e. group size number of weights in the channel dimension share quantization parameters
    (scales).

    :param weight: Weight array to compress.
    :param reduction_axis: Axis, along which to reduce (collect) different statistics (e.g. min, max).
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
        weight, reduction_axis = _reshape_weights_for_grouped_quantization(weight, reduction_axis, group_size)

    if mode in [CompressWeightsMode.INT8_ASYM, CompressWeightsMode.INT4_ASYM]:
        min_values = np.min(weight, axis=reduction_axis, keepdims=True)  # [a1, r, a2] -> [a1, 1, a2]
        max_values = np.max(weight, axis=reduction_axis, keepdims=True)  # [a1, r, a2] -> [a1, 1, a2]
        scale, zero_point = calculate_scale_zero_point(
            min_values, max_values, level_low, level_high, narrow_range=False
        )
    else:
        scale = np.max(np.abs(weight), axis=reduction_axis, keepdims=True)  # [a1, r//gs, 1, a2]
        level_low_sym = -(2 ** (num_bits - 1))
        level_high_sym = 2 ** (num_bits - 1) - 1
        scale = scale / level_high_sym
        zero_point = np.array([-level_low_sym]).astype(np.int8)

    eps = np.finfo(weight.dtype).eps
    # NOTE: adding machine epsilon to avoid division by zero
    scale[np.abs(scale) < eps] = eps
    compressed_weights = np.round(weight / scale + zero_point)
    compressed_weights = np.clip(compressed_weights, level_low, level_high).astype(np.uint8)
    return compressed_weights, scale, zero_point


def _get_integer_quantization_error(weight: np.ndarray, reduction_axis: int, config: WeightCompressionConfig) -> float:
    """
    Calculates a quantity characterizing the difference between floating point weights and fake quantized
    (compressed and decompressed) to integer ones.

    :param weight: Weight array to compress.
    :param reduction_axis: Axis, along which to reduce (collect) different statistics (e.g. min, max).
    :param config: Information on how to compress (quantize) a specific weight.
    :return: The quantity characterizing the error of integer quantization.
    """
    orig_shape = weight.shape
    compressed_weights, scale, zero_point = _do_integer_quantization(weight, reduction_axis, config)

    decompressed_weight = compressed_weights.astype(dtype=scale.dtype)
    decompressed_weight = (compressed_weights - zero_point) * scale

    decompressed_weight = decompressed_weight.reshape(orig_shape)
    diff = (decompressed_weight - weight) ** 2
    layer_err = np.mean(diff, axis=reduction_axis)
    val = np.max(layer_err)
    return val


def _reshape_weights_for_grouped_quantization(
    weight: np.ndarray, reduction_axis: int, group_size: int
) -> Tuple[np.ndarray, int]:
    """
    Reshapes weights for group-wise quantization and return a new reduction axis for collecting statistics per group
    dimension. Having weights with shapes [c_out, c_in] and group size = 128, shape of reshaped weights is
    [c_out, c_in // 128, 128].

    :param weight: Weight array to compress.
    :param reduction_axis: Axis, along which to reduce (collect) different statistics (e.g. min, max).
    :param group_size: Number of weights (e.g. 128) in the channel dimension that share quantization parameters (scale).
    :return: reshaped weights and new reduction axis.
    """
    assert group_size != -1
    assert isinstance(reduction_axis, int)
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
    weight: np.ndarray, reduction_axis: int, group_size: int = -1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates scale for nf4 quantization and normalizes weights by the scale.
    Weights are reshaped in case of positive value of group size.

    :param weight: Weight array to compress.
    :param reduction_axis: Axis, along which to reduce (collect) different statistics (e.g. min, max).
    :param group_size: Number of weights (e.g. 128) in the channel dimension that share quantization parameters (scale).
        The value -1 means no grouping. Defaults to -1.
    :return: Normalized weights and nf4 scale.
    """
    if group_size != -1:
        # weights are reshaped: [a1, r, a2] -> [a1, r//gs, gs, a2]
        weight, reduction_axis = _reshape_weights_for_grouped_quantization(weight, reduction_axis, group_size)
        scale = np.max(np.abs(weight), axis=reduction_axis, keepdims=True)  # [a1, r//gs, 1, a2]
    else:
        scale = np.max(np.abs(weight), axis=reduction_axis, keepdims=True)  # [a1, 1, a2]
    eps = np.finfo(weight.dtype).eps
    # NOTE: adding machine epsilon to avoid division by zero
    scale[np.abs(scale) < eps] = eps
    norm_weight = weight / scale
    return norm_weight, scale


def _proportion_str(num_weights_list: List[int], total_num_weights: int, total_num_params: int) -> str:
    percentage = sum(num_weights_list) / max(total_num_weights, 1) * 100
    return f"{percentage:.0f}% ({len(num_weights_list)} / {total_num_params})"


def _get_bitwidth_distribution_str(all_params: List[WeightNodeParams], internal_params: List[WeightNodeParams]) -> str:
    """
    Generates a table that shows the ratio of weights quantized to different number of bits.

    :param all_params: List of information about each weight node.
    :param internal_params: List of information about weight nodes that are considered for mixed precision.
    :return: A string containing the table.
    """
    num_bits_vs_num_weights_map = {}
    internal_fq_names = set(wp.fq_name for wp in internal_params)
    for data in all_params:
        num_bits = data.compression_config.num_bits
        n_total, n_internal = num_bits_vs_num_weights_map.get(num_bits, ([], []))
        if data.fq_name in internal_fq_names:
            n_internal.append(data.num_weights)
        n_total.append(data.num_weights)
        num_bits_vs_num_weights_map[num_bits] = (n_total, n_internal)

    num_internal_weights = sum(ws.num_weights for ws in internal_params)
    num_internal_params = len(internal_params)
    num_total_weights = sum(ws.num_weights for ws in all_params)
    num_params = len(all_params)
    num_bits_vs_num_weights_map = OrderedDict(sorted(num_bits_vs_num_weights_map.items(), reverse=True))
    # Table creation
    header = ["Num bits (N)", "% all parameters (layers)", "% internal parameters (layers)"]
    rows = []
    for bitwidth, (n_total, n_internal) in num_bits_vs_num_weights_map.items():
        rows.append(
            [
                bitwidth,
                _proportion_str(n_total, num_total_weights, num_params),
                _proportion_str(n_internal, num_internal_weights, num_internal_params),
            ]
        )

    table = create_table(header, rows)
    pretty_string = f"Statistics of the bitwidth distribution:\n{table}"
    return pretty_string


def _get_internal_weight_params(
    all_weight_params: List[WeightNodeParams],
    mode: CompressWeightsMode,
    is_last_layer_shared: bool,
    all_layers: bool,
) -> List[WeightNodeParams]:
    """
    Returns the internal weight parameters.

    :param all_weight_params: List of all weight parameters.
    :param mode: Weight compression mode.
    :param is_last_layer_shared: Indicates whether the last layer shares the weight to be quantized.
    :param all_layers: Indicates whether embeddings and last layers should be compressed to a primary
        precision. By default, the backup precision is assigned for the embeddings and last layers.
    :return: List of information about weight nodes that are considered for mixed precision.
    """
    internal_weight_params = all_weight_params
    if mode not in [CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM] and not all_layers:
        internal_weight_params = list(filter(lambda wp: wp.metatype != OVEmbeddingMetatype, internal_weight_params))
        if not is_last_layer_shared:
            internal_weight_params = internal_weight_params[:-1]
    return internal_weight_params


def _assign_mixed_precision(
    internal_weight_params: List[WeightNodeParams], ratio: float, primary_config: WeightCompressionConfig
) -> None:
    """
    Assigns mixed quantization scheme (e.g. uniform int8 or non-uniform nf4) for weights based on some criteria.
    :param internal_weight_params: List of information about internal weight nodes. Only internal nodes are considered
        for mixed precision. The quantization scheme is added to this info.
    :param ratio: The ratio between primary and backup precisions (e.g. 0.9 means 90% of layers quantized to NF4
        and the rest to INT8_ASYM).
    :param primary_config: Information on how to compress (quantize) weights to primary precision.
    :return: None.
    """
    errors = []
    num_internal_weights = 0
    for weight_param in track(internal_weight_params, description="Searching for Mixed-Precision Configuration"):
        weight = get_const_value(weight_param.weight_node)
        backup_config = weight_param.compression_config
        reduction_axis = weight_param.reduction_axis
        backup_error = _get_integer_quantization_error(weight, reduction_axis, backup_config)
        eps = np.finfo(weight.dtype).eps
        error = 1 / (backup_error + eps)
        errors.append(error)
        num_internal_weights += weight_param.num_weights
    indexes_of_layers_in_ascending_order_of_errors = [
        i[0] for i in sorted(enumerate(errors), reverse=False, key=lambda x: x[1])
    ]
    num_weights_in_4bit = 0
    for index in indexes_of_layers_in_ascending_order_of_errors:
        weight_param = internal_weight_params[index]
        current_ratio = (num_weights_in_4bit + weight_param.num_weights) / num_internal_weights
        if current_ratio >= ratio:
            break
        weight_param.compression_config = primary_config
        num_weights_in_4bit += weight_param.num_weights


def _set_weight_compression_config(
    internal_weight_params: List[WeightNodeParams], mode: CompressWeightsMode, ratio: float, group_size: int
) -> None:
    """
    Set the appropriate compression configuration for weights based on some criteria.

    :param internal_weight_params: List of information about internal weight nodes.
    :param mode: Weight compression mode.
    :param ratio: The ratio between primary and backup precisions.
    :param group_size: number of weights (e.g. 128) in the channel dimension that share quantization parameters (scale).
    :return: None.
    """
    primary_config = WeightCompressionConfig(mode=mode, group_size=group_size)
    if ratio == 1:
        for weight_param in internal_weight_params:
            weight_param.compression_config = primary_config
    else:
        _assign_mixed_precision(internal_weight_params, ratio, primary_config)


def _get_mean_statistic_collector(num_samples: int, channel_axis: int, inplace: bool = True):
    """
    Raw statistic collector builder.

    :param num_samples: Maximum number of samples to collect.
    :param channel_axis: Channel axis to use during reduction phase.
    :param window_size: Number of samples from the end of the list of collected samples to aggregate.
        Aggregates all available collected statistics in case parameter is None.
    :param inplace: Whether the mean reducer should be calculated inplace or out of place.
    :return: Mean statistic collector.
    """
    reducer = OVMeanPerChanelReducer(channel_axis=channel_axis, inplace=inplace)

    aggregate_mean = NoopAggregator(num_samples)

    collector = TensorCollector(OVRawTensorStatistic)
    collector.register_statistic_branch(OVMeanTensorStatistic.MEAN_STAT, reducer, aggregate_mean)
    return collector


def _get_statistic_points(nodes_to_compress, num_samples=32, algorithm="AWQ") -> StatisticPointsContainer:
    """
    Returns statistic points, for which StatisticsCollector should collect statistics.
    :return: Statistic points, for which StatisticsCollector should collect statistics.
    """
    statistic_container = StatisticPointsContainer()
    INPUT_PORT_OF_NODE = 0

    for node in nodes_to_compress:
        statistic_point_in = OVTargetPoint(TargetType.PRE_LAYER_OPERATION, node.node_name, port_id=INPUT_PORT_OF_NODE)

        channel_axis = node.metatype.output_channel_axis
        if channel_axis is None:
            channel_axis = -1

        stat_collector_in = _get_mean_statistic_collector(
            channel_axis=channel_axis, num_samples=num_samples, inplace=False
        )

        statistic_container.add_statistic_point(
            StatisticPoint(target_point=statistic_point_in, tensor_collector=stat_collector_in, algorithm=algorithm)
        )

    return statistic_container


def _decompress(compressed_weights, scale, zero_point):
    decompressed_weights = (compressed_weights - zero_point) * scale
    return decompressed_weights


def _apply_AWQ(
    model: ov.Model,
    graph: NNCFGraph,
    all_weight_params: List[WeightNodeParams],
    nodes_to_compress: List[NNCFNode],
    dataset: Dataset,
    subset_size: int = 32,
    percent_to_apply=0.002,
    alpha_min=0.01,
    alpha_max=1.0,
    steps=100,
):
    matches = []

    inference_nncf_graph = transform_to_inference_graph(deepcopy(graph), [], [])
    nx_graph = inference_nncf_graph.get_nx_graph_copy()
    for _, pattern_graph in get_awq_patterns().items():
        matches.extend(find_subgraphs_matching_pattern(nx_graph, pattern_graph(), strict=False))

    if len(matches) == 0:
        return model

    @dataclass
    class AWQTriplet:
        """
        Information on how to compress (quantize) a specific weight.

        :param mode: Defines a mode for weight compression. Defaults to INT8_ASYM mode.
        :param group_size: Number of weights (e.g. 128) in the channel dimension that share quantization parameters (scale).
            The value -1 means no grouping. Defaults to -1.
        """

        weight_params: WeightNodeParams = None
        target_node: NNCFNode = None
        merge_node: NNCFNode = None

    target_node_names = []
    merge_node_names = []
    awq_data = {}
    name_mapping = {wp.weight_node.get_friendly_name(): idx for idx, wp in enumerate(all_weight_params)}

    for match in matches:
        nncf_node = graph.get_node_by_key(match[-1])
        weight_port_ids = nncf_node.layer_attributes.get_const_port_ids()
        for weight_port_id in weight_port_ids:
            weight_op_friendly_name = nncf_node.layer_attributes.constant_attributes[weight_port_id]["name"]
            target_node_names.append(weight_op_friendly_name)

        nncf_node = graph.get_node_by_key(match[0])
        weight_port_ids = nncf_node.layer_attributes.get_const_port_ids()
        for weight_port_id in weight_port_ids:
            weight_op_friendly_name = nncf_node.layer_attributes.constant_attributes[weight_port_id]["name"]
            merge_node_names.append(weight_op_friendly_name)

        assert len(target_node_names) == len(merge_node_names)
        weight_params = all_weight_params[name_mapping[target_node_names[-1]]]
        target_node = nodes_to_compress[name_mapping[target_node_names[-1]]]
        merge_node = nodes_to_compress[name_mapping[merge_node_names[-1]]]

        awq_data[target_node.node_name] = AWQTriplet(weight_params, target_node, merge_node)

    nodes_for_stats = [v.target_node for _, v in awq_data.items()]
    statistic_points = _get_statistic_points(nodes_for_stats, subset_size)

    statistics_aggregator = OVStatisticsAggregator(dataset)
    statistics_aggregator.register_statistic_points(statistic_points)
    statistics_aggregator.collect_statistics(model, graph)

    alpha_step = (alpha_max - alpha_min) / steps

    model_transformer = ModelTransformerFactory.create(model, True)
    transformation_layout = TransformationLayout()

    for k, v in track(statistics_aggregator.statistic_points.items(), description="Applying AWQ"):
        stats = list(v[0].algorithm_to_tensor_collectors["AWQ"][0].aggregators.values())[0]._container
        stats = [stat.squeeze() for stat in stats]

        awq_data_item = awq_data[k]
        wp = awq_data_item.weight_params
        target_node = awq_data_item.target_node
        merge_node = awq_data_item.merge_node

        config = wp.compression_config

        X = np.vstack(stats).transpose()
        s = np.max(np.abs(X), axis=1)

        top_k = max(int(s.shape[0] * percent_to_apply), 1)
        topk_idxs = (-s).argsort()[:top_k]

        groups_to_correct = set()
        for idx in topk_idxs:
            groups_to_correct.add(idx // config.group_size)

        groups_to_correct = list(groups_to_correct)

        weight = get_const_value(wp.weight_node)
        reduction_axis = wp.reduction_axis

        if reduction_axis == 0:
            weight = weight.transpose()
            reduction_axis = 1

        scale_shape = weight.shape[reduction_axis]

        scale = np.ones(scale_shape).astype(np.float32)

        awq_config = deepcopy(config)
        awq_config.group_size = -1

        for gi in groups_to_correct:
            offset = gi * config.group_size
            gscale = s[offset : offset + config.group_size]

            a_min = np.quantile(gscale, 0.1)
            a_max = 1e2
            gscale = np.clip(gscale, a_min=a_min, a_max=a_max)

            gweight = weight[:, offset : offset + config.group_size].copy()
            gacts = X[offset : offset + config.group_size, :].copy()

            fp32_out = np.matmul(gweight, gacts)
            min_diff = np.max(fp32_out)
            best_scale = None

            alpha = alpha_min
            for _ in range(steps):
                cur_scale = gscale**alpha

                g_compressed_weighs, g_c_scale, g_c_zp = _do_integer_quantization(
                    gweight * cur_scale, reduction_axis, awq_config
                )
                g_decompressed_weighs = _decompress(g_compressed_weighs, g_c_scale, g_c_zp)
                sacts = gacts / np.expand_dims(cur_scale, 1)

                cur_out = np.matmul(g_decompressed_weighs, sacts)
                cur_diff = np.mean(np.abs(cur_out - fp32_out))
                if cur_diff < min_diff:
                    min_diff = cur_diff
                    best_scale = cur_scale
                alpha += alpha_step

            if best_scale is not None:
                scale[offset : offset + config.group_size] = best_scale

        a_scale = scale
        w_scale = scale
        if wp.reduction_axis == 0:
            w_scale = np.expand_dims(w_scale, 1)
            a_scale = np.expand_dims(1.0 / a_scale, 0)
        else:
            w_scale = np.expand_dims(w_scale, 0)
            a_scale = np.expand_dims(1.0 / a_scale, 1)

        weight_port = OVSmoothQuantAlgoBackend.get_weight_tensor_port_id(target_node)
        weight_value = OVSmoothQuantAlgoBackend.get_weight_value(target_node, model, weight_port)
        scaled_weight = weight_value * w_scale
        weight_update_command = OVSmoothQuantAlgoBackend.weight_update_command(target_node, scaled_weight, weight_port)

        transformation_layout.register(weight_update_command)

        weight_port = OVSmoothQuantAlgoBackend.get_weight_tensor_port_id(merge_node)
        weight_value = OVSmoothQuantAlgoBackend.get_weight_value(merge_node, model, weight_port)
        scaled_weight = weight_value * a_scale
        weight_update_command = OVSmoothQuantAlgoBackend.weight_update_command(merge_node, scaled_weight, weight_port)

        transformation_layout.register(weight_update_command)

    model = model_transformer.transform(transformation_layout)

    friendly_name_to_op_map = {op.get_friendly_name(): op for op in model.get_ops()}
    for wp in all_weight_params:
        name = wp.fq_name
        idx = name.find("fq_weights_")
        name = name[: idx - 1]
        wp.weight_node = friendly_name_to_op_map[name]

    return model
