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
from typing import Dict, List, Optional

import numpy as np
import openvino.runtime as ov
from openvino.runtime import opset13 as opset

from nncf.common.factory import ModelTransformerFactory
from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.graph_matching import find_subgraphs_matching_pattern
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.logging import nncf_logger
from nncf.common.logging.track_progress import track
from nncf.common.utils.helpers import create_table
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.openvino.graph.metatypes.openvino_metatypes import OVEmbeddingMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVMatMulMetatype
from nncf.openvino.graph.node_utils import get_channel_agnostic_reduction_axes
from nncf.openvino.graph.node_utils import get_const_value
from nncf.openvino.graph.node_utils import get_weight_channel_axes
from nncf.openvino.graph.transformations.commands import OVTargetPoint
from nncf.openvino.rt_info import dump_parameters
from nncf.openvino.statistics.collectors import get_raw_stat_collector
from nncf.parameters import CompressWeightsMode
from nncf.parameters import SensitivityMetric
from nncf.quantization.algorithms.smooth_quant.openvino_backend import OVSmoothQuantAlgoBackend
from nncf.quantization.algorithms.weight_compression.awq_patterns import get_awq_patterns
from nncf.quantization.algorithms.weight_compression.backend import WeightCompressionAlgoBackend
from nncf.quantization.algorithms.weight_compression.compression_info import WeightCompressionConfig
from nncf.quantization.algorithms.weight_compression.compression_info import WeightNodeParams
from nncf.quantization.algorithms.weight_compression.mixed_precision import MIXED_PRECISION_CRITERIA
from nncf.quantization.algorithms.weight_compression.quantize import do_dequantization
from nncf.quantization.algorithms.weight_compression.quantize import do_integer_quantization
from nncf.quantization.algorithms.weight_compression.quantize import get_norm_weight_and_nf4_scale
from nncf.quantization.passes import transform_to_inference_graph
from nncf.scopes import IgnoredScope


class OVWeightCompressionAlgoBackend(WeightCompressionAlgoBackend):
    @property
    def matmul_metatypes(self) -> List[OperatorMetatype]:
        return [OVMatMulMetatype]

    @property
    def embedding_metatypes(self) -> List[OperatorMetatype]:
        return [OVEmbeddingMetatype]

    @staticmethod
    def is_node_with_weights(node: NNCFNode) -> bool:
        return node.layer_attributes and node.layer_attributes.constant_attributes

    @staticmethod
    def validate_params(mode: CompressWeightsMode, ignored_scope: Optional[IgnoredScope] = None) -> None:
        pass

    @staticmethod
    def target_point(target_type: TargetType, target_node_name: str, port_id: int) -> OVTargetPoint:
        return OVTargetPoint(target_type, target_node_name, port_id)

    @staticmethod
    def raw_statistic_collector(inplace: bool, num_samples: int = None) -> TensorCollector:
        return get_raw_stat_collector(num_samples, inplace)

    @staticmethod
    def get_activation_port_id(node: NNCFNode, nncf_graph: NNCFGraph) -> int:
        constant_ports = node.layer_attributes.get_const_port_ids()
        activation_ports = [
            e.input_port_id for e in nncf_graph.get_input_edges(node) if e.input_port_id not in constant_ports
        ]
        assert len(activation_ports) == 1
        return activation_ports[0]

    @staticmethod
    def do_compression(
        model: ov.Model,
        graph: NNCFGraph,
        nodes_to_compress: List[NNCFNode],
        mode: CompressWeightsMode,
        ratio: float = None,
        group_size: int = None,
        all_layers: Optional[bool] = False,
        activations: Optional[Dict[str, np.ndarray]] = None,
        sensitivity_metric: Optional[SensitivityMetric] = SensitivityMetric.WEIGHT_QUANTIZATION_ERROR,
        use_awq: Optional[bool] = False,
    ) -> ov.Model:
        all_weight_params: List[WeightNodeParams] = []
        quantized_nodes_ids = set()
        friendly_name_to_op_map = {op.get_friendly_name(): op for op in model.get_ops()}
        is_last_layer_shared = False
        n = len(nodes_to_compress)
        for i, nncf_node in enumerate(nodes_to_compress):
            node_name = nncf_node.node_name
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
                        f"node name: {node_name}. The node won't be quantized."
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
                    node_name=node_name,
                )
                all_weight_params.append(weight_params)
                quantized_nodes_ids.add(id(weight_node))

        ratio_defining_params = _get_ratio_defining_params(all_weight_params, mode, is_last_layer_shared, all_layers)
        _set_weight_compression_config(ratio_defining_params, mode, ratio, group_size, activations, sensitivity_metric)
        nncf_logger.info(_get_bitwidth_distribution_str(all_weight_params, ratio_defining_params))

        if use_awq and activations is not None:
            apply_AWQ(model, graph, all_weight_params, nodes_to_compress, activations)

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
                norm_weight, scale = get_norm_weight_and_nf4_scale(weight, wp.reduction_axis, group_size)
                compressed_const = opset.constant(norm_weight, dtype=ov.Type.nf4, name=weight_name)
                convert = opset.convert(compressed_const, original_weight_dtype)
                mul = opset.multiply(convert, scale.astype(original_weight_dtype), name=wp.fq_name)
            else:
                compressed_weights, scale, zero_point = do_integer_quantization(weight, wp.reduction_axis, config)
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
                "all_layers": all_layers,
                "sensitivity_metric": sensitivity_metric.value,
            },
            algo_name="weight_compression",
        )
        return model


def _proportion_str(num_weights_list: List[int], total_num_weights: int, total_num_params: int) -> str:
    percentage = sum(num_weights_list) / max(total_num_weights, 1) * 100
    return f"{percentage:.0f}% ({len(num_weights_list)} / {total_num_params})"


def _get_bitwidth_distribution_str(
    all_params: List[WeightNodeParams], ratio_defining_params: List[WeightNodeParams]
) -> str:
    """
    Generates a table that shows the ratio of weights quantized to different number of bits.

    :param all_params: Information about each weight node.
    :param ratio_defining_params: Information about weights that are used for calculating ratio between primary and
        backup precisions.
    :return: A string containing the table.
    """
    num_bits_vs_num_weights_map = {}
    ratio_defining_fq_names = set(wp.fq_name for wp in ratio_defining_params)
    for data in all_params:
        num_bits = data.compression_config.num_bits
        n_total, n_ratio_defining = num_bits_vs_num_weights_map.get(num_bits, ([], []))
        if data.fq_name in ratio_defining_fq_names:
            n_ratio_defining.append(data.num_weights)
        n_total.append(data.num_weights)
        num_bits_vs_num_weights_map[num_bits] = (n_total, n_ratio_defining)

    num_ratio_defining_weights = sum(ws.num_weights for ws in ratio_defining_params)
    num_ratio_defining_params = len(ratio_defining_params)
    num_total_weights = sum(ws.num_weights for ws in all_params)
    num_params = len(all_params)
    num_bits_vs_num_weights_map = OrderedDict(sorted(num_bits_vs_num_weights_map.items(), reverse=True))
    # Table creation
    header = ["Num bits (N)", "% all parameters (layers)", "% ratio-defining parameters (layers)"]
    rows = []
    for bitwidth, (n_total, n_ratio_defining) in num_bits_vs_num_weights_map.items():
        rows.append(
            [
                bitwidth,
                _proportion_str(n_total, num_total_weights, num_params),
                _proportion_str(n_ratio_defining, num_ratio_defining_weights, num_ratio_defining_params),
            ]
        )

    table = create_table(header, rows)
    pretty_string = f"Statistics of the bitwidth distribution:\n{table}"
    return pretty_string


def _get_ratio_defining_params(
    all_weight_params: List[WeightNodeParams],
    mode: CompressWeightsMode,
    is_last_layer_shared: bool,
    all_layers: bool,
) -> List[WeightNodeParams]:
    """
    Returns the information about weights that are used for calculating ratio between primary and backup precisions.

    :param all_weight_params: List of all weight parameters.
    :param mode: Weight compression mode.
    :param is_last_layer_shared: Indicates whether the last layer shares the weight to be quantized.
    :param all_layers: Indicates whether embeddings and last layers should be compressed to a primary
        precision. By default, the backup precision is assigned for the embeddings and last layers.
    :return: List of information about weight nodes that are considered for mixed precision.
    """
    ratio_defining_params = all_weight_params
    if mode not in [CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM] and not all_layers:
        ratio_defining_params = list(filter(lambda wp: wp.metatype != OVEmbeddingMetatype, ratio_defining_params))
        if not is_last_layer_shared:
            ratio_defining_params = ratio_defining_params[:-1]
    return ratio_defining_params


def _set_weight_compression_config(
    ratio_defining_params: List[WeightNodeParams],
    mode: CompressWeightsMode,
    ratio: float,
    group_size: int,
    activations: Optional[Dict[str, np.ndarray]] = None,
    sensitivity_metric: Optional[SensitivityMetric] = SensitivityMetric.WEIGHT_QUANTIZATION_ERROR,
) -> None:
    """
    Set the appropriate compression configuration for weights based on some criteria.

    :param ratio_defining_params: Information about weights that are used for calculating ratio between primary and
        backup precisions.
    :param mode: Weight compression mode.
    :param ratio: The ratio between primary and backup precisions.
    :param group_size: number of weights (e.g. 128) in the channel dimension that share quantization parameters (scale).
    :return: None.
    """
    primary_config = WeightCompressionConfig(mode=mode, group_size=group_size)
    if ratio == 1:
        for weight_param in ratio_defining_params:
            weight_param.compression_config = primary_config
    else:
        criterion_cls = MIXED_PRECISION_CRITERIA.get(sensitivity_metric)
        criterion = criterion_cls(ratio_defining_params, primary_config, ratio, activations)
        criterion.assign_mixed_precision()


def apply_AWQ(
    model: ov.Model,
    graph: NNCFGraph,
    all_weight_params: List[WeightNodeParams],
    nodes_to_compress: List[NNCFNode],
    activations: Optional[Dict[str, np.ndarray]] = None,
    subset_size: int = 64,
    percent_to_apply=0.002,
    alpha_min=0.01,
    alpha_max=1.0,
    steps=100,
):
    matches = []

    inference_nncf_graph = transform_to_inference_graph(deepcopy(graph), [], [], [])
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
        :param group_size: Number of weights (e.g. 128) in the channel dimension,
                           that share quantization parameters (scale).
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
        skip = False
        for m in match[:-1]:
            node = graph.get_node_by_key(m)
            n_outupts = len(graph.get_output_edges(node))
            if n_outupts > 1:
                skip = True
        if skip:
            continue

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
        if weight_params.compression_config.num_bits != 4:
            continue
        target_node = nodes_to_compress[name_mapping[target_node_names[-1]]]
        merge_node = nodes_to_compress[name_mapping[merge_node_names[-1]]]

        awq_data[target_node.node_name] = AWQTriplet(weight_params, target_node, merge_node)

    alpha_step = (alpha_max - alpha_min) / steps

    model_transformer = ModelTransformerFactory.create(model, True)
    transformation_layout = TransformationLayout()

    for k, awq_data_item in track(awq_data.items(), description="Applying AWQ"):
        wp = awq_data_item.weight_params
        target_node = awq_data_item.target_node
        merge_node = awq_data_item.merge_node

        config = wp.compression_config

        stats = activations[k]
        X = np.vstack([stat.mean(axis=0).squeeze() for stat in stats]).transpose()
        if X.shape[1] > subset_size:
            X = X[:, :subset_size]

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

                g_compressed_weighs, g_c_scale, g_c_zp = do_integer_quantization(
                    gweight * cur_scale, reduction_axis, awq_config
                )
                g_decompressed_weighs = do_dequantization(g_compressed_weighs, g_c_scale, g_c_zp)
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

    # update nodes in parameters
    friendly_name_to_op_map = {op.get_friendly_name(): op for op in model.get_ops()}
    for wp in all_weight_params:
        name = wp.fq_name
        idx = name.find("fq_weights_")
        name = name[: idx - 1]
        wp.weight_node = friendly_name_to_op_map[name]

    return model
