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

from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, TypeVar

import nncf
from nncf import nncf_logger
from nncf.common.factory import ModelTransformerFactory
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.graph_matching import find_subgraphs_matching_pattern
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.logging.track_progress import track
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.experimental.common.tensor_statistics.statistics import WCTensorStatistic
from nncf.quantization.algorithms.algorithm import Algorithm
from nncf.quantization.algorithms.weight_compression.activation_stats import process_stats
from nncf.quantization.algorithms.weight_compression.backend import WeightCompressionAlgoBackend
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionParameters
from nncf.quantization.algorithms.weight_compression.weight_lowering import float_quantize_dequantize_weight
from nncf.quantization.algorithms.weight_compression.weight_lowering import integer_quantize_dequantize_weight
from nncf.quantization.passes import transform_to_inference_graph
from nncf.tensor import TensorDataType
from nncf.tensor import functions as fns

TModel = TypeVar("TModel")
TTensor = TypeVar("TTensor")
TWeightType = TypeVar("TWeightType")


FP16_MAX_VALUE = 65504.0
FP16_OVERFLOW_MARGIN = 0.25


@dataclass
class AWQCompressionInfo:
    """
    Information on AWQ nodes.
    """

    weight_params: WeightCompressionParameters = None
    target_node: NNCFNode = None
    merge_node: NNCFNode = None


class AWQ(Algorithm):
    """
    Modified AWQ algorithm implementation.
    """

    def __init__(
        self,
        subset_size: int = 32,
        percent_to_apply: float = 0.002,
        alpha_min: float = 0.0,
        alpha_max: float = 1.0,
        steps: int = 100,
        prefer_data_aware_scaling: bool = True,
    ):
        """
        :param subset_size: The number of samples for AWQ.
        :param percent_to_apply: The percent of outliers for correction.
        :param alpha_min: Minimum value of smoothness parameter for grid search.
        :param alpha_max: Maximal value of smoothness parameter for grid search.
        :param steps: The number of the steps in grid search.
        :param prefer_data_aware_scaling: Determines whether to use activations to calculate scales.
        """
        super().__init__()
        self._subset_size = subset_size
        self._percent_to_apply = percent_to_apply
        self._alpha_min = alpha_min
        self._alpha_max = alpha_max
        self._steps = steps
        self._prefer_data_aware_scaling = prefer_data_aware_scaling
        self._backend_entity = None
        self._patterns = None
        self._scale_per_target_node = {}

    @property
    def available_backends(self) -> list[BackendType]:
        return [BackendType.OPENVINO, BackendType.TORCH, BackendType.ONNX]

    def _set_backend_entity(
        self, model: TModel, wc_backend_entity: Optional[WeightCompressionAlgoBackend] = None
    ) -> None:
        """
        Creates a helper class with a backed-specific logic of the algorithm.

        :param model: Backend-specific input model.
        :param wc_backend_entity: Weight compression algorithm backend.
        """
        model_backend = get_backend(model)
        if model_backend == BackendType.OPENVINO:
            from nncf.quantization.algorithms.weight_compression.openvino_backend import OVAWQAlgoAlgoBackend

            self._backend_entity = OVAWQAlgoAlgoBackend(model, wc_backend_entity.name_to_node_mapping)
        elif model_backend == BackendType.TORCH:
            from nncf.quantization.algorithms.weight_compression.torch_backend import PTAWQAlgoAlgoBackend

            self._backend_entity = PTAWQAlgoAlgoBackend()
        elif model_backend == BackendType.TORCH_FX:
            from nncf.quantization.algorithms.weight_compression.torch_fx_backend import FXAWQAlgoAlgoBackend

            self._backend_entity = FXAWQAlgoAlgoBackend()
        elif model_backend == BackendType.ONNX:
            from nncf.quantization.algorithms.weight_compression.onnx_backend import ONNXAWQAlgoAlgoBackend

            self._backend_entity = ONNXAWQAlgoAlgoBackend(model)
        else:
            msg = f"Cannot return backend-specific AWQ entity because {model_backend.value} is not supported!"
            raise nncf.UnsupportedBackendError(msg)
        self._patterns = self._backend_entity.get_awq_patterns()

    def apply(
        self,
        model: TModel,
        graph: NNCFGraph,
        all_weight_params: list[WeightCompressionParameters],
        statistics: Optional[dict[str, WCTensorStatistic]] = None,
        wc_backend_entity: Optional[WeightCompressionAlgoBackend] = None,
    ) -> TModel:
        """
        Applies the algorithm to the model.
        :param model: Model for applying algorithm.
        :param graph: Model graph.
        :param all_weight_params: List of all weight parameters.
        :param statistics: Input activation statistics for each node.
        :param wc_backend_entity: Weight compression algorithm backend.
        :return: A resulting model.
        """
        self._set_backend_entity(model, wc_backend_entity)

        awq_data = self._get_awq_data(graph, all_weight_params)
        if len(awq_data) == 0:
            return model

        transformation_layout = TransformationLayout()
        model_transformer = ModelTransformerFactory.create(model, inplace=True)

        is_data_free = statistics is None or not self._prefer_data_aware_scaling

        description = "Applying data-free AWQ" if is_data_free else "Applying data-aware AWQ"

        for k, awq_data_item in track(awq_data.items(), description=description):
            wp = awq_data_item.weight_params
            merge_node = awq_data_item.merge_node
            weight_data = self._backend_entity.get_weight_names_and_port_ids(wp.node_with_weight, graph)
            if len(weight_data) != 1:  # not supported by the algorithm
                continue
            is_mergeable = self._backend_entity.is_node_with_weights(merge_node, graph)

            nncf_logger.debug(f"{description} for: {wp.node_with_weight.node_name}")

            _, weight_port_id = weight_data[0]
            weight = self._backend_entity.get_weight(
                wp.node_with_weight, weight_port_id, model, graph
            )  # get_const_value(wp.weight_node)
            weight_dtype = weight.dtype
            weight = weight.astype(TensorDataType.float32)

            if is_data_free:
                scale = self._data_free_step(weight, 1 - wp.reduction_axes[0])
            else:
                prev_weight, prev_statistics = None, None
                if is_mergeable:
                    # for MatMul->Multiply->MatMul pattern we need to use statistics from the first MatMul
                    prev_weight_data = self._backend_entity.get_weight_names_and_port_ids(merge_node, graph)
                    _, prev_weight_port_id = prev_weight_data[0]
                    prev_weight = self._backend_entity.get_weight(merge_node, prev_weight_port_id, model, graph)

                    prev_statistics = statistics[merge_node.node_name]
                scale = self._data_aware_step(wp, weight, statistics[k], prev_weight, prev_statistics)

            w_scale = fns.unsqueeze(scale, -1 - wp.reduction_axes[0])
            a_scale = fns.unsqueeze(1.0 / scale, -wp.reduction_axes[0])

            scaled_weight = (weight * w_scale).astype(weight_dtype)
            self._backend_entity.set_weight(wp.node_with_weight, weight_port_id, model, graph, scaled_weight)

            if is_mergeable:  # for MatMul->Multiply->MatMul pattern the scale is merged to the first MatMul
                for _, port_id in self._backend_entity.get_weight_names_and_port_ids(merge_node, graph):
                    merge_weight = self._backend_entity.get_weight(merge_node, port_id, model, graph)
                    merge_weight = (merge_weight * a_scale).astype(weight_dtype)
                    self._backend_entity.set_weight(merge_node, port_id, model, graph, merge_weight)
                a_scale = fns.moveaxis(a_scale, -1, -2)
            else:  # for Act->Multiply->MatMul and Act->MatMul patterns scale inserted after Act as extra node
                a_scale = fns.moveaxis(a_scale, -1, -2).astype(weight_dtype)
                next_nodes = graph.get_next_nodes(merge_node)
                source_node_output_port = graph.get_output_edges(merge_node)[0].output_port_id
                scale_insertion_command = self._backend_entity.scale_insertion_command(
                    merge_node, next_nodes, source_node_output_port, a_scale.data
                )
                transformation_layout.register(scale_insertion_command)

            self._scale_per_target_node[k] = a_scale

        transformed_model = model_transformer.transform(transformation_layout)

        return transformed_model

    def _data_aware_step(self, wp, weight, statistics, prev_weight=None, prev_statistics=None):
        alpha_step = (self._alpha_max - self._alpha_min) / self._steps
        config = wp.compression_config
        s, X = process_stats(statistics, self._subset_size)
        s = s.astype(TensorDataType.float32)
        X = X.astype(TensorDataType.float32)

        is_2d_weight = weight.ndim == 2

        assert isinstance(wp.reduction_axes, tuple) and len(wp.reduction_axes) == 1
        reduction_axis = wp.reduction_axes[0]

        prev_s, prev_w = None, None
        if prev_statistics is not None and prev_weight is not None:
            prev_s, _ = process_stats(prev_statistics, self._subset_size)
            prev_s = prev_s.astype(TensorDataType.float32).max().item()
            prev_weight = fns.unsqueeze(prev_weight, 0)  # [out_features, hidden_dim] -> [1, out_features, hidden_dim]
            prev_w = fns.mean(fns.abs(prev_weight), axis=reduction_axis + 1)

        if is_2d_weight:
            s = fns.unsqueeze(s, 0)  # [hidden_dim] -> [1, hidden_dim]
            X = fns.unsqueeze(X, 0)  # [hidden_dim, samples] -> [1, hidden_dim, samples]
            weight = fns.unsqueeze(weight, 0)  # [out_features, hidden_dim] -> [1, out_features, hidden_dim]
            reduction_axis += 1

        top_k = max(int(s.shape[-1] * self._percent_to_apply), 1)
        topk_idxs = fns.argsort(-s)[:, :top_k]

        group_size = config.group_size
        if group_size == -1:
            group_size = s.shape[-1]

        groups_to_correct = set()
        for expert_idx in range(topk_idxs.shape[0]):
            for k_idx in range(topk_idxs.shape[1]):
                idx = topk_idxs[expert_idx, k_idx].item()
                group_idx = idx // group_size
                groups_to_correct.add((expert_idx, group_idx))

        groups_to_correct = list(groups_to_correct)

        if reduction_axis == 1:
            # Weights
            # 3D: [num_experts, hidden_dimension, out_features] -> [num_experts, out_features, hidden_dimension]
            # 2D: [1, hidden_dimension, out_features] -> [1, out_features, hidden_dimension]
            weight = fns.moveaxis(weight, -1, -2)
            reduction_axis = weight.ndim - 1

        shape_vector = fns.mean(X, axis=-1)
        scale = fns.ones_like(shape_vector)

        awq_config = deepcopy(config)
        awq_config.group_size = -1

        for expert_idx, gi in groups_to_correct:
            offset = gi * group_size
            gscale = s[expert_idx, offset : offset + group_size]
            gweight = weight[expert_idx, :, offset : offset + group_size]
            gacts = X[expert_idx, offset : offset + group_size, :]

            a_min = fns.astype(fns.quantile(gscale, 0.1), TensorDataType.float32)
            a_max = 1e2
            gscale = fns.clip(gscale, a_min=a_min, a_max=a_max)

            fp32_out = fns.matmul(gweight, gacts)
            min_diff = fns.max(fns.abs(fp32_out))
            best_scale = None

            alpha = self._alpha_min
            for _ in range(self._steps):
                cur_scale = gscale**alpha

                if prev_s is not None:
                    threshold = (
                        FP16_OVERFLOW_MARGIN * FP16_MAX_VALUE
                    )  # take the threshold from the fp16 type with some margin
                    # per channel magnitudes for the previous MatMul
                    # mean(abs(prev_weight)) * max(abs((prev_activation))) * prev_weight.shape[reduction_axis]
                    magnitudes = (
                        (prev_w[expert_idx, offset : offset + group_size] / cur_scale)
                        * prev_s
                        * prev_weight.shape[reduction_axis]
                    )
                    if magnitudes.max() >= threshold:
                        cur_scale = AWQ._clamp_scale(
                            magnitudes,
                            threshold,
                            cur_scale,
                            prev_w[expert_idx, offset : offset + group_size]
                            * prev_s
                            * prev_weight.shape[reduction_axis]
                            / threshold,
                        )

                weights_to_fake_quantize = gweight * cur_scale
                if not config.is_integer:
                    g_decompressed_weighs = float_quantize_dequantize_weight(weights_to_fake_quantize, awq_config, -1)
                else:
                    g_decompressed_weighs = integer_quantize_dequantize_weight(weights_to_fake_quantize, awq_config, -1)
                sacts = gacts / fns.unsqueeze(cur_scale, 1)

                cur_out = fns.matmul(g_decompressed_weighs, sacts)
                cur_diff = fns.mean(fns.abs(cur_out - fp32_out))
                if cur_diff < min_diff:
                    min_diff = cur_diff
                    best_scale = cur_scale
                alpha += alpha_step

            if best_scale is not None:
                scale.data[expert_idx, offset : offset + group_size] = best_scale.data

        if is_2d_weight:
            scale = fns.squeeze(scale, 0)  # [1, hidden_dim] -> [hidden_dim]

        return scale

    @staticmethod
    def _clamp_scale(magnitudes, threshold, scale, clamped_scale):
        return fns.where(magnitudes < threshold, scale, clamped_scale)

    def _data_free_step(self, weight, axis):
        eps = fns.finfo(weight).eps
        scale = fns.maximum(fns.mean(fns.abs(weight), axis=axis), eps)
        return 1 / scale

    def _get_awq_data(
        self, graph: NNCFGraph, all_weight_params: list[WeightCompressionParameters]
    ) -> dict[str, AWQCompressionInfo]:
        """
        Finds awq patterns in graph and returns it.
        :param graph: Model graph.
        :param all_weight_params: list of all weight parameters.
        :return: A dict with node names and matched AWQ patterns.
        """
        matches = []
        inference_nncf_graph = transform_to_inference_graph(deepcopy(graph), [], [], [], [])
        nx_graph = inference_nncf_graph.get_nx_graph_copy()
        for pattern_graph in self._patterns.values():
            matches.extend(find_subgraphs_matching_pattern(nx_graph, pattern_graph(), strict=False))

        if len(matches) == 0:
            nncf_logger.info("No matching patterns were found for applying AWQ algorithm, it will be skipped.")
            return {}

        awq_data = {}
        name_mapping = {wp.weight_name: idx for idx, wp in enumerate(all_weight_params)}

        for match in matches:
            nncf_node = graph.get_node_by_key(match[-1])
            if not self._backend_entity.is_node_with_weights(nncf_node, graph):
                continue

            target_node_names = []
            for weight_op_friendly_name, _ in self._backend_entity.get_weight_names_and_port_ids(nncf_node, graph):
                target_node_names.append(weight_op_friendly_name)

            # skip node if it is in IgnoredScope or should not be compressed
            if target_node_names[-1] not in name_mapping:
                continue

            weight_params = all_weight_params[name_mapping[target_node_names[-1]]]

            if weight_params.compression_config.num_bits != 4:
                continue
            target_node = weight_params.node_with_weight

            # avoid matching different patterns for the same node
            if target_node.node_name in awq_data:
                continue

            merge_node = graph.get_node_by_key(match[0])

            awq_data[target_node.node_name] = AWQCompressionInfo(weight_params, target_node, merge_node)
        return awq_data

    def update_statistics(self, statistics):
        if not statistics:
            return statistics

        # Multiply activations by the computed scales
        for node_name, scale in self._scale_per_target_node.items():
            for mean_stat in statistics[node_name].mean_values:
                mean_stat *= fns.squeeze(scale)
        return statistics

    def get_statistic_points(self, model: TModel, graph: NNCFGraph) -> StatisticPointsContainer:
        """
        Returns statistic points, for which StatisticsCollector should collect statistics.

        :param model: Model for statistics collection.
        :param graph: Model graph.
        :return: Statistic points, for which StatisticsCollector should collect statistics.
        """
        return StatisticPointsContainer()
