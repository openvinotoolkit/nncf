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

from typing import Callable, Iterable, Optional

import torch
import torch.fx

import nncf
import nncf.errors
import nncf.tensor
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.patterns.patterns import GraphPattern
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.tensor_statistics.statistic_point import StatisticPoint
from nncf.experimental.common.tensor_statistics.collectors import MeanReducer
from nncf.experimental.common.tensor_statistics.collectors import NoopAggregator
from nncf.experimental.common.tensor_statistics.collectors import ShapeReducer
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.experimental.common.tensor_statistics.statistics import WCTensorStatistic
from nncf.experimental.torch.fx.commands import FXApplyTransformationCommand
from nncf.experimental.torch.fx.model_transformer import FXModelTransformer
from nncf.experimental.torch.fx.node_utils import get_graph_node_by_name
from nncf.experimental.torch.fx.node_utils import get_tensor_constant_from_node
from nncf.experimental.torch.fx.transformations import constant_update_transformation_builder
from nncf.experimental.torch.fx.transformations import module_insertion_transformation_builder
from nncf.parameters import CompressionFormat
from nncf.parameters import CompressWeightsMode
from nncf.quantization.advanced_parameters import AdvancedCompressionParameters
from nncf.quantization.algorithms.weight_compression.backend import AWQAlgoBackend
from nncf.quantization.algorithms.weight_compression.backend import MixedPrecisionAlgoBackend
from nncf.quantization.algorithms.weight_compression.backend import WeightCompressionAlgoBackend
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionParameters
from nncf.quantization.algorithms.weight_compression.lora_correction import LoraCorrectionAlgorithm
from nncf.quantization.algorithms.weight_compression.parameters import CompressedWeight
from nncf.quantization.algorithms.weight_compression.torch_backend import PTAWQAlgoAlgoBackend
from nncf.quantization.algorithms.weight_compression.torch_backend import PTMixedPrecisionAlgoBackend
from nncf.quantization.algorithms.weight_compression.torch_backend import PTWeightCompressionAlgoBackend
from nncf.quantization.algorithms.weight_compression.weight_lowering import compress_weight
from nncf.tensor import Tensor
from nncf.tensor.definitions import TensorDataType
from nncf.torch.graph import operator_metatypes as om
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.model_graph_manager import get_const_node
from nncf.torch.model_graph_manager import get_weight_tensor_port_ids
from nncf.torch.quantization.ignored_patterns import create_rope
from nncf.torch.quantization.layers import INT4AsymmetricWeightsDecompressor
from nncf.torch.quantization.layers import INT4SymmetricWeightsDecompressor
from nncf.torch.quantization.layers import INT8AsymmetricWeightsDecompressor
from nncf.torch.quantization.layers import INT8SymmetricWeightsDecompressor


class FXWeightCompressionAlgoBackend(WeightCompressionAlgoBackend):
    MATMUL_METATYPES = PTWeightCompressionAlgoBackend.MATMUL_METATYPES
    EMBEDDING_METATYPES = PTWeightCompressionAlgoBackend.EMBEDDING_METATYPES
    CONVOLUTION_METATYPES = PTWeightCompressionAlgoBackend.CONVOLUTION_METATYPES

    @property
    def matmul_metatypes(self) -> list[OperatorMetatype]:
        return FXWeightCompressionAlgoBackend.MATMUL_METATYPES

    @property
    def embedding_metatypes(self) -> list[OperatorMetatype]:
        return FXWeightCompressionAlgoBackend.EMBEDDING_METATYPES

    @property
    def convolution_metatypes(self) -> list[OperatorMetatype]:
        return FXWeightCompressionAlgoBackend.CONVOLUTION_METATYPES

    @staticmethod
    def is_node_with_weights(node: NNCFNode, graph: NNCFGraph) -> bool:
        return PTWeightCompressionAlgoBackend.is_node_with_weights(node, graph)

    @staticmethod
    def get_weight_names_and_port_ids(node: NNCFNode, graph: NNCFGraph) -> list[tuple[str, int]]:
        port_ids = get_weight_tensor_port_ids(node, graph)
        weight_name_port_ids = [(get_const_node(node, pid, graph).node_name, pid) for pid in port_ids]
        return weight_name_port_ids

    @staticmethod
    def get_reduction_axes(node_with_weight: NNCFNode, weight_port_id: int, graph: NNCFGraph) -> Optional[tuple[int]]:
        weight_node = get_const_node(node_with_weight, weight_port_id, graph)
        edge = graph.get_edge(weight_node, graph.get_next_nodes(weight_node)[0])

        ndims = len(edge.tensor_shape)
        reduction_axes = None
        if node_with_weight.metatype == om.PTAtenEmbeddingMetatype:
            reduction_axes = [1]
        elif node_with_weight.metatype == om.PTLinearMetatype:
            reduction_axes = [ndims - 1]
        elif node_with_weight.metatype == om.PTMatMulMetatype:
            if weight_port_id == 0:
                reduction_axes = [ndims - 1]
            elif weight_port_id == 1:
                reduction_axes = [max(0, ndims - 2)]
        elif node_with_weight.metatype == om.PTAddmmMetatype:
            if weight_port_id == 1:
                reduction_axes = [ndims - 1]
            elif weight_port_id == 2:
                reduction_axes = [max(0, ndims - 2)]
        elif node_with_weight.metatype in FXWeightCompressionAlgoBackend.CONVOLUTION_METATYPES:
            channel_idx = (
                1
                if node_with_weight.metatype
                in [om.PTConvTranspose1dMetatype, om.PTConvTranspose2dMetatype, om.PTConvTranspose3dMetatype]
                else 0
            )
            reduction_axes = [i for i in range(ndims) if i != channel_idx]
        return tuple(reduction_axes)

    @staticmethod
    def target_point(target_type: TargetType, target_node_name: str, port_id: int) -> PTTargetPoint:
        return PTWeightCompressionAlgoBackend.target_point(target_type, target_node_name, port_id)

    def mean_statistic_collector(
        self, reduction_axes: tuple[int], subset_size: Optional[int] = None
    ) -> TensorCollector:
        mean_reducer = MeanReducer(reduction_axes)
        shape_reducer = ShapeReducer()
        collector = TensorCollector(WCTensorStatistic)
        collector.register_statistic_branch(WCTensorStatistic.MEAN_STAT, mean_reducer, NoopAggregator(subset_size))
        collector.register_statistic_branch(WCTensorStatistic.SHAPE_STAT, shape_reducer, NoopAggregator(subset_size))
        return collector

    @staticmethod
    def get_activation_port_id(node: NNCFNode, graph: NNCFGraph) -> int:
        return PTWeightCompressionAlgoBackend.get_activation_port_id(node, graph)

    def get_weight(
        self, node_with_weight: NNCFNode, weight_port_id: int, model: torch.fx.GraphModule, graph: NNCFGraph
    ) -> Tensor:
        graph_node_with_weight = get_graph_node_by_name(model.graph, node_with_weight.node_name)
        graph_weight_node = graph_node_with_weight.all_input_nodes[weight_port_id]
        weight = get_tensor_constant_from_node(graph_weight_node, model).data
        if weight is None:
            msg = f"Could not find a node in the model by name {graph_weight_node}."
            raise nncf.InternalError(msg)

        return Tensor(weight)

    def get_weight_dtype(
        self, node_with_weight: NNCFNode, weight_port_id: int, model: torch.fx.GraphModule, graph: NNCFGraph
    ) -> TensorDataType:
        return self.get_weight(node_with_weight, weight_port_id, model, graph).dtype

    @staticmethod
    def get_weight_shape(node_with_weight: NNCFNode, weight_port_id: int, graph: NNCFGraph) -> tuple:
        weight_node = get_const_node(node_with_weight, weight_port_id, graph)
        edge = graph.get_edge(weight_node, node_with_weight)
        return tuple(edge.tensor_shape)

    def set_weight(
        self,
        node_with_weight: NNCFNode,
        weight_port_id: int,
        model: torch.fx.GraphModule,
        graph: NNCFGraph,
        weight: Tensor,
    ) -> None:
        constant_update_transformation_builder(node_with_weight, weight.data, input_port_id=weight_port_id)(model)

    def insert_adapters(
        self, wc_params: WeightCompressionParameters, lora_A: Tensor, lora_B: Tensor, int8_lora: bool
    ) -> None:
        pass

    @staticmethod
    def get_filter_fn_for_statistics(activation_port_id: int, algorithm_key: str) -> Callable[[StatisticPoint], bool]:
        def filter_func(point: StatisticPoint) -> bool:
            return (
                algorithm_key in point.algorithm_to_tensor_collectors
                and point.target_point.type
                == PTWeightCompressionAlgoBackend.TARGET_TYPE_TO_PT_INS_TYPE_MAP[TargetType.POST_LAYER_OPERATION]
            )

        return filter_func

    def transform_model(
        self,
        model: torch.fx.GraphModule,
        graph: NNCFGraph,
        weight_compression_parameters: Iterable[WeightCompressionParameters],
        precomputed_compressed_weights: Optional[dict[str, CompressedWeight]] = None,
        lora_correction_algo: Optional[LoraCorrectionAlgorithm] = None,
        compression_format: CompressionFormat = CompressionFormat.DQ,
        advanced_parameters: AdvancedCompressionParameters = AdvancedCompressionParameters(),
    ) -> torch.fx.GraphModule:
        transformation_layout = TransformationLayout()
        for wc_params in weight_compression_parameters:
            compression_config = wc_params.compression_config
            if compression_config.mode in [
                CompressWeightsMode.NF4,
                CompressWeightsMode.E2M1,
            ]:
                msg = f"{compression_config.mode.value} is not supported."
                raise nncf.ParameterNotSupportedError(msg)
            weight_node = get_const_node(wc_params.node_with_weight, wc_params.weight_port_id, graph)
            weight_name = weight_node.node_name
            weight = self.get_weight(wc_params.node_with_weight, wc_params.weight_port_id, model, graph)
            if weight is None or not isinstance(weight, Tensor):
                msg = f"Could not find a nncf.tensor in the model by name {weight_name}."
                raise nncf.InternalError(msg)

            # calculates compressed weights and decompression parameters
            compressed_weight = compress_weight(
                weight,
                wc_params.reduction_axes,
                compression_config,
                None
                if precomputed_compressed_weights is None
                else precomputed_compressed_weights.get(wc_params.weight_name),
            )

            # creates weight decompressor
            if compression_config.mode == CompressWeightsMode.INT8_SYM:
                decompressor = INT8SymmetricWeightsDecompressor(
                    compressed_weight.scale.data, result_dtype=weight.data.dtype
                )
            elif compression_config.mode == CompressWeightsMode.INT8_ASYM:
                decompressor = INT8AsymmetricWeightsDecompressor(
                    compressed_weight.scale.data, compressed_weight.zero_point.data, result_dtype=weight.data.dtype
                )
            elif compression_config.mode == CompressWeightsMode.INT4_SYM:
                decompressor = INT4SymmetricWeightsDecompressor(
                    scale=compressed_weight.scale.data,
                    compressed_weight_shape=compressed_weight.tensor.shape,
                    result_shape=weight.shape,
                    result_dtype=weight.data.dtype,
                )
            elif compression_config.mode == CompressWeightsMode.INT4_ASYM:
                decompressor = INT4AsymmetricWeightsDecompressor(
                    scale=compressed_weight.scale.data,
                    zero_point=compressed_weight.zero_point.data,
                    compressed_weight_shape=compressed_weight.tensor.shape,
                    result_shape=weight.shape,
                    result_dtype=weight.data.dtype,
                )

            # pack tensor
            packed_tensor = decompressor.pack_weight(compressed_weight.tensor.data)

            # sets compressed tensor
            self.set_weight(wc_params.node_with_weight, wc_params.weight_port_id, model, graph, packed_tensor)

            # register weight decompression module in the model
            graph_weight_node = get_graph_node_by_name(model.graph, wc_params.node_with_weight.node_name)
            compressed_weight_name = graph_weight_node.all_input_nodes[wc_params.weight_port_id].name

            decompressor_suffix = "_".join(compressed_weight_name.replace(".", "_").split("_")[:-2])
            decompressor_name = f"{decompressor.quantization_mode}_weights_decompressor_{decompressor_suffix}"

            # inserts the weight decompressor into the model as the post hook on the model weight
            transformation_layout.register(
                FXApplyTransformationCommand(
                    module_insertion_transformation_builder(
                        decompressor,
                        [PTTargetPoint(TargetType.OPERATOR_POST_HOOK, target_node_name=compressed_weight_name)],
                        decompressor_name,
                    )
                )
            )
        # apply transformations
        transformed_model = FXModelTransformer(model).transform(transformation_layout)

        return transformed_model

    @staticmethod
    def get_ignored_patterns() -> GraphPattern:
        return create_rope()


class FXMixedPrecisionAlgoBackend(MixedPrecisionAlgoBackend, FXWeightCompressionAlgoBackend):
    @staticmethod
    def mean_variance_statistic_collector(
        reduction_axes: tuple[int], subset_size: Optional[int] = None
    ) -> TensorCollector:
        return PTMixedPrecisionAlgoBackend.mean_variance_statistic_collector(
            reduction_axes=reduction_axes, subset_size=subset_size
        )

    @staticmethod
    def max_variance_statistic_collector(
        reduction_axes: tuple[int], subset_size: Optional[int] = None
    ) -> TensorCollector:
        return PTMixedPrecisionAlgoBackend.max_variance_statistic_collector(
            reduction_axes=reduction_axes, subset_size=subset_size
        )

    @staticmethod
    def mean_abs_max_statistic_collector(
        reduction_axes: tuple[int], subset_size: Optional[int] = None
    ) -> TensorCollector:
        return PTMixedPrecisionAlgoBackend.mean_abs_max_statistic_collector(
            reduction_axes=reduction_axes, subset_size=subset_size
        )


class FXAWQMultiply(torch.nn.Module):
    def __init__(self, scale: torch.Tensor):
        super().__init__()
        self.register_buffer("_scale_value", scale)
        self._scale_value: torch.Tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mul(x, self._scale_value)


class FXAWQAlgoAlgoBackend(AWQAlgoBackend, FXWeightCompressionAlgoBackend):
    @staticmethod
    def get_awq_patterns():
        return PTAWQAlgoAlgoBackend.get_awq_patterns()

    @staticmethod
    def scale_insertion_command(source_node, next_nodes, source_node_output_port, scale):
        input_port_id = 0
        target_points = []
        for node in next_nodes:
            target_points.append(
                PTTargetPoint(
                    TargetType.OPERATOR_PRE_HOOK,
                    node.node_name,
                    input_port_id=input_port_id,
                )
            )
        awq_multiply = FXAWQMultiply(scale)
        awq_node_name = f"{source_node.node_name}/awq_mul"
        return FXApplyTransformationCommand(
            module_insertion_transformation_builder(
                awq_multiply,
                target_points,
                awq_node_name,
            )
        )
