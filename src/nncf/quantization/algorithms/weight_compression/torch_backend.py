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

from typing import Callable, Iterable, Optional, Union

import torch

import nncf
import nncf.torch.graph.operator_metatypes as om
from nncf.common.graph.definitions import NNCFGraphNodeType
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.operator_metatypes import CONST_NOOP_METATYPES
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.patterns import GraphPattern
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.quantization.structs import QuantizationScheme
from nncf.common.tensor_statistics.statistic_point import StatisticPoint
from nncf.experimental.common.tensor_statistics.collectors import MaxVarianceReducer
from nncf.experimental.common.tensor_statistics.collectors import MeanAbsMaxReducer
from nncf.experimental.common.tensor_statistics.collectors import MeanAggregator
from nncf.experimental.common.tensor_statistics.collectors import MeanReducer
from nncf.experimental.common.tensor_statistics.collectors import MeanVarianceReducer
from nncf.experimental.common.tensor_statistics.collectors import NoopAggregator
from nncf.experimental.common.tensor_statistics.collectors import ShapeReducer
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.experimental.common.tensor_statistics.statistics import MaxVarianceTensorStatistic
from nncf.experimental.common.tensor_statistics.statistics import MeanMagnitudeTensorStatistic
from nncf.experimental.common.tensor_statistics.statistics import MeanVarianceTensorStatistic
from nncf.experimental.common.tensor_statistics.statistics import WCTensorStatistic
from nncf.parameters import CompressionFormat
from nncf.parameters import CompressWeightsMode
from nncf.quantization.advanced_parameters import AdvancedCompressionParameters
from nncf.quantization.algorithms.weight_compression.awq_patterns import get_awq_patterns
from nncf.quantization.algorithms.weight_compression.backend import AWQAlgoBackend
from nncf.quantization.algorithms.weight_compression.backend import MixedPrecisionAlgoBackend
from nncf.quantization.algorithms.weight_compression.backend import WeightCompressionAlgoBackend
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionParameters
from nncf.quantization.algorithms.weight_compression.lora_correction import LoraCorrectionAlgorithm
from nncf.quantization.algorithms.weight_compression.parameters import CompressedWeight
from nncf.quantization.algorithms.weight_compression.weight_lowering import compress_weight
from nncf.tensor import Tensor
from nncf.tensor.definitions import TensorDataType
from nncf.torch.function_hook.commands import PT2InsertionCommand
from nncf.torch.function_hook.model_transformer import PT2ModelTransformer
from nncf.torch.function_hook.nncf_graph.nncf_graph_builder import GraphModelWrapper
from nncf.torch.graph.graph import PTTargetPoint
from nncf.torch.graph.operator_metatypes import CONVOLUTION_METATYPES
from nncf.torch.graph.operator_metatypes import EMBEDDING_METATYPES
from nncf.torch.graph.operator_metatypes import MATMUL_METATYPES
from nncf.torch.graph.operator_metatypes import PTMulMetatype
from nncf.torch.graph.pattern_operations import ATOMIC_ACTIVATIONS_OPERATIONS
from nncf.torch.graph.transformations.commands import PTSharedFnInsertionCommand
from nncf.torch.graph.transformations.commands import PTTransformationCommand
from nncf.torch.model_graph_manager import find_const_node_in_constant_subgraph
from nncf.torch.model_graph_manager import get_const_data
from nncf.torch.model_graph_manager import get_const_node
from nncf.torch.model_graph_manager import get_module_by_name
from nncf.torch.model_graph_manager import get_reduction_axes_from_metatype
from nncf.torch.model_graph_manager import split_const_name
from nncf.torch.model_transformer import PTModelTransformer
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.quantization.ignored_patterns import create_rope
from nncf.torch.quantization.layers import QUANTIZATION_MODULES
from nncf.torch.quantization.layers import INT4AsymmetricWeightsDecompressor
from nncf.torch.quantization.layers import INT4SymmetricWeightsDecompressor
from nncf.torch.quantization.layers import INT8AsymmetricWeightsDecompressor
from nncf.torch.quantization.layers import INT8SymmetricWeightsDecompressor
from nncf.torch.quantization.layers import PTLoraNLSSpec
from nncf.torch.quantization.layers import PTLoraSpec
from nncf.torch.quantization.layers import PTQuantizerSpec
from nncf.torch.quantization.layers import SQMultiply


class PTWeightCompressionAlgoBackend(WeightCompressionAlgoBackend):
    TARGET_TYPE_TO_PT_INS_TYPE_MAP = {
        TargetType.PRE_LAYER_OPERATION: TargetType.OPERATOR_PRE_HOOK,
        TargetType.POST_LAYER_OPERATION: TargetType.OPERATOR_POST_HOOK,
    }

    @property
    def matmul_metatypes(self) -> list[OperatorMetatype]:
        return MATMUL_METATYPES

    @property
    def embedding_metatypes(self) -> list[OperatorMetatype]:
        return EMBEDDING_METATYPES

    @property
    def convolution_metatypes(self) -> list[OperatorMetatype]:
        return CONVOLUTION_METATYPES

    @staticmethod
    def is_node_with_weights(node: NNCFNode, graph: NNCFGraph) -> bool:
        if (
            node.metatype not in MATMUL_METATYPES
            and node.metatype not in EMBEDDING_METATYPES
            and node.metatype not in CONVOLUTION_METATYPES
        ):
            return False
        for prev_node in graph.get_previous_nodes(node):
            edge = graph.get_edge(prev_node, node)
            if edge.input_port_id not in node.metatype.weight_port_ids:
                continue
            weight_node = find_const_node_in_constant_subgraph(prev_node, graph)
            if weight_node is not None:
                return True
        return False

    @staticmethod
    def get_weight_names_and_port_ids(node: NNCFNode, graph: NNCFGraph) -> list[tuple[str, int]]:
        weight_port_ids = []
        for prev_node in graph.get_previous_nodes(node):
            weight_node = find_const_node_in_constant_subgraph(prev_node, graph)
            if weight_node is None:
                continue
            edge = graph.get_edge(prev_node, node)
            if edge.input_port_id in node.metatype.weight_port_ids:
                weight_port_ids.append((weight_node.layer_attributes.name, edge.input_port_id))
        return weight_port_ids

    @staticmethod
    def get_reduction_axes(node_with_weight: NNCFNode, weight_port_id: int, graph: NNCFGraph) -> Optional[tuple[int]]:
        weight_node = get_const_node(node_with_weight, weight_port_id, graph)
        node_with_weight_metatype = node_with_weight.metatype

        ndims = len(weight_node.layer_attributes.shape)
        reduction_axes = get_reduction_axes_from_metatype(node_with_weight_metatype, weight_port_id, ndims)
        if node_with_weight_metatype == om.PTEmbeddingMetatype:
            reduction_axes = [1]
        return tuple(reduction_axes)

    @staticmethod
    def target_point(target_type: TargetType, target_node_name: str, port_id: int) -> PTTargetPoint:
        if NNCFGraphNodeType.INPUT_NODE in target_node_name or target_type == TargetType.POST_LAYER_OPERATION:
            port_id = None
        if target_type in PTWeightCompressionAlgoBackend.TARGET_TYPE_TO_PT_INS_TYPE_MAP:
            target_type = PTWeightCompressionAlgoBackend.TARGET_TYPE_TO_PT_INS_TYPE_MAP[target_type]
        return PTTargetPoint(target_type, target_node_name, input_port_id=port_id)

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
        activation_ports = []
        for prev_node in graph.get_previous_nodes(node):
            if prev_node.metatype in CONST_NOOP_METATYPES:
                continue
            edge = graph.get_edge(prev_node, node)
            activation_ports.append(edge.input_port_id)
        assert len(activation_ports) == 1
        return activation_ports[0]

    def get_weight(
        self,
        node_with_weight: NNCFNode,
        weight_port_id: int,
        model: Union[GraphModelWrapper, torch.nn.Module],
        graph: NNCFGraph,
    ) -> Tensor:
        if isinstance(model, GraphModelWrapper):
            model = model.model
        weight_node = get_const_node(node_with_weight, weight_port_id, graph)
        weight_name = weight_node.layer_attributes.name
        weight = get_const_data(weight_node, model)
        if weight is None:
            msg = f"Could not find a torch.nn.Parameter in the model by name {weight_name}."
            raise nncf.InternalError(msg)
        return Tensor(weight)

    def get_weight_dtype(
        self,
        node_with_weight: NNCFNode,
        weight_port_id: int,
        model: Union[GraphModelWrapper, torch.nn.Module],
        graph: NNCFGraph,
    ) -> TensorDataType:
        return self.get_weight(node_with_weight, weight_port_id, model, graph).dtype

    @staticmethod
    def get_weight_shape(node_with_weight: NNCFNode, weight_port_id: int, graph: NNCFGraph) -> tuple:
        weight_node = get_const_node(node_with_weight, weight_port_id, graph)
        return tuple(weight_node.layer_attributes.shape)

    def set_weight(
        self, node_with_weight: NNCFNode, weight_port_id: int, model: torch.nn.Module, graph: NNCFGraph, weight: Tensor
    ):
        weight_node = get_const_node(node_with_weight, weight_port_id, graph)
        module_name, weight_attr_name = split_const_name(weight_node.layer_attributes.name)
        module = get_module_by_name(module_name, model.model)
        weight_param = getattr(module, weight_attr_name)
        weight_param.data = weight.data

    def insert_adapters(
        self, wc_params: WeightCompressionParameters, lora_A: Tensor, lora_B: Tensor, int8_lora: bool
    ) -> None:
        raise NotImplementedError()

    @staticmethod
    def init_lora_adapters(svd_residual: torch.Tensor, rank: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Initializes LoRA adapters using Singular Value Decomposition (SVD).

        :param svd_residual: The residual tensor to be decomposed.
        :param rank: The rank for the decomposition. If None, the full rank is used.
        :return: A tuple containing the U and V matrices from the SVD.
        """
        # O stands for output dimension, H - input dimension or hidden size, R - rank.
        U_full, S_full, V_full = torch.linalg.svd(svd_residual, full_matrices=False)
        U = U_full[:, :rank]  # [H, R]
        S_sqrt = torch.sqrt(S_full)
        S = torch.diag(S_sqrt[:rank])  # [R, R]
        V = V_full[:rank, :]  # [R, O]
        V = S @ V  # [R, O]
        U = U @ S  # [H, R]
        return U, V

    @staticmethod
    def get_filter_fn_for_statistics(activation_port_id: int, algorithm_key: str) -> Callable[[StatisticPoint], bool]:
        def filter_func(point: StatisticPoint) -> bool:
            return (
                algorithm_key in point.algorithm_to_tensor_collectors
                and point.target_point.type
                == PTWeightCompressionAlgoBackend.TARGET_TYPE_TO_PT_INS_TYPE_MAP[TargetType.POST_LAYER_OPERATION]
            )

        return filter_func

    @staticmethod
    def get_fq_insertion_command(
        compressed_weight: CompressedWeight,
        wc_params: WeightCompressionParameters,
        orig_weight_shape: tuple[int, ...],
        compression_format: CompressionFormat,
        lora_adapter_rank: int,
        is_all_8bit: bool,
    ) -> PTTransformationCommand:
        """
        Creates a fake quantization insertion command for the given compressed weight.

        :param compressed_weight: The compressed weight tensor.
        :param wc_params: Parameters for weight compression.
        :param orig_weight_shape: The original shape of the weight tensor.
        :param compression_format: The format of compression.
        :param is_all_8bit: Flag indicating if all weights should be compressed to 8-bit.
        :return: A PTTransformationCommand for inserting fake quantization to the model.
        """
        compression_config = wc_params.compression_config
        # default mapping for 4bit weight compression and FQ_LORA format, no need to add lora adapters for 8bit weight
        mode_vs_schema_map = {
            CompressWeightsMode.INT4_ASYM: QuantizationScheme.ASYMMETRIC_LORA,
            CompressWeightsMode.INT4_SYM: QuantizationScheme.SYMMETRIC_LORA,
            CompressWeightsMode.INT8_ASYM: QuantizationScheme.ASYMMETRIC,
            CompressWeightsMode.INT8_SYM: QuantizationScheme.SYMMETRIC,
        }
        if compression_format == CompressionFormat.FQ:
            mode_vs_schema_map[CompressWeightsMode.INT4_ASYM] = QuantizationScheme.ASYMMETRIC
            mode_vs_schema_map[CompressWeightsMode.INT4_SYM] = QuantizationScheme.SYMMETRIC
        if is_all_8bit and compression_format == CompressionFormat.FQ_LORA:
            mode_vs_schema_map[CompressWeightsMode.INT8_ASYM] = QuantizationScheme.ASYMMETRIC_LORA
            mode_vs_schema_map[CompressWeightsMode.INT8_SYM] = QuantizationScheme.SYMMETRIC_LORA
        if compression_format == CompressionFormat.FQ_LORA_NLS:
            mode_vs_schema_map[CompressWeightsMode.INT4_ASYM] = QuantizationScheme.ASYMMETRIC_LORA_NLS
            mode_vs_schema_map[CompressWeightsMode.INT4_SYM] = QuantizationScheme.SYMMETRIC_LORA_NLS
            if is_all_8bit:
                mode_vs_schema_map[CompressWeightsMode.INT8_ASYM] = QuantizationScheme.ASYMMETRIC_LORA_NLS
                mode_vs_schema_map[CompressWeightsMode.INT8_SYM] = QuantizationScheme.SYMMETRIC_LORA_NLS

        schema = mode_vs_schema_map[compression_config.mode]

        device = compressed_weight.tensor.data.device
        scale = compressed_weight.scale.data

        weight_shape = compressed_weight.tensor.shape

        quantizer_spec = PTQuantizerSpec(
            num_bits=compression_config.num_bits,
            mode=schema,
            signedness_to_force=True,
            narrow_range=False,
            half_range=False,
            scale_shape=scale.shape,
            logarithm_scale=False,
        )

        quantizer_cls = QUANTIZATION_MODULES.get(schema)
        if schema in [
            QuantizationScheme.ASYMMETRIC_LORA,
            QuantizationScheme.ASYMMETRIC_LORA_NLS,
            QuantizationScheme.SYMMETRIC_LORA,
            QuantizationScheme.SYMMETRIC_LORA_NLS,
        ]:
            if schema in [QuantizationScheme.ASYMMETRIC_LORA, QuantizationScheme.SYMMETRIC_LORA]:
                lora_spec = PTLoraSpec(
                    lora_rank=lora_adapter_rank, orig_weight_shape=orig_weight_shape, weight_shape=weight_shape
                )
            else:
                lora_spec = PTLoraNLSSpec(
                    lora_rank=lora_adapter_rank,
                    active_lora_rank=lora_adapter_rank,
                    orig_weight_shape=orig_weight_shape,
                    weight_shape=weight_shape,
                )
            quantizer = quantizer_cls(quantizer_spec, lora_spec)
            lora_dtype = quantizer.lora_A.dtype
            svd_residual = torch.rand(weight_shape).to(device) * scale / 100  # value on [0,1] * (1/100 of quant size)
            svd_residual = svd_residual.reshape(orig_weight_shape)
            B, A = PTWeightCompressionAlgoBackend.init_lora_adapters(svd_residual, rank=lora_adapter_rank)
            quantizer.lora_A = torch.nn.Parameter(A.type(dtype=lora_dtype))
            quantizer.lora_B = torch.nn.Parameter(B.type(dtype=lora_dtype))
        else:
            quantizer = quantizer_cls(quantizer_spec)

        levels = quantizer.levels
        if schema in [
            QuantizationScheme.ASYMMETRIC_LORA,
            QuantizationScheme.ASYMMETRIC_LORA_NLS,
            QuantizationScheme.ASYMMETRIC,
        ]:
            zero_point = compressed_weight.zero_point.data
            dtype = quantizer.input_low.dtype
            # NOTE: Lose some accuracy, because of inversion of round
            input_low = -zero_point * scale
            input_range = scale * (levels - 1)
            quantizer.input_low = torch.nn.Parameter(input_low.type(dtype))
            quantizer.input_range = torch.nn.Parameter(input_range.type(dtype) - quantizer.eps)
        else:
            scale = scale.type(quantizer.scale.dtype)
            quantizer.scale = torch.nn.Parameter(scale * levels / 2)

        target_node_name = wc_params.weight_name
        target_point = PTTargetPoint(TargetType.OPERATOR_POST_HOOK, target_node_name=target_node_name)

        return PT2InsertionCommand([target_point], quantizer)

    @staticmethod
    def get_dq_insertion_command(
        compressed_weight: CompressedWeight,
        wc_params: WeightCompressionParameters,
        model: NNCFNetwork,
        graph: NNCFGraph,
        weight_node: NNCFNode,
    ) -> PTTransformationCommand:
        """
        Creates an insertion command that performs dequantization of the given compressed weight.

        :param compressed_weight: The compressed weight tensor.
        :param wc_params: Parameters for weight compression.
        :param model: The PyTorch model.
        :param graph: The NNCF graph.
        :param weight_node: The node representing the weight in the graph.
        :return: A PTTransformationCommand for inserting decompression to the model.
        """
        weight_name = weight_node.layer_attributes.name
        module_name, weight_attr_name = split_const_name(weight_name)
        module = get_module_by_name(module_name, model)
        weight = getattr(module, weight_attr_name)
        if not isinstance(weight, torch.nn.Parameter):
            msg = f"Weight is not a torch.nn.Parameter in the model by name {weight_name}."
            raise nncf.InternalError(msg)
        weight_dtype = weight.dtype
        weight_shape = weight.shape

        compression_config = wc_params.compression_config
        # creates weight decompressor
        if compression_config.mode == CompressWeightsMode.INT8_SYM:
            decompressor = INT8SymmetricWeightsDecompressor(compressed_weight.scale.data, result_dtype=weight_dtype)
        elif compression_config.mode == CompressWeightsMode.INT8_ASYM:
            decompressor = INT8AsymmetricWeightsDecompressor(
                compressed_weight.scale.data, compressed_weight.zero_point.data, result_dtype=weight_dtype
            )
        elif compression_config.mode == CompressWeightsMode.INT4_SYM:
            decompressor = INT4SymmetricWeightsDecompressor(
                scale=compressed_weight.scale.data,
                compressed_weight_shape=compressed_weight.tensor.shape,
                result_shape=weight_shape,
                result_dtype=weight_dtype,
            )
        elif compression_config.mode == CompressWeightsMode.INT4_ASYM:
            decompressor = INT4AsymmetricWeightsDecompressor(
                scale=compressed_weight.scale.data,
                zero_point=compressed_weight.zero_point.data,
                compressed_weight_shape=compressed_weight.tensor.shape,
                result_shape=weight_shape,
                result_dtype=weight_dtype,
            )

        # pack tensor
        packed_tensor = decompressor.pack_weight(compressed_weight.tensor.data)

        # sets compressed tensor
        # TODO:(AlexanderDokuchaev): update set_const_data
        module_name, weight_attr_name = split_const_name(weight_name)
        module = get_module_by_name(module_name, model)
        weight = getattr(module, weight_attr_name)

        if not isinstance(weight, torch.nn.Parameter):
            msg = f"Weight is not a torch.nn.Parameter in the model by name {weight_name}."
            raise nncf.InternalError(msg)

        weight.requires_grad = False
        weight.data = packed_tensor

        return PT2InsertionCommand(
            [PTTargetPoint(TargetType.OPERATOR_POST_HOOK, target_node_name=weight_node.node_name)],
            decompressor,
        )

    def transform_model(
        self,
        model: Union[GraphModelWrapper, torch.nn.Module],
        graph: NNCFGraph,
        weight_compression_parameters: Iterable[WeightCompressionParameters],
        precomputed_compressed_weights: Optional[dict[str, CompressedWeight]] = None,
        lora_correction_algo: Optional[LoraCorrectionAlgorithm] = None,
        compression_format: CompressionFormat = CompressionFormat.DQ,
        advanced_parameters: AdvancedCompressionParameters = AdvancedCompressionParameters(),
    ) -> NNCFNetwork:
        if isinstance(model, GraphModelWrapper):
            model_transformer = PT2ModelTransformer(model)
            model = model.model
        else:
            model_transformer = PTModelTransformer(model)

        transformation_layout = TransformationLayout()
        is_all_8bit = all(wc_params.compression_config.num_bits == 8 for wc_params in weight_compression_parameters)
        for wc_params in weight_compression_parameters:
            compression_config = wc_params.compression_config
            if compression_config.mode in [
                CompressWeightsMode.NF4,
                CompressWeightsMode.E2M1,
            ]:
                msg = f"{compression_config.mode.value} is not supported."
                raise nncf.ParameterNotSupportedError(msg)

            weight_node = get_const_node(wc_params.node_with_weight, wc_params.weight_port_id, graph)
            weight_name = weight_node.layer_attributes.name
            weight = get_const_data(weight_node, model)
            if weight is None:
                msg = f"Could not find a torch.nn.Parameter in the model by name {weight_name}."
                raise nncf.InternalError(msg)

            precomputed_compressed_weights = precomputed_compressed_weights or {}
            # calculates compressed weights and decompression parameters
            compressed_weight = compress_weight(
                Tensor(weight),
                wc_params.reduction_axes,
                compression_config,
                precomputed_compressed_weights.get(wc_params.weight_name),
            )

            if compression_format == CompressionFormat.DQ:
                command = self.get_dq_insertion_command(compressed_weight, wc_params, model, graph, weight_node)
            else:
                rank = advanced_parameters.lora_adapter_rank
                command = self.get_fq_insertion_command(
                    compressed_weight, wc_params, weight.shape, compression_format, rank, is_all_8bit
                )
            transformation_layout.register(command)

        # To have FQ's with requires_grad=True only
        model.requires_grad_(False)

        # apply transformations
        transformed_model = model_transformer.transform(transformation_layout)

        return transformed_model

    @staticmethod
    def get_ignored_patterns() -> GraphPattern:
        return create_rope()


class PTAWQAlgoAlgoBackend(AWQAlgoBackend, PTWeightCompressionAlgoBackend):
    @staticmethod
    def get_awq_patterns():
        return get_awq_patterns(
            MATMUL_METATYPES,
            PTMulMetatype,
            ATOMIC_ACTIVATIONS_OPERATIONS[GraphPattern.METATYPE_ATTR],
        )

    @staticmethod
    def scale_insertion_command(
        source_node: NNCFNode,
        next_nodes,
        source_output_port_id: int,
        scale: torch.Tensor,
    ) -> PTSharedFnInsertionCommand:
        input_port_id = 0
        target_points = []
        for node in next_nodes:
            target_points.append(
                PTTargetPoint(
                    PTWeightCompressionAlgoBackend.TARGET_TYPE_TO_PT_INS_TYPE_MAP[TargetType.PRE_LAYER_OPERATION],
                    node.node_name,
                    input_port_id=input_port_id,
                )
            )

        sq_multiply = SQMultiply(scale.shape)
        sq_multiply.scale = scale

        return PT2InsertionCommand(target_points, sq_multiply)


class PTMixedPrecisionAlgoBackend(MixedPrecisionAlgoBackend, PTWeightCompressionAlgoBackend):
    @staticmethod
    def mean_variance_statistic_collector(
        reduction_axes: tuple[int], subset_size: Optional[int] = None
    ) -> TensorCollector:
        reducer = MeanVarianceReducer(reduction_axes)
        aggregator = MeanAggregator(num_samples=subset_size)
        collector = TensorCollector(MeanVarianceTensorStatistic)
        collector.register_statistic_branch(MeanVarianceTensorStatistic.MEAN_VARIANCE_STAT, reducer, aggregator)
        return collector

    @staticmethod
    def max_variance_statistic_collector(
        reduction_axes: tuple[int], subset_size: Optional[int] = None
    ) -> TensorCollector:
        reducer = MaxVarianceReducer(reduction_axes)
        aggregator = MeanAggregator(num_samples=subset_size)
        collector = TensorCollector(MaxVarianceTensorStatistic)
        collector.register_statistic_branch(MaxVarianceTensorStatistic.MAX_VARIANCE_STAT, reducer, aggregator)
        return collector

    @staticmethod
    def mean_abs_max_statistic_collector(
        reduction_axes: tuple[int], subset_size: Optional[int] = None
    ) -> TensorCollector:
        reducer = MeanAbsMaxReducer(reduction_axes)
        aggregator = MeanAggregator(num_samples=subset_size)
        collector = TensorCollector(MeanMagnitudeTensorStatistic)
        collector.register_statistic_branch(MeanMagnitudeTensorStatistic.MEAN_MAGNITUDE_STAT, reducer, aggregator)
        return collector
