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

import openvino as ov
from openvino import opset13 as opset

import nncf
from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.patterns.patterns import GraphPattern
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.utils import get_reduction_axes
from nncf.common.tensor_statistics.statistic_point import StatisticPoint
from nncf.common.utils.caching import disable_results_caching
from nncf.experimental.common.tensor_statistics.collectors import MeanAggregator
from nncf.experimental.common.tensor_statistics.collectors import NoopAggregator
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.experimental.common.tensor_statistics.statistics import MaxVarianceTensorStatistic
from nncf.experimental.common.tensor_statistics.statistics import MeanMagnitudeTensorStatistic
from nncf.experimental.common.tensor_statistics.statistics import MeanVarianceTensorStatistic
from nncf.experimental.common.tensor_statistics.statistics import WCTensorStatistic
from nncf.openvino.graph.metatypes import openvino_metatypes as om
from nncf.openvino.graph.metatypes.groups import ATOMIC_ACTIVATIONS_OPERATIONS
from nncf.openvino.graph.model_transformer import OVModelTransformer
from nncf.openvino.graph.node_utils import convert_op
from nncf.openvino.graph.node_utils import create_ov_codebook_subgraph
from nncf.openvino.graph.node_utils import create_ov_const_from_tensor
from nncf.openvino.graph.node_utils import get_const_value_as_numpy_tensor
from nncf.openvino.graph.node_utils import get_const_value_as_ov_tensor
from nncf.openvino.graph.node_utils import get_weight_channel_axes
from nncf.openvino.graph.transformations.command_creation import OVCommandCreator
from nncf.openvino.graph.transformations.commands import OVTargetPoint
from nncf.openvino.optimized_functions import clear_ov_model_cache
from nncf.openvino.optimized_functions.models import OV_MODEL_CACHE
from nncf.openvino.quantization.ignored_patterns import create_rope
from nncf.openvino.rt_info import dump_parameters
from nncf.openvino.statistics.collectors import OVMaxVarianceReducer
from nncf.openvino.statistics.collectors import OVMeanAbsMaxReducer
from nncf.openvino.statistics.collectors import OVMeanReducer
from nncf.openvino.statistics.collectors import OVMeanVarianceReducer
from nncf.openvino.statistics.collectors import OVShapeReducer
from nncf.parameters import CompressionFormat
from nncf.parameters import CompressWeightsMode
from nncf.quantization.advanced_parameters import AdvancedCompressionParameters
from nncf.quantization.algorithms.weight_compression.awq_patterns import get_awq_patterns
from nncf.quantization.algorithms.weight_compression.backend import AWQAlgoBackend
from nncf.quantization.algorithms.weight_compression.backend import MixedPrecisionAlgoBackend
from nncf.quantization.algorithms.weight_compression.backend import WeightCompressionAlgoBackend
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionConfig
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionParameters
from nncf.quantization.algorithms.weight_compression.lora_correction import LoraCorrectionAlgorithm
from nncf.quantization.algorithms.weight_compression.parameters import CompressedWeight
from nncf.quantization.algorithms.weight_compression.weight_lowering import compress_weight
from nncf.tensor import Tensor
from nncf.tensor.definitions import TensorDataType
from nncf.tensor.functions.openvino_numeric import DTYPE_MAP_REV


class OVWeightCompressionAlgoBackend(WeightCompressionAlgoBackend):
    def __init__(self, model: ov.Model, name_to_node_mapping: dict = None):
        if name_to_node_mapping is None:
            self.name_to_node_mapping = OVModelTransformer._get_name_to_node_mapping(model)
        else:
            self.name_to_node_mapping = name_to_node_mapping

    @property
    def matmul_metatypes(self) -> list[OperatorMetatype]:
        return [om.OVMatMulMetatype]

    @property
    def convolution_metatypes(self) -> list[OperatorMetatype]:
        return [
            om.OVConvolutionMetatype,
            om.OVDepthwiseConvolutionMetatype,
            om.OVConvolutionBackpropDataMetatype,
            om.OVGroupConvolutionMetatype,
            om.OVGroupConvolutionBackpropDataMetatype,
        ]

    @property
    def embedding_metatypes(self) -> list[OperatorMetatype]:
        return [om.OVEmbeddingMetatype]

    @staticmethod
    def is_node_with_weights(node: NNCFNode, graph: NNCFGraph) -> bool:
        return node.layer_attributes and node.layer_attributes.constant_attributes

    @staticmethod
    def get_reduction_axes(node_with_weight: NNCFNode, weight_port_id: int, graph: NNCFGraph) -> Optional[tuple[int]]:
        channel_axes = get_weight_channel_axes(node_with_weight)
        const_shape = node_with_weight.layer_attributes.constant_attributes[weight_port_id]["shape"]
        return get_reduction_axes(channel_axes, const_shape)

    @staticmethod
    def target_point(target_type: TargetType, target_node_name: str, port_id: int) -> OVTargetPoint:
        return OVTargetPoint(target_type, target_node_name, port_id)

    def mean_statistic_collector(
        self, reduction_axes: tuple[int], subset_size: Optional[int] = None
    ) -> TensorCollector:
        mean_reducer = OVMeanReducer(reduction_axes, inplace=True)
        shape_reducer = OVShapeReducer(inplace=True)
        collector = TensorCollector(WCTensorStatistic)
        collector.register_statistic_branch(WCTensorStatistic.MEAN_STAT, mean_reducer, NoopAggregator(subset_size))
        collector.register_statistic_branch(WCTensorStatistic.SHAPE_STAT, shape_reducer, NoopAggregator(subset_size))
        return collector

    @staticmethod
    def get_activation_port_id(node: NNCFNode, nncf_graph: NNCFGraph) -> int:
        if node.layer_attributes.input_attributes["transpose"]:
            msg = "Transposed input is not supported"
            raise nncf.UnsupportedModelError(msg)
        constant_ports = node.layer_attributes.get_const_port_ids()
        activation_ports = [
            e.input_port_id for e in nncf_graph.get_input_edges(node) if e.input_port_id not in constant_ports
        ]
        assert len(activation_ports) == 1
        return activation_ports[0]

    @staticmethod
    def get_weight_names_and_port_ids(node: NNCFNode, graph: NNCFGraph) -> list[tuple[str, int]]:
        result = []
        for weight_port_id in node.layer_attributes.get_const_port_ids():
            weight_name = node.layer_attributes.constant_attributes[weight_port_id]["name"]
            result.append((weight_name, weight_port_id))
        return result

    def get_weight(self, node_with_weight: NNCFNode, weight_port_id: int, model: ov.Model, graph: NNCFGraph) -> Tensor:
        weight_name = node_with_weight.layer_attributes.constant_attributes[weight_port_id]["name"]
        weight_node = self.name_to_node_mapping[weight_name]
        weight_tensor = get_const_value_as_numpy_tensor(weight_node)
        return Tensor(weight_tensor)

    def get_weight_dtype(
        self, node_with_weight: NNCFNode, weight_port_id: int, model: ov.Model, graph: NNCFGraph
    ) -> TensorDataType:
        ov_type_name = node_with_weight.layer_attributes.constant_attributes[weight_port_id]["dtype"]
        ov_type = getattr(ov.Type, ov_type_name)
        return DTYPE_MAP_REV[ov_type]

    @staticmethod
    def get_weight_shape(node_with_weight: NNCFNode, weight_port_id: int, graph: NNCFGraph) -> tuple:
        return node_with_weight.layer_attributes.constant_attributes[weight_port_id]["shape"]

    def set_weight(
        self, node_with_weight: NNCFNode, weight_port_id: int, model: ov.Model, graph: NNCFGraph, weight: Tensor
    ):
        const_op_friendly_name = node_with_weight.layer_attributes.constant_attributes[weight_port_id]["name"]
        const_op = self.name_to_node_mapping[const_op_friendly_name]

        dtype = const_op.get_element_type()
        name = const_op.get_friendly_name()
        new_const_op = create_ov_const_from_tensor(weight, dtype, name)
        self.name_to_node_mapping[const_op_friendly_name] = new_const_op

        new_output = new_const_op.output(0)
        for target_input in const_op.output(0).get_target_inputs():
            target_input.replace_source_output(new_output)

        del const_op

    def insert_adapters(
        self, wc_params: WeightCompressionParameters, lora_A: Tensor, lora_B: Tensor, int8_lora: bool
    ) -> None:
        input_node = self.name_to_node_mapping[wc_params.node_with_weight.node_name].input_value(0)
        activation_dtype = input_node.get_element_type()
        should_add_convert_node = activation_dtype != ov.Type.f16
        mm_node = self.name_to_node_mapping[wc_params.node_with_weight.node_name]

        if int8_lora:
            const_node_name = wc_params.node_with_weight.node_name
            int8_compression_config = WeightCompressionConfig(mode=CompressWeightsMode.INT8_ASYM, group_size=-1)
            A_W, _ = self._create_compression_subgraph(
                weight=lora_A,
                compression_config=int8_compression_config,
                reduction_axes=wc_params.reduction_axes,
                const_node_name=const_node_name + "_lora_A",
                weight_port_id=1,
                const_dtype=activation_dtype,
                should_add_convert_node=should_add_convert_node,
            )
            B_W, _ = self._create_compression_subgraph(
                weight=lora_B,
                compression_config=int8_compression_config,
                reduction_axes=wc_params.reduction_axes,
                const_node_name=const_node_name + "_lora_B",
                weight_port_id=1,
                const_dtype=activation_dtype,
                should_add_convert_node=should_add_convert_node,
            )
        else:
            A_W = opset.constant(lora_A.data)
            B_W = opset.constant(lora_B.data)

        A_MM = opset.matmul(input_node, A_W, transpose_a=False, transpose_b=True)
        B_MM = opset.matmul(A_MM, B_W, transpose_a=False, transpose_b=True)

        node_output_port = mm_node.output(0)
        node_output_source_ports = node_output_port.get_target_inputs()
        add = opset.add(mm_node, B_MM)
        for node_output_source_port in node_output_source_ports:
            node_output_source_port.replace_source_output(add.output(0))

    def _create_compression_subgraph(
        self,
        weight: Tensor,
        compression_config: WeightCompressionConfig,
        reduction_axes: tuple[int, ...],
        const_node_name: str,
        weight_port_id: int,
        const_dtype,
        should_add_convert_node: bool,
        precomputed_compressed_weight: Optional[CompressedWeight] = None,
    ):
        scale_dtype = ov.Type.f16
        if compression_config.mode == CompressWeightsMode.NF4:
            compression_dtype = ov.Type.nf4
        elif compression_config.mode == CompressWeightsMode.MXFP4:
            compression_dtype = ov.Type.f4e2m1
            scale_dtype = ov.Type.f8e8m0
        elif compression_config.mode == CompressWeightsMode.MXFP8_E4M3:
            compression_dtype = ov.Type.f8e4m3
            scale_dtype = ov.Type.f8e8m0
        elif compression_config.mode == CompressWeightsMode.INT4_SYM:
            compression_dtype = ov.Type.i4
        elif compression_config.mode == CompressWeightsMode.INT4_ASYM:
            compression_dtype = ov.Type.u4
        elif compression_config.mode == CompressWeightsMode.INT8_SYM:
            compression_dtype = ov.Type.i8
        elif compression_config.mode == CompressWeightsMode.INT8_ASYM:
            compression_dtype = ov.Type.u8
        elif compression_config.is_codebook:
            compression_dtype = None
        else:
            msg = f"{compression_config.mode.value} is not supported."
            raise nncf.ParameterNotSupportedError(msg)

        original_shape = weight.shape

        with disable_results_caching(OV_MODEL_CACHE):
            compressed_weight = compress_weight(
                weight,
                reduction_axes,
                compression_config,
                precomputed_compressed_weight,
            )

        if compression_config.is_codebook:
            n_quants = compressed_weight.codebook.size - 1
            compression_dtype = ov.Type.u16 if n_quants > 255 else (ov.Type.u8 if n_quants > 15 else ov.Type.u4)
            converted_const = create_ov_codebook_subgraph(
                codebook=compressed_weight.codebook
                if compression_config.mode == CompressWeightsMode.CODEBOOK
                else compressed_weight.codebook.as_openvino_tensor().astype(TensorDataType.f8e4m3),
                indexes=compressed_weight.tensor,
                dtype=compression_dtype,
                name=const_node_name,
            )
        else:
            compressed_const = create_ov_const_from_tensor(
                compressed_weight.tensor, compression_dtype, name=const_node_name
            )
            converted_const = opset.convert(compressed_const, ov.Type.f16)

            if compressed_weight.zero_point is not None:
                zero_point_const = create_ov_const_from_tensor(
                    compressed_weight.zero_point, compression_dtype, name=f"{const_node_name}/zero_point"
                )
                zero_point_const = opset.convert(zero_point_const, ov.Type.f16)
                converted_const = opset.subtract(
                    converted_const, zero_point_const, name=f"{const_node_name}/zero_point/subtract"
                )

        scale_const = create_ov_const_from_tensor(compressed_weight.scale, scale_dtype, name=f"{const_node_name}/scale")
        scale_const = convert_op(scale_const, ov.Type.f16)

        mul = opset.multiply(
            converted_const,
            scale_const,
            name=f"{const_node_name}/fq_weights_{weight_port_id}",
        )

        if compression_config.group_size != -1:
            mul = opset.reshape(mul, output_shape=original_shape, special_zero=False)

        if should_add_convert_node:
            mul = opset.convert(mul, const_dtype, name=f"{const_node_name}/fq_weights_{weight_port_id}/convert")
        return mul, compressed_weight

    def _replace_node(self, src_node, new_node):
        """
        Replace all occurrences of a source constant node with a new node in the computational graph with
        the elimination of unnecessary convert operations.

        :param src_node: The source node (typically a constant node) to be replaced.
        :param new_node: The new node that will replace the source node in the graph.
        :return: None. The graph is modified in place.
        """
        new_output_type = new_node.get_element_type()
        for target_input in src_node.output(0).get_target_inputs():
            target_node = target_input.get_node()
            if target_node.get_type_name() == "Convert" and target_node.get_element_type() == new_output_type:
                self._replace_node(target_node, new_node)
            else:
                target_input.replace_source_output(new_node.output(0))

    def transform_model(
        self,
        model: ov.Model,
        graph: NNCFGraph,
        weight_compression_parameters: Iterable[WeightCompressionParameters],
        precomputed_compressed_weights: Optional[dict[str, CompressedWeight]] = None,
        lora_correction_algo: Optional[LoraCorrectionAlgorithm] = None,
        compression_format: CompressionFormat = CompressionFormat.DQ,
        advanced_parameters: AdvancedCompressionParameters = AdvancedCompressionParameters(),
    ) -> ov.Model:
        for wc_params in weight_compression_parameters:
            const_attributes = wc_params.node_with_weight.layer_attributes.constant_attributes[wc_params.weight_port_id]
            const_node_name = const_attributes["name"]
            const_node = self.name_to_node_mapping[const_node_name]
            const_node_output = const_node.output(0)
            const_dtype = const_node_output.get_element_type()
            # Creation of ov.Tensor is required for two reasons:
            #   1. To be able to process BF16 weight properly
            #   2. To indicate that it is allowed for the compressed constant to be returned as int4/uint4 if needed
            weight = Tensor(get_const_value_as_ov_tensor(const_node))

            should_add_convert_node = False
            if const_dtype != ov.Type.f16:
                for inp in const_node_output.get_target_inputs():
                    if inp.get_node().get_type_name() != "Convert":
                        should_add_convert_node = True
                        break

            mul, compressed_weight = self._create_compression_subgraph(
                weight=weight,
                compression_config=wc_params.compression_config,
                reduction_axes=wc_params.reduction_axes,
                const_node_name=const_node_name,
                weight_port_id=wc_params.weight_port_id,
                const_dtype=const_dtype,
                should_add_convert_node=should_add_convert_node,
                precomputed_compressed_weight=None
                if precomputed_compressed_weights is None
                else precomputed_compressed_weights.get(wc_params.weight_name),
            )

            self._replace_node(const_node, mul)

            if lora_correction_algo is not None and lora_correction_algo.is_applicable(wc_params):
                # These tensors can potentially be in ov backend
                weight = weight.as_numpy_tensor()
                compressed_weight.tensor = compressed_weight.tensor.as_numpy_tensor()
                if compressed_weight.zero_point is not None:
                    compressed_weight.zero_point = compressed_weight.zero_point.as_numpy_tensor()
                adapters = lora_correction_algo.calculate_adapters(weight, compressed_weight, wc_params)
                self.insert_adapters(wc_params, *adapters, int8_lora=lora_correction_algo.use_int8_adapters)
        self.name_to_node_mapping = None

        clear_ov_model_cache()

        return model

    @staticmethod
    def dump_parameters(
        model: ov.Model, parameters: dict, algo_name: Optional[str] = "quantization", path: Optional[list] = None
    ) -> None:
        dump_parameters(model, parameters, algo_name, path)

    @staticmethod
    def get_filter_fn_for_statistics(activation_port_id: int, algorithm_key: str) -> Callable[[StatisticPoint], bool]:
        def filter_func(point: StatisticPoint) -> bool:
            return (
                algorithm_key in point.algorithm_to_tensor_collectors
                and point.target_point.type == TargetType.POST_LAYER_OPERATION
                and point.target_point.port_id == activation_port_id
            )

        return filter_func

    @staticmethod
    def get_ignored_patterns() -> GraphPattern:
        return create_rope()


class OVTensorWeightCompressionAlgoBackend(OVWeightCompressionAlgoBackend):
    """
    OpenVINO backend for weight compression algorithms that fetches model weights as openvino.Tensor instances.
    This allows to natively process BF16/FP16 weights.
    """

    def get_weight(self, node_with_weight: NNCFNode, weight_port_id: int, model: ov.Model, graph: NNCFGraph) -> Tensor:
        weight_name = node_with_weight.layer_attributes.constant_attributes[weight_port_id]["name"]
        weight_node = self.name_to_node_mapping[weight_name]
        weight_tensor = get_const_value_as_ov_tensor(weight_node)
        return Tensor(weight_tensor)


class OVAWQAlgoAlgoBackend(AWQAlgoBackend, OVWeightCompressionAlgoBackend):
    @staticmethod
    def get_awq_patterns():
        return get_awq_patterns(om.OVMatMulMetatype, om.OVMultiplyMetatype, ATOMIC_ACTIVATIONS_OPERATIONS)

    @staticmethod
    def scale_insertion_command(source_node, next_nodes, source_node_output_port, scale):
        return OVCommandCreator.multiply_insertion_command(
            source_node, next_nodes, source_node_output_port, scale, f"{source_node.node_name}/awq_mul"
        )


class OVMixedPrecisionAlgoBackend(MixedPrecisionAlgoBackend, OVWeightCompressionAlgoBackend):
    @staticmethod
    def mean_variance_statistic_collector(
        reduction_axes: tuple[int], subset_size: Optional[int] = None
    ) -> TensorCollector:
        reducer = OVMeanVarianceReducer(reduction_axes, inplace=True)
        aggregator = MeanAggregator(num_samples=subset_size)
        collector = TensorCollector(MeanVarianceTensorStatistic)
        collector.register_statistic_branch(MeanVarianceTensorStatistic.MEAN_VARIANCE_STAT, reducer, aggregator)
        return collector

    @staticmethod
    def max_variance_statistic_collector(
        reduction_axes: tuple[int], subset_size: Optional[int] = None
    ) -> TensorCollector:
        reducer = OVMaxVarianceReducer(reduction_axes, inplace=True)
        aggregator = MeanAggregator(num_samples=subset_size)
        collector = TensorCollector(MaxVarianceTensorStatistic)
        collector.register_statistic_branch(MaxVarianceTensorStatistic.MAX_VARIANCE_STAT, reducer, aggregator)
        return collector

    @staticmethod
    def mean_abs_max_statistic_collector(
        reduction_axes: tuple[int], subset_size: Optional[int] = None
    ) -> TensorCollector:
        reducer = OVMeanAbsMaxReducer(reduction_axes, inplace=True)
        aggregator = MeanAggregator(num_samples=subset_size)
        collector = TensorCollector(MeanMagnitudeTensorStatistic)
        collector.register_statistic_branch(MeanMagnitudeTensorStatistic.MEAN_MAGNITUDE_STAT, reducer, aggregator)
        return collector
