# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Dict, Iterable, List, Optional, Tuple

import openvino as ov
from openvino.runtime import opset13 as opset

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.utils import get_reduction_axes
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.openvino.graph.metatypes import openvino_metatypes as om
from nncf.openvino.graph.metatypes.groups import ATOMIC_ACTIVATIONS_OPERATIONS
from nncf.openvino.graph.model_transformer import OVModelTransformer
from nncf.openvino.graph.node_utils import get_const_value
from nncf.openvino.graph.node_utils import get_weight_channel_axes
from nncf.openvino.graph.transformations.command_creation import OVCommandCreator
from nncf.openvino.graph.transformations.commands import OVTargetPoint
from nncf.openvino.rt_info import dump_parameters
from nncf.openvino.statistics.collectors import get_raw_stat_collector
from nncf.parameters import CompressWeightsMode
from nncf.quantization.algorithms.weight_compression.awq_patterns import get_awq_patterns
from nncf.quantization.algorithms.weight_compression.backend import AWQAlgoBackend
from nncf.quantization.algorithms.weight_compression.backend import WeightCompressionAlgoBackend
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionConfig
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionParameters
from nncf.quantization.algorithms.weight_compression.lora_correction import LoraCorrectionAlgorithm
from nncf.quantization.algorithms.weight_compression.weight_lowering import compress_weight
from nncf.tensor import Tensor
from nncf.tensor.definitions import TensorDataType


class OVWeightCompressionAlgoBackend(WeightCompressionAlgoBackend):
    def __init__(self, model: ov.Model, name_to_node_mapping: Dict = None):
        if name_to_node_mapping is None:
            self.name_to_node_mapping = OVModelTransformer._get_name_to_node_mapping(model)
        else:
            self.name_to_node_mapping = name_to_node_mapping

    @property
    def matmul_metatypes(self) -> List[OperatorMetatype]:
        return [om.OVMatMulMetatype]

    @property
    def convolution_metatypes(self) -> List[OperatorMetatype]:
        return [
            om.OVConvolutionMetatype,
            om.OVDepthwiseConvolutionMetatype,
            om.OVConvolutionBackpropDataMetatype,
            om.OVGroupConvolutionMetatype,
            om.OVGroupConvolutionBackpropDataMetatype,
        ]

    @property
    def embedding_metatypes(self) -> List[OperatorMetatype]:
        return [om.OVEmbeddingMetatype]

    @staticmethod
    def is_node_with_weights(node: NNCFNode, graph: NNCFGraph) -> bool:
        return node.layer_attributes and node.layer_attributes.constant_attributes

    @staticmethod
    def get_reduction_axes(node_with_weight: NNCFNode, weight_port_id: int, graph: NNCFGraph) -> Optional[Tuple[int]]:
        channel_axes = get_weight_channel_axes(node_with_weight)
        const_shape = node_with_weight.layer_attributes.constant_attributes[weight_port_id]["shape"]
        return get_reduction_axes(channel_axes, const_shape)

    @staticmethod
    def target_point(target_type: TargetType, target_node_name: str, port_id: int) -> OVTargetPoint:
        return OVTargetPoint(target_type, target_node_name, port_id)

    @staticmethod
    def raw_statistic_collector(num_samples: Optional[int] = None) -> TensorCollector:
        return get_raw_stat_collector(num_samples)

    @staticmethod
    def get_activation_port_id(node: NNCFNode, nncf_graph: NNCFGraph) -> int:
        constant_ports = node.layer_attributes.get_const_port_ids()
        activation_ports = [
            e.input_port_id for e in nncf_graph.get_input_edges(node) if e.input_port_id not in constant_ports
        ]
        assert len(activation_ports) == 1
        return activation_ports[0]

    @staticmethod
    def get_weight_names_and_port_ids(node: NNCFNode, graph: NNCFGraph) -> List[Tuple[str, int]]:
        result = []
        for weight_port_id in node.layer_attributes.get_const_port_ids():
            weight_name = node.layer_attributes.constant_attributes[weight_port_id]["name"]
            result.append((weight_name, weight_port_id))
        return result

    def get_weight(self, node_with_weight: NNCFNode, weight_port_id: int, model: ov.Model, graph: NNCFGraph) -> Tensor:
        weight_name = node_with_weight.layer_attributes.constant_attributes[weight_port_id]["name"]
        weight_node = self.name_to_node_mapping[weight_name]
        weight_tensor = get_const_value(weight_node)
        return Tensor(weight_tensor)

    def get_weight_dtype(
        self, node_with_weight: NNCFNode, weight_port_id: int, model: ov.Model, graph: NNCFGraph
    ) -> TensorDataType:
        ov_type_name = node_with_weight.layer_attributes.constant_attributes[weight_port_id]["dtype"]
        dtype_map = {
            "f16": TensorDataType.float16,
            "bf16": TensorDataType.bfloat16,
            "f32": TensorDataType.float32,
            "f64": TensorDataType.float64,
            "i8": TensorDataType.int8,
            "i32": TensorDataType.int32,
            "i64": TensorDataType.int64,
            "u8": TensorDataType.uint8,
        }
        return dtype_map.get(ov_type_name)

    @staticmethod
    def get_weight_shape(node_with_weight: NNCFNode, weight_port_id: int, graph: NNCFGraph) -> Tuple:
        return node_with_weight.layer_attributes.constant_attributes[weight_port_id]["shape"]

    def set_weight(
        self, node_with_weight: NNCFNode, weight_port_id: int, model: ov.Model, graph: NNCFGraph, weight: Tensor
    ):
        node_with_const = self.name_to_node_mapping[node_with_weight.node_name]

        const_port = node_with_const.input(weight_port_id)
        const_node = node_with_const.input_value(weight_port_id).get_node()

        shared_memory = True
        if const_node.get_element_type() == ov.Type.bf16:
            # Shared memory does not work for BF16 precision
            shared_memory = False

        new_const_node = ov.runtime.op.Constant(weight.data, shared_memory=shared_memory)
        new_const_node.set_friendly_name(const_node.get_friendly_name())
        const_port.replace_source_output(new_const_node.output(0))

        const_name = node_with_weight.layer_attributes.constant_attributes[weight_port_id]["name"]
        self.name_to_node_mapping[const_name] = new_const_node

        new_output = new_const_node.output(0)
        for target_input in const_node.output(0).get_target_inputs():
            target_input.replace_source_output(new_output)

        del const_node

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
        reduction_axes: Tuple[int, ...],
        const_node_name: str,
        weight_port_id: int,
        const_dtype,
        should_add_convert_node: bool,
        layer_scales: Optional[Tensor] = None,
        layer_zero_points: Optional[Tensor] = None,
    ):
        scale_dtype = ov.Type.f16
        if compression_config.mode == CompressWeightsMode.NF4:
            compression_dtype = ov.Type.nf4
        elif compression_config.mode == CompressWeightsMode.E2M1:
            compression_dtype = ov.Type.f4e2m1
            scale_dtype = ov.Type.f8e8m0
        elif compression_config.mode == CompressWeightsMode.INT4_SYM:
            compression_dtype = ov.Type.i4
        elif compression_config.mode == CompressWeightsMode.INT4_ASYM:
            compression_dtype = ov.Type.u4
        elif compression_config.mode == CompressWeightsMode.INT8_SYM:
            compression_dtype = ov.Type.i8
        elif compression_config.mode == CompressWeightsMode.INT8_ASYM:
            compression_dtype = ov.Type.u8
        else:
            raise ValueError(f"{compression_config.mode.value} is not supported.")

        original_shape = weight.shape
        compressed_weight = compress_weight(weight, reduction_axes, compression_config, layer_scales, layer_zero_points)

        compressed_const = opset.constant(compressed_weight.tensor.data, dtype=compression_dtype, name=const_node_name)
        converted_const = opset.convert(compressed_const, ov.Type.f16)
        if compressed_weight.zero_point is not None and compressed_weight.tensor.dtype == TensorDataType.uint8:
            zero_point_const = opset.constant(
                compressed_weight.zero_point.data,
                dtype=compression_dtype,
                name=f"{const_node_name}/zero_point",
            )
            converted_zero_point = opset.convert(zero_point_const, ov.Type.f16)
            converted_const = opset.subtract(
                converted_const, converted_zero_point, name=f"{const_node_name}/zero_point/subtract"
            )

        scale_const = opset.constant(compressed_weight.scale.data, dtype=scale_dtype, name=f"{const_node_name}/scale")
        if scale_dtype != ov.Type.f16:
            scale_const = opset.convert(scale_const, ov.Type.f16)

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

    def transform_model(
        self,
        model: ov.Model,
        graph: NNCFGraph,
        weight_compression_parameters: Iterable[WeightCompressionParameters],
        precomputed_scales: Dict[str, Tensor] = None,
        precomputed_zero_points: Dict[str, Tensor] = None,
        lora_correction_algo: LoraCorrectionAlgorithm = None,
    ) -> ov.Model:
        for wc_params in weight_compression_parameters:
            const_attributes = wc_params.node_with_weight.layer_attributes.constant_attributes[wc_params.weight_port_id]
            const_node_name = const_attributes["name"]
            const_node = self.name_to_node_mapping[const_node_name]
            const_node_output = const_node.output(0)
            const_dtype = const_node_output.get_element_type()
            weight = Tensor(get_const_value(const_node))

            should_add_convert_node = False
            if const_dtype != ov.Type.f16:
                for inp in const_node_output.get_target_inputs():
                    if inp.get_node().get_type_name() != "Convert":
                        should_add_convert_node = True
                        break

            layer_scales = None if precomputed_scales is None else precomputed_scales.get(wc_params.weight_name)
            layer_zero_points = (
                None if precomputed_zero_points is None else precomputed_zero_points.get(wc_params.weight_name)
            )
            import os
            os.environ["CURRENT_NODE_NAME"] = wc_params.weight_name
            mul, compressed_weight = self._create_compression_subgraph(
                weight=weight,
                compression_config=wc_params.compression_config,
                reduction_axes=wc_params.reduction_axes,
                const_node_name=const_node_name,
                weight_port_id=wc_params.weight_port_id,
                const_dtype=const_dtype,
                should_add_convert_node=should_add_convert_node,
                layer_scales=layer_scales,
                layer_zero_points=layer_zero_points,
            )

            mul_output = mul.output(0)
            for target_input in const_node.output(0).get_target_inputs():
                target_input.replace_source_output(mul_output)

            if lora_correction_algo is not None and lora_correction_algo.is_applicable(wc_params):
                adapters = lora_correction_algo.calculate_adapters(weight, compressed_weight, wc_params)
                self.insert_adapters(wc_params, *adapters, int8_lora=lora_correction_algo.use_int8_adapters)

        # reset name_to_node_mapping
        self.name_to_node_mapping = None

        return model

    @staticmethod
    def dump_parameters(
        model: ov.Model, parameters: Dict, algo_name: Optional[str] = "quantization", path: Optional[List] = None
    ) -> None:
        dump_parameters(model, parameters, algo_name, path)


class OVAWQAlgoAlgoBackend(AWQAlgoBackend, OVWeightCompressionAlgoBackend):
    @staticmethod
    def get_awq_patterns():
        return get_awq_patterns(om.OVMatMulMetatype, om.OVMultiplyMetatype, ATOMIC_ACTIVATIONS_OPERATIONS)

    @staticmethod
    def scale_insertion_command(source_node, next_nodes, source_node_output_port, scale):
        return OVCommandCreator.multiply_insertion_command(
            source_node, next_nodes, source_node_output_port, scale, f"{source_node.node_name}/awq_mul"
        )
