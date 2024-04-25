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
from nncf.experimental.tensor.tensor import Tensor
from nncf.openvino.graph.metatypes import openvino_metatypes as om
from nncf.openvino.graph.model_transformer import OVModelTransformer
from nncf.openvino.graph.node_utils import get_const_value
from nncf.openvino.graph.node_utils import get_weight_channel_axes
from nncf.openvino.graph.transformations.commands import OVTargetPoint
from nncf.openvino.rt_info import dump_parameters
from nncf.openvino.statistics.collectors import get_raw_stat_collector
from nncf.parameters import CompressWeightsMode
from nncf.quantization.algorithms.weight_compression.awq_patterns import get_awq_patterns
from nncf.quantization.algorithms.weight_compression.backend import WeightCompressionAlgoBackend
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionParameters
from nncf.quantization.algorithms.weight_compression.weight_lowering import compress_weight


class OVWeightCompressionAlgoBackend(WeightCompressionAlgoBackend):
    def __init__(self, model: ov.Model):
        self.name_to_node_mapping = OVModelTransformer._get_name_to_node_mapping(model)

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

    def set_weight(
        self, node_with_weight: NNCFNode, weight_port_id: int, model: ov.Model, graph: NNCFGraph, weight: Tensor
    ):
        node_with_const = self.name_to_node_mapping[node_with_weight.node_name]

        const_port = node_with_const.input(weight_port_id)
        const_node = node_with_const.input_value(weight_port_id).get_node()

        new_const_node = ov.runtime.op.Constant(weight.data, shared_memory=True)
        new_const_node.set_friendly_name(const_node.get_friendly_name())
        const_port.replace_source_output(new_const_node.output(0))

        const_name = node_with_weight.layer_attributes.constant_attributes[weight_port_id]["name"]
        self.name_to_node_mapping[const_name] = new_const_node

        new_output = new_const_node.output(0)
        for target_input in const_node.output(0).get_target_inputs():
            target_input.replace_source_output(new_output)

        del const_node

    def transform_model(
        self, model: ov.Model, graph: NNCFGraph, weight_compression_parameters: Iterable[WeightCompressionParameters]
    ) -> ov.Model:
        for wc_params in weight_compression_parameters:
            compression_config = wc_params.compression_config
            if compression_config.mode == CompressWeightsMode.NF4:
                compression_dtype = ov.Type.nf4
            elif compression_config.mode in [
                CompressWeightsMode.INT8_ASYM,
                CompressWeightsMode.INT8_SYM,
                CompressWeightsMode.INT8,
                CompressWeightsMode.INT4_ASYM,
                CompressWeightsMode.INT4_SYM,
            ]:
                if compression_config.mode in [CompressWeightsMode.INT4_ASYM, CompressWeightsMode.INT4_SYM]:
                    compression_dtype = ov.Type.u4
                else:
                    compression_dtype = ov.Type.u8
            else:
                raise ValueError(f"{compression_config.mode.value} is not supported.")

            const_attributes = wc_params.node_with_weight.layer_attributes.constant_attributes[wc_params.weight_port_id]
            const_node_name = const_attributes["name"]
            const_node = self.name_to_node_mapping[const_node_name]
            const_dtype = const_node.output(0).get_element_type().to_dtype()

            weight = Tensor(get_const_value(const_node))
            original_shape = weight.shape
            compressed_weight = compress_weight(weight, wc_params.reduction_axes, compression_config)

            compressed_const = opset.constant(
                compressed_weight.tensor.data, dtype=compression_dtype, name=const_node_name
            )
            converted_const = opset.convert(compressed_const, const_dtype)
            if compressed_weight.zero_point is not None:
                zero_point_const = opset.constant(
                    compressed_weight.zero_point.data,
                    dtype=compression_dtype,
                    name=f"{const_node_name}/zero_point",
                )
                converted_zero_point = opset.convert(zero_point_const, const_dtype)
                converted_const = opset.subtract(converted_const, converted_zero_point)

            scale_const = opset.constant(compressed_weight.scale.data, dtype="float16", name=f"{const_node_name}/scale")
            if const_dtype != "float16":
                scale_const = opset.convert(scale_const, const_dtype, name=f"{const_node_name}/scale_convert")
            mul = opset.multiply(
                converted_const,
                scale_const,
                name=f"{const_node_name}/fq_weights_{wc_params.weight_port_id}",
            )

            if compression_config.group_size != -1:
                mul = opset.reshape(mul, output_shape=original_shape, special_zero=False)

            mul_output = mul.output(0)
            for target_input in const_node.output(0).get_target_inputs():
                target_input.replace_source_output(mul_output)

        # reset name_to_node_mapping
        self.name_to_node_mapping = None

        return model

    @staticmethod
    def dump_parameters(
        model: ov.Model, parameters: Dict, algo_name: Optional[str] = "quantization", path: Optional[List] = None
    ) -> None:
        dump_parameters(model, parameters, algo_name, path)


class OVAWQAlgoAlgoBackend(OVWeightCompressionAlgoBackend):
    @staticmethod
    def get_awq_patterns():
        return get_awq_patterns(om.OVMatMulMetatype, om.OVMultiplyMetatype)
