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
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import onnx
from onnx import helper
from onnx import numpy_helper

import nncf.onnx.graph.metatypes.onnx_metatypes as metatypes
from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.utils import get_reduction_axes
from nncf.common.tensor_statistics.statistic_point import StatisticPoint
from nncf.experimental.common.tensor_statistics.collectors import MeanReducer
from nncf.experimental.common.tensor_statistics.collectors import NoopAggregator
from nncf.experimental.common.tensor_statistics.collectors import ShapeReducer
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.experimental.common.tensor_statistics.statistics import WCTensorStatistic
from nncf.onnx.graph.metatypes.groups import MATMUL_METATYPES
from nncf.onnx.graph.node_utils import get_weight_quantization_axis
from nncf.onnx.graph.onnx_helper import get_name_to_node_map
from nncf.onnx.graph.onnx_helper import get_tensor
from nncf.onnx.graph.onnx_helper import get_tensor_value
from nncf.onnx.graph.transformations.commands import ONNXTargetPoint
from nncf.parameters import CompressWeightsMode
from nncf.quantization.algorithms.weight_compression.backend import WeightCompressionAlgoBackend
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionParameters
from nncf.quantization.algorithms.weight_compression.lora_correction import LoraCorrectionAlgorithm
from nncf.quantization.algorithms.weight_compression.weight_lowering import compress_weight
from nncf.tensor import Tensor
from nncf.tensor.definitions import TensorDataType

DTYPE_MAP = {
    TensorDataType.float16: onnx.TensorProto.FLOAT16,
    TensorDataType.bfloat16: onnx.TensorProto.BFLOAT16,
    TensorDataType.float32: onnx.TensorProto.FLOAT,
    TensorDataType.float64: onnx.TensorProto.DOUBLE,
    TensorDataType.int8: onnx.TensorProto.INT8,
    TensorDataType.int32: onnx.TensorProto.INT32,
    TensorDataType.int64: onnx.TensorProto.INT64,
    TensorDataType.uint8: onnx.TensorProto.UINT8,
    TensorDataType.uint4: onnx.TensorProto.UINT4,
    TensorDataType.int4: onnx.TensorProto.INT4,
}

DTYPE_MAP_REV = {v: k for k, v in DTYPE_MAP.items()}


class ONNXWeightCompressionAlgoBackend(WeightCompressionAlgoBackend):
    @property
    def matmul_metatypes(self) -> List[OperatorMetatype]:
        return MATMUL_METATYPES

    @property
    def convolution_metatypes(self) -> List[OperatorMetatype]:
        return [
            metatypes.ONNXConvolutionMetatype,
            metatypes.ONNXDepthwiseConvolutionMetatype,
            metatypes.ONNXConvolutionTransposeMetatype,
            metatypes.ONNXDeformableConvolutionMetatype,
        ]

    @property
    def embedding_metatypes(self) -> List[OperatorMetatype]:
        return [metatypes.ONNXEmbeddingMetatype]

    @staticmethod
    def is_node_with_weights(node: NNCFNode, graph: NNCFGraph) -> bool:
        return node.layer_attributes.has_weight()

    @staticmethod
    def get_reduction_axes(node_with_weight: NNCFNode, weight_port_id: int, graph: NNCFGraph) -> Optional[Tuple[int]]:
        channel_axes = (get_weight_quantization_axis(node_with_weight, weight_port_id),)
        const_shape = node_with_weight.layer_attributes.weight_attrs[weight_port_id]["shape"]
        return get_reduction_axes(channel_axes, const_shape)

    @staticmethod
    def target_point(target_type: TargetType, target_node_name: str, port_id: int) -> ONNXTargetPoint:
        return ONNXTargetPoint(target_type, target_node_name, port_id)

    def mean_statistic_collector(
        self, reduction_axes: Tuple[int], subset_size: Optional[int] = None
    ) -> TensorCollector:
        mean_reducer = MeanReducer(reduction_axes)
        shape_reducer = ShapeReducer()
        collector = TensorCollector(WCTensorStatistic)
        collector.register_statistic_branch(WCTensorStatistic.MEAN_STAT, mean_reducer, NoopAggregator(subset_size))
        collector.register_statistic_branch(WCTensorStatistic.SHAPE_STAT, shape_reducer, NoopAggregator(subset_size))
        return collector

    @staticmethod
    def get_activation_port_id(node: NNCFNode, nncf_graph: NNCFGraph) -> int:
        raise NotImplementedError()

    @staticmethod
    def get_weight_names_and_port_ids(node: NNCFNode, graph: NNCFGraph) -> List[Tuple[str, int]]:
        return [(attr["name"], port_id) for port_id, attr in node.layer_attributes.weight_attrs.items()]

    def get_weight(
        self, node_with_weight: NNCFNode, weight_port_id: int, model: onnx.ModelProto, graph: NNCFGraph
    ) -> Tensor:
        weight_name = node_with_weight.layer_attributes.weight_attrs[weight_port_id]["name"]
        weight_tensor = get_tensor_value(model, weight_name)
        return Tensor(weight_tensor)

    def get_weight_dtype(
        self, node_with_weight: NNCFNode, weight_port_id: int, model: onnx.ModelProto, graph: NNCFGraph
    ) -> TensorDataType:
        weight_name = node_with_weight.layer_attributes.weight_attrs[weight_port_id]["name"]
        weight_tensor = get_tensor(model, weight_name)
        return DTYPE_MAP_REV[weight_tensor.data_type]

    @staticmethod
    def get_weight_shape(node_with_weight: NNCFNode, weight_port_id: int, graph: NNCFGraph) -> Tuple:
        return node_with_weight.layer_attributes.weight_attrs[weight_port_id]["shape"]

    def set_weight(
        self, node_with_weight: NNCFNode, weight_port_id: int, model: onnx.ModelProto, graph: NNCFGraph, weight: Tensor
    ):
        name_to_node_map = get_name_to_node_map(model)
        node = name_to_node_map[node_with_weight.target_node_name]
        initializer_name = node.input[weight_port_id]
        initializer = get_tensor(model, initializer_name)

        new_tensor = onnx.numpy_helper.from_array(weight.data, initializer_name)
        initializer.CopyFrom(new_tensor)

    def transform_model(
        self,
        model: onnx.ModelProto,
        graph: NNCFGraph,
        weight_compression_parameters: Iterable[WeightCompressionParameters],
        precomputed_scales: Dict[str, Tensor] = None,
        precomputed_zero_points: Dict[str, Tensor] = None,
        lora_correction_algo: LoraCorrectionAlgorithm = None,
    ) -> onnx.ModelProto:
        for wc_params in weight_compression_parameters:
            compression_config = wc_params.compression_config
            node = wc_params.node_with_weight
            weight = self.get_weight(node, wc_params.weight_port_id, model, graph)
            # calculates compressed weights and decompression parameters
            compressed_weight = compress_weight(
                Tensor(weight),
                wc_params.reduction_axes,
                compression_config,
                None if precomputed_scales is None else precomputed_scales.get(wc_params.weight_name),
                None if precomputed_zero_points is None else precomputed_zero_points.get(wc_params.weight_name),
            )
            if compression_config.mode == CompressWeightsMode.INT8_SYM:
                dtype = onnx.TensorProto.INT8
            elif compression_config.mode == CompressWeightsMode.INT8_ASYM:
                dtype = onnx.TensorProto.UINT8
            elif compression_config.mode == CompressWeightsMode.INT4_SYM:
                dtype = onnx.TensorProto.INT4
            elif compression_config.mode == CompressWeightsMode.INT4_ASYM:
                dtype = onnx.TensorProto.UINT4
            block_size = compression_config.group_size
            channel_axes = get_weight_quantization_axis(node, wc_params.weight_port_id)
            c_w = compressed_weight.tensor.reshape(weight.shape)
            scale = compressed_weight.scale.squeeze()
            if compressed_weight.zero_point is not None:
                z_p = compressed_weight.zero_point.squeeze()
            add_dequantize_linear_layer(
                model,
                c_w,
                scale,
                z_p if compressed_weight.zero_point is not None else None,
                wc_params.weight_name,
                channel_axes,
                dtype,
                block_size,
            )
        return model

    @staticmethod
    def get_filter_fn_for_statistics(activation_port_id: int, algorithm_key: str) -> Callable[[StatisticPoint], bool]:
        def filter_func(point: StatisticPoint) -> bool:
            return (
                algorithm_key in point.algorithm_to_tensor_collectors
                and point.target_point.type == TargetType.POST_LAYER_OPERATION
                and point.target_point.port_id == activation_port_id
            )

        return filter_func

    def insert_adapters(
        self, wc_params: WeightCompressionParameters, lora_A: Tensor, lora_B: Tensor, int8_lora: bool
    ) -> None:
        raise NotImplementedError()


def add_dequantize_linear_layer(
    model, quantized_weights, scale, zero_point, weight_name, channel_axes, dtype, block_size
):
    quantized_weights = quantized_weights.data
    scale = scale.data

    if zero_point is not None:
        zero_point = zero_point.data
    # Create a DequantizeLinear node
    axis = 0
    if block_size == -1:
        block_size = None
        print(channel_axes)
        axis = channel_axes
        scale = scale.squeeze()
        zero_point = zero_point.squeeze()
    deq_inputs = [weight_name + "_quantized", weight_name + "_scale"]
    if zero_point is not None:
        deq_inputs.append(weight_name + "_zero_point")
    dequantize_node = helper.make_node(
        "DequantizeLinear",
        inputs=deq_inputs,
        outputs=[weight_name + "_dequantized"],
        block_size=block_size,
        axis=axis,
    )

    # Create initializers for the quantized weights, scale, and zero point
    quantized_weights_initializer = onnx.helper.make_tensor(
        weight_name + "_quantized",
        dtype,
        quantized_weights.shape,
        quantized_weights,
    )
    scale_initializer = numpy_helper.from_array(np.array(scale, dtype=np.float32), name=weight_name + "_scale")
    initials = [quantized_weights_initializer, scale_initializer]
    if zero_point is not None:
        zero_point_initializer = onnx.helper.make_tensor(
            weight_name + "_zero_point",
            dtype,
            scale.shape,
            zero_point,
        )
        initials.append(zero_point_initializer)
    # Add the node and initializers to the model
    model.graph.node.append(dequantize_node)
    model.graph.initializer.extend(initials)

    # Update the consumer nodes to use the dequantized weights
    for node in model.graph.node:
        for i, input_name in enumerate(node.input):
            if input_name == weight_name:
                node.input[i] = weight_name + "_dequantized"

    # Find the index of the original weight initializer and remove it
    initializer_index = next((i for i, init in enumerate(model.graph.initializer) if init.name == weight_name), None)
    if initializer_index is not None:
        del model.graph.initializer[initializer_index]

    return model
