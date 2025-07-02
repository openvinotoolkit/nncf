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

import numpy as np
import onnx
from onnx import helper
from onnx import numpy_helper

import nncf
import nncf.onnx.graph.metatypes.onnx_metatypes as metatypes
import nncf.tensor.functions as fns
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
from nncf.onnx.graph.metatypes.groups import CONVOLUTION_METATYPES
from nncf.onnx.graph.metatypes.groups import MATMUL_METATYPES
from nncf.onnx.graph.model_transformer import remove_initializer
from nncf.onnx.graph.model_transformer import remove_node
from nncf.onnx.graph.model_transformer import set_initializer
from nncf.onnx.graph.node_utils import get_weight_quantization_axis
from nncf.onnx.graph.onnx_helper import ONNX_DTYPE_TO_NNCF_DTYPE
from nncf.onnx.graph.onnx_helper import get_name_to_node_map
from nncf.onnx.graph.onnx_helper import get_node_index
from nncf.onnx.graph.onnx_helper import get_tensor
from nncf.onnx.graph.onnx_helper import get_tensor_value
from nncf.onnx.graph.onnx_helper import pack_4_bits
from nncf.onnx.graph.onnx_helper import pack_int4_to_uint8
from nncf.onnx.graph.transformations.commands import ONNXTargetPoint
from nncf.parameters import CompressionFormat
from nncf.parameters import CompressWeightsMode
from nncf.quantization.advanced_parameters import AdvancedCompressionParameters
from nncf.quantization.algorithms.weight_compression.backend import WeightCompressionAlgoBackend
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionParameters
from nncf.quantization.algorithms.weight_compression.lora_correction import LoraCorrectionAlgorithm
from nncf.quantization.algorithms.weight_compression.weight_lowering import CompressedWeight
from nncf.quantization.algorithms.weight_compression.weight_lowering import compress_weight
from nncf.tensor import Tensor
from nncf.tensor.definitions import TensorDataType


class ONNXWeightCompressionAlgoBackend(WeightCompressionAlgoBackend):
    MODE_TO_WEIGHT_DTYPE = {
        CompressWeightsMode.INT8_SYM: onnx.TensorProto.INT8,
        CompressWeightsMode.INT8_ASYM: onnx.TensorProto.UINT8,
        CompressWeightsMode.INT4_SYM: onnx.TensorProto.INT4,
        CompressWeightsMode.INT4_ASYM: onnx.TensorProto.UINT4,
    }

    def __init__(self, model: onnx.ModelProto):
        super().__init__()
        self.name_to_node_map = get_name_to_node_map(model)

    def _get_weight_dtype(self, mode: CompressWeightsMode) -> onnx.TensorProto.DataType:
        """
        Returns the ONNX data type corresponding to the specified compression mode.
        :param mode: The compression mode.
        :return: The ONNX data type.
        """
        dtype = self.MODE_TO_WEIGHT_DTYPE.get(mode)
        if dtype is None:
            msg = f"{mode.value} is not supported."
            raise nncf.ParameterNotSupportedError(msg)
        return dtype

    def _preprocess_compressed_weight(
        self,
        compressed_weight: CompressedWeight,
        weight_shape: tuple[int],
        dequantize_block_size: Optional[int] = None,
        apply_transpose: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Preprocess compressed weight tensor to ONNX-compatible form.

        :param compressed_weight: Compressed weight struct.
        :param weight_shape: Target shape for the weight tensor.
        :param dequantize_block_size: If given, affects squeezing shape for scale and zero_point.
        :param apply_transpose: Whether to transpose scale and zero_point.
        :return: A tuple containing the reshaped weight tensor, scale tensor, and zero point tensor (if applicable).
        """
        tensor = compressed_weight.tensor.reshape(weight_shape)
        scale = compressed_weight.scale
        zero_point = compressed_weight.zero_point

        axis = 1 if dequantize_block_size else None
        scale = scale.squeeze(axis=axis)
        if zero_point is not None:
            zero_point = zero_point.squeeze(axis=axis)

        if apply_transpose:
            scale = fns.transpose(scale)
            if zero_point is not None:
                zero_point = fns.transpose(zero_point)

        if zero_point is not None:
            zero_point = zero_point.astype(tensor.dtype)

        return tensor.data, scale.data, zero_point.data if zero_point is not None else None

    @property
    def matmul_metatypes(self) -> list[OperatorMetatype]:
        return MATMUL_METATYPES

    @property
    def convolution_metatypes(self) -> list[OperatorMetatype]:
        return CONVOLUTION_METATYPES

    @property
    def embedding_metatypes(self) -> list[OperatorMetatype]:
        return [metatypes.ONNXEmbeddingMetatype]

    @staticmethod
    def is_node_with_weights(node: NNCFNode, graph: NNCFGraph) -> bool:
        return node.layer_attributes.has_weight()

    @staticmethod
    def get_reduction_axes(node_with_weight: NNCFNode, weight_port_id: int, graph: NNCFGraph) -> Optional[tuple[int]]:
        channel_axes = (get_weight_quantization_axis(node_with_weight, weight_port_id),)
        const_shape = node_with_weight.layer_attributes.weight_attrs[weight_port_id]["shape"]
        return get_reduction_axes(channel_axes, const_shape)

    @staticmethod
    def target_point(target_type: TargetType, target_node_name: str, port_id: int) -> ONNXTargetPoint:
        return ONNXTargetPoint(target_type, target_node_name, port_id)

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
    def get_activation_port_id(node: NNCFNode, nncf_graph: NNCFGraph) -> int:
        raise NotImplementedError()

    @staticmethod
    def get_weight_names_and_port_ids(node: NNCFNode, graph: NNCFGraph) -> list[tuple[str, int]]:
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
        return ONNX_DTYPE_TO_NNCF_DTYPE[weight_tensor.data_type]

    @staticmethod
    def get_weight_shape(node_with_weight: NNCFNode, weight_port_id: int, graph: NNCFGraph) -> tuple:
        return node_with_weight.layer_attributes.weight_attrs[weight_port_id]["shape"]

    def set_weight(
        self, node_with_weight: NNCFNode, weight_port_id: int, model: onnx.ModelProto, graph: NNCFGraph, weight: Tensor
    ):
        node = self.name_to_node_map[node_with_weight.target_node_name]
        initializer_name = node.input[weight_port_id]
        set_initializer(initializer_name, model, weight.data)

    @staticmethod
    def _check_arguments_for_transform_model(
        lora_correction_algo: Optional[LoraCorrectionAlgorithm],
        compression_format: CompressionFormat,
        advanced_parameters: AdvancedCompressionParameters,
    ):
        if lora_correction_algo is not None:
            msg = "LORA correction is not supported for the ONNX backend"
            raise nncf.ValidationError(msg)
        if compression_format != CompressionFormat.DQ:
            msg = "Compression format is not supported for the ONNX backend"
            raise nncf.ValidationError(msg)

    def transform_model(
        self,
        model: onnx.ModelProto,
        graph: NNCFGraph,
        weight_compression_parameters: Iterable[WeightCompressionParameters],
        precomputed_scales: dict[str, Tensor] = None,
        precomputed_zero_points: dict[str, Tensor] = None,
        lora_correction_algo: Optional[LoraCorrectionAlgorithm] = None,
        compression_format: CompressionFormat = CompressionFormat.DQ,
        advanced_parameters: AdvancedCompressionParameters = AdvancedCompressionParameters(),
    ) -> onnx.ModelProto:
        self._check_arguments_for_transform_model(lora_correction_algo, compression_format, advanced_parameters)
        opset_version = model.opset_import[0].version

        for wc_params in weight_compression_parameters:
            compression_config = wc_params.compression_config
            node = wc_params.node_with_weight
            weight = self.get_weight(node, wc_params.weight_port_id, model, graph)
            compressed_weight = compress_weight(
                Tensor(weight),
                wc_params.reduction_axes,
                compression_config,
                None if precomputed_scales is None else precomputed_scales.get(wc_params.weight_name),
                None if precomputed_zero_points is None else precomputed_zero_points.get(wc_params.weight_name),
            )
            dequantize_block_size = max(compression_config.group_size, 0)  # 0 - is no block wise quantization
            dequantize_axis = (
                get_weight_quantization_axis(node, wc_params.weight_port_id) if dequantize_block_size <= 0 else 0
            )  # axis = 0 when blockwise

            # NOTE: The `DequantizeLinear` operation supports the `block_size` attribute only starting from opset 21.
            # For opsets earlier than 21, we use the `MatMulNBits` operation from ONNX Runtime contrib operators.
            # See https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md
            if opset_version < 21 and dequantize_block_size > 0:
                compressed_weight, scale, zero_point = self._preprocess_compressed_weight(
                    compressed_weight, weight.shape, dequantize_block_size=None, apply_transpose=True
                )
                self._replace_matmul_with_matmulnbits(
                    model,
                    wc_params,
                    compressed_weight,
                    weight,
                    scale,
                    zero_point,
                    dequantize_axis,
                    dequantize_block_size,
                    wc_params.weight_name,
                    self._get_weight_dtype(compression_config.mode),
                )
            else:
                compressed_weight, scale, zero_point = self._preprocess_compressed_weight(
                    compressed_weight, weight.shape, dequantize_block_size=dequantize_block_size
                )
                self._add_dequantize_linear_layer(
                    model,
                    compressed_weight,
                    scale,
                    zero_point,
                    dequantize_axis,
                    dequantize_block_size,
                    wc_params.weight_name,
                    self._get_weight_dtype(compression_config.mode),
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

    def _add_dequantize_linear_layer(
        self,
        model: onnx.ModelProto,
        quantized_weights: np.ndarray,
        scale: np.ndarray,
        zero_point: Optional[np.ndarray],
        axis: int,
        block_size: int,
        weight_name: str,
        weight_dtype: onnx.TensorProto.DataType,
        scale_dtype: onnx.TensorProto.DataType = onnx.TensorProto.FLOAT,
    ) -> None:
        """
        Add a DequantizeLinear node to the ONNX model to dequantize the weights.
        :param model: The ONNX model.
        :param quantized_weights: The quantized weights to be dequantized.
        :param scale: The scale tensor for dequantization.
        :param zero_point: The zero point tensor for dequantization (optional).
        :param axis: The axis along which to dequantize.
        :param block_size: The block size for dequantization.
        :param weight_name: The name of the weight tensor.
        :param weight_dtype: The data type of the weight tensor.
        :param scale_dtype: The data type of the scale tensor (default is FLOAT).
        :return: The modified ONNX model with the DequantizeLinear node added.
        """
        orig_shape = quantized_weights.shape
        if zero_point is not None:
            orig_zero_point_shape = zero_point.shape

        quantized_weight_name = weight_name + "_quantized"
        scale_name = weight_name + "_scale"
        dequantized_weight_output = weight_name + "_dequantized"
        deq_inputs = [quantized_weight_name, scale_name]

        if weight_dtype in [onnx.TensorProto.INT4, onnx.TensorProto.UINT4]:
            quantized_weights = pack_4_bits(quantized_weights)
            if zero_point is not None:
                zero_point = pack_4_bits(zero_point)

        # Create initializers for the quantized weights, scale, and zero point
        quantized_weights_initializer = onnx.helper.make_tensor(
            quantized_weight_name, weight_dtype, orig_shape, quantized_weights.tobytes(), raw=True
        )
        scale_initializer = numpy_helper.from_array(
            np.array(scale, dtype=helper.tensor_dtype_to_np_dtype(scale_dtype)), name=scale_name
        )

        new_initializers = [quantized_weights_initializer, scale_initializer]

        if zero_point is not None:
            deq_inputs.append(weight_name + "_zero_point")
            zero_point_initializer = onnx.helper.make_tensor(
                weight_name + "_zero_point", weight_dtype, orig_zero_point_shape, zero_point.tobytes(), raw=True
            )
            new_initializers.append(zero_point_initializer)

        if block_size != 0:
            dequantize_node = helper.make_node(
                "DequantizeLinear",
                inputs=deq_inputs,
                outputs=[dequantized_weight_output],
                block_size=block_size,
                axis=axis,
            )
        else:
            dequantize_node = helper.make_node(
                "DequantizeLinear",
                inputs=deq_inputs,
                outputs=[dequantized_weight_output],
                axis=axis,
            )

        # Add the node and initializers to the model
        model.graph.initializer.extend(new_initializers)

        # Insert the DequantizeLinear node before the consumer nodes
        insert_index = len(model.graph.node)

        for i, node in enumerate(model.graph.node):
            for j, input_name in enumerate(node.input):
                if input_name == weight_name:
                    insert_index = min(insert_index, get_node_index(model, node.name))
                    node.input[j] = dequantized_weight_output

        # Insert the DequantizeLinear node before the first consumer node
        model.graph.node.insert(insert_index, dequantize_node)
        # Remove original weight initializer
        remove_initializer(weight_name, model)
        # Update the node mapping
        self.name_to_node_map[dequantize_node.name] = dequantize_node
        return model

    def _replace_matmul_with_matmulnbits(
        self,
        model: onnx.ModelProto,
        weight_compression_parameters: WeightCompressionParameters,
        quantized_weights: np.ndarray,
        orig_weight,
        scale: np.ndarray,
        zero_point: Optional[np.ndarray],
        axis: int,
        block_size: int,
        weight_name: str,
        weight_dtype: onnx.TensorProto.DataType,
        scale_dtype: onnx.TensorProto.DataType = onnx.TensorProto.FLOAT,
    ):
        if weight_dtype == onnx.TensorProto.INT4:
            quantized_weights = pack_int4_to_uint8(quantized_weights, block_size, signed=True)
        elif weight_dtype == onnx.TensorProto.UINT4:
            quantized_weights = pack_int4_to_uint8(quantized_weights, block_size, signed=False)

        quantized_weight_name = weight_name + "_quantized"
        scale_name = weight_name + "_scale"
        zero_point_name = weight_name + "_zero_point"

        quantized_weights_initializer = onnx.helper.make_tensor(
            quantized_weight_name,
            onnx.TensorProto.UINT8,
            quantized_weights.shape,
            quantized_weights.tobytes(),
            raw=True,
        )

        # Create initializers for the quantized weights, scale, and zero point
        scale_initializer = numpy_helper.from_array(
            np.array(scale, dtype=helper.tensor_dtype_to_np_dtype(scale_dtype)), name=scale_name
        )
        new_initializers = [quantized_weights_initializer, scale_initializer]
        if zero_point is not None:
            zero_point_initializer = onnx.helper.make_tensor(
                zero_point_name, onnx.TensorProto.UINT8, zero_point.shape, zero_point.tobytes(), raw=True
            )
            new_initializers.append(zero_point_initializer)

        original_matmul = self.name_to_node_map[weight_compression_parameters.node_with_weight.node_name]

        activation_input_name = None
        for input_name in original_matmul.input:
            if input_name != weight_name:
                activation_input_name = input_name
        assert activation_input_name is not None, "Activation input name not found in original matmul node"

        # Create MatMulNBits
        inputs = [activation_input_name, quantized_weight_name, scale_name]
        if zero_point is not None:
            inputs.append(zero_point_name)

        K, N = orig_weight.shape[0], orig_weight.shape[1]
        matmul_n_bits = helper.make_node(
            op_type="MatMulNBits",
            inputs=inputs,
            outputs=[original_matmul.output[0]],
            K=K,
            N=N,
            accuracy_level=0,
            bits=weight_compression_parameters.compression_config.num_bits,
            block_size=weight_compression_parameters.compression_config.group_size,
            domain="com.microsoft",
            name=original_matmul.name + "_compressed_matmul",
        )

        # Add the node and initializers to the model
        model.graph.initializer.extend(new_initializers)

        # Insert the MatMulNBits node before the consumer nodes
        insert_index = len(model.graph.node)

        for i, node in enumerate(model.graph.node):
            for j, input_name in enumerate(node.input):
                if input_name == original_matmul.name:
                    insert_index = min(insert_index, get_node_index(model, node.name))
                    node.input[j] = matmul_n_bits

        # Insert the MatMulNBits node before the first consumer node
        model.graph.node.insert(insert_index, matmul_n_bits)
        # Remove original weight initializer
        remove_initializer(weight_name, model)
        # Remove original matmul
        remove_node(original_matmul.name, model)
        del self.name_to_node_map[original_matmul.name]
        # Update the node mapping
        self.name_to_node_map[matmul_n_bits.name] = matmul_n_bits
