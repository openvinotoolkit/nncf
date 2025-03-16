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

import onnx

import nncf
from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype
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

from nncf.onnx.graph.metatypes import onnx_metatypes as om
from nncf.onnx.graph.metatypes.groups import ATOMIC_ACTIVATIONS_OPERATIONS
from nncf.onnx.graph.model_transformer import ONNXModelTransformer
from nncf.onnx.graph.node_utils import convert_op
from nncf.onnx.graph.node_utils import create_ov_const_from_tensor
from nncf.onnx.graph.node_utils import get_const_value_as_numpy_tensor
from nncf.onnx.graph.node_utils import get_const_value_as_onnx_tensor
from nncf.onnx.graph.node_utils import get_weight_channel_axes
from nncf.onnx.graph.transformations.command_creation import OVCommandCreator
from nncf.onnx.graph.transformations.commands import OVTargetPoint
from nncf.onnx.optimized_functions import clear_onnx_model_cache
from nncf.onnx.optimized_functions.models import ONNX_MODEL_CACHE
from nncf.onnx.rt_info import dump_parameters
from nncf.onnx.statistics.collectors import OVMaxVarianceReducer
from nncf.onnx.statistics.collectors import OVMeanAbsMaxReducer
from nncf.onnx.statistics.collectors import OVMeanReducer
from nncf.onnx.statistics.collectors import OVMeanVarianceReducer
from nncf.onnx.statistics.collectors import OVShapeReducer


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
from nncf.quantization.algorithms.weight_compression.weight_lowering import compress_weight
from nncf.tensor import Tensor
from nncf.tensor.definitions import TensorDataType
from nncf.tensor.functions.onnx_numeric import DTYPE_MAP_REV


class ONNXWeightCompressionAlgoBackend(WeightCompressionAlgoBackend):

    @property
    def matmul_metatypes(self) -> List[OperatorMetatype]:
        return [om.ONNXMatMulMetatype]

    @property
    def convolution_metatypes(self) -> List[OperatorMetatype]:
        # TODO: Add more convolution metatypes
        return [
            om.ONNXConvolutionMetatype,
            om.ONNXDepthwiseConvolutionMetatype,
            om.ONNXGroupConvolutionMetatype,
        ]

    @property
    def embedding_metatypes(self) -> List[OperatorMetatype]:
        return [om.ONNXEmbeddingMetatype]

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

    def mean_statistic_collector(
        self, reduction_axes: Tuple[int], subset_size: Optional[int] = None
    ) -> TensorCollector:
        mean_reducer = ONNXMeanReducer(reduction_axes, inplace=True)
        shape_reducer = ONNXShapeReducer(inplace=True)
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
    def get_weight_names_and_port_ids(node: NNCFNode, graph: NNCFGraph) -> List[Tuple[str, int]]:
        result = []
        for weight_port_id in node.layer_attributes.get_const_port_ids():
            weight_name = node.layer_attributes.constant_attributes[weight_port_id]["name"]
            result.append((weight_name, weight_port_id))
        return result

    def get_weight(self, node_with_weight: NNCFNode, weight_port_id: int, model: onnx.ModelProtoo, graph: NNCFGraph) -> Tensor:
        weight_name = node_with_weight.layer_attributes.constant_attributes[weight_port_id]["name"]
        weight_node = self.name_to_node_mapping[weight_name]
        weight_tensor = get_const_value_as_numpy_tensor(weight_node)
        return Tensor(weight_tensor)

    def get_weight_dtype(
        self, node_with_weight: NNCFNode, weight_port_id: int, model: onnx.ModelProto, graph: NNCFGraph
    ) -> TensorDataType:
        onnx_type_name = node_with_weight.layer_attributes.constant_attributes[weight_port_id]["dtype"]
        onnx_type = getattr(onnx.TensorProto.DataType, onnx_type_name)
        return DTYPE_MAP_REV[onnx_type]

    @staticmethod
    def get_weight_shape(node_with_weight: NNCFNode, weight_port_id: int, graph: NNCFGraph) -> Tuple:
        return node_with_weight.layer_attributes.constant_attributes[weight_port_id]["shape"]

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
        if compression_config.mode == CompressWeightsMode.INT8_SYM:
            compression_dtype = onnx.TensorProto.INT8
            is_symmetric = True
        elif compression_config.mode == CompressWeightsMode.INT8_ASYM:
            compression_dtype = onnx.TensorProto.UINT8
            is_symmetric = False
        elif compression_config.mode in [CompressWeightsMode.INT4_SYM, CompressWeightsMode.INT4_ASYM, 
                                        CompressWeightsMode.NF4, CompressWeightsMode.E2M1]:
            msg = f"{compression_config.mode.value} is not directly supported in ONNX backend yet."
            raise nncf.ParameterNotSupportedError(msg)
        else:
            msg = f"{compression_config.mode.value} is not supported."
            raise nncf.ParameterNotSupportedError(msg)

        original_shape = weight.shape
        
        compressed_weight = compress_weight(
            weight,
            reduction_axes,
            compression_config,
            layer_scales,
            layer_zero_points,
        )
        
        prefix = f"{const_node_name}_{weight_port_id}"
        
        compressed_tensor_name = f"{prefix}_compressed"
        if is_symmetric:
            compressed_data = compressed_weight.tensor.data.astype(np.int8)
        else:
            compressed_data = compressed_weight.tensor.data.astype(np.uint8)
        
        compressed_initializer = numpy_helper.from_array(compressed_data, compressed_tensor_name)
        self.new_initializers.append(compressed_initializer)
        
        # 创建缩放因子的初始化器
        scale_tensor_name = f"{prefix}_scale"
        scale_data = compressed_weight.scale.data
        scale_initializer = numpy_helper.from_array(scale_data.astype(np.float32), scale_tensor_name)
        self.new_initializers.append(scale_initializer)
        
        # 执行解量化操作
        # 步骤1: Cast操作 - 将压缩的整数转换为浮点数
        cast_output_name = f"{prefix}_casted"
        cast_node = onnx.helper.make_node(
            'Cast',
            inputs=[compressed_tensor_name],
            outputs=[cast_output_name],
            name=f"{prefix}_cast",
            to=onnx.TensorProto.FLOAT  # 转换为浮点数
        )
        self.new_nodes.append(cast_node)
        
        # 对于非对称量化，处理零点
        if not is_symmetric and compressed_weight.zero_point is not None:
            # 创建零点的初始化器
            zero_point_tensor_name = f"{prefix}_zero_point"
            zero_point_data = compressed_weight.zero_point.data
            if is_symmetric:
                zero_point_data = zero_point_data.astype(np.int8)
            else:
                zero_point_data = zero_point_data.astype(np.uint8)
            
            zero_point_initializer = numpy_helper.from_array(zero_point_data, zero_point_tensor_name)
            self.new_initializers.append(zero_point_initializer)
            
            # 创建零点的Cast节点
            zp_cast_output_name = f"{prefix}_zp_casted"
            zp_cast_node = onnx.helper.make_node(
                'Cast',
                inputs=[zero_point_tensor_name],
                outputs=[zp_cast_output_name],
                name=f"{prefix}_zp_cast",
                to=onnx.TensorProto.FLOAT
            )
            self.new_nodes.append(zp_cast_node)
            
            # 减去零点
            sub_output_name = f"{prefix}_sub_zp"
            sub_node = onnx.helper.make_node(
                'Sub',
                inputs=[cast_output_name, zp_cast_output_name],
                outputs=[sub_output_name],
                name=f"{prefix}_sub"
            )
            self.new_nodes.append(sub_node)
            
            # 更新当前输出名称，用于下一步乘以缩放因子
            current_output_name = sub_output_name
        else:
            # 如果是对称量化，不需要减去零点
            current_output_name = cast_output_name
        
        # 乘以缩放因子
        mul_output_name = f"{prefix}_dequantized"
        mul_node = onnx.helper.make_node(
            'Mul',
            inputs=[current_output_name, scale_tensor_name],
            outputs=[mul_output_name],
            name=f"{prefix}_mul"
        )
        self.new_nodes.append(mul_node)
        
        # 处理分组量化的重塑操作
        if compression_config.group_size != -1:
            reshape_output_name = f"{prefix}_reshaped"
            # 创建原始形状的初始化器
            shape_tensor_name = f"{prefix}_shape"
            shape_data = np.array(original_shape, dtype=np.int64)
            shape_initializer = numpy_helper.from_array(shape_data, shape_tensor_name)
            self.new_initializers.append(shape_initializer)
            
            # 创建Reshape节点
            reshape_node = onnx.helper.make_node(
                'Reshape',
                inputs=[mul_output_name, shape_tensor_name],
                outputs=[reshape_output_name],
                name=f"{prefix}_reshape"
            )
            self.new_nodes.append(reshape_node)
            current_output_name = reshape_output_name
        else:
            current_output_name = mul_output_name
        
        # 如果需要，添加额外的类型转换节点
        if should_add_convert_node:
            final_output_name = f"{prefix}_final"
            convert_node = onnx.helper.make_node(
                'Cast',
                inputs=[current_output_name],
                outputs=[final_output_name],
                name=f"{prefix}_final_cast",
                to=self._convert_dtype_to_onnx(const_dtype)  # 转换为原始常量的数据类型
            )
            self.new_nodes.append(convert_node)
            final_node = convert_node
        else:
            final_output_name = current_output_name
            # 找到最后创建的节点
            final_node = self.new_nodes[-1]
        
        return final_node, compressed_weight
    
    def _convert_dtype_to_onnx(self, dtype):
        """将内部数据类型转换为ONNX的数据类型"""
        dtype_mapping = {
            "float32": onnx.TensorProto.FLOAT,
            "float16": onnx.TensorProto.FLOAT16,
            "int8": onnx.TensorProto.INT8,
            "uint8": onnx.TensorProto.UINT8,
        }
        
        if isinstance(dtype, str):
            if dtype in dtype_mapping:
                return dtype_mapping[dtype]
        
        if isinstance(dtype, int) and dtype in [item for item in dtype_mapping.values()]:
            return dtype
        
        return onnx.TensorProto.FLOAT

    def transform_model(
        self,
        model: onnx.ModelProto,
        graph: NNCFGraph,
        weight_compression_parameters: Iterable[WeightCompressionParameters],
        precomputed_scales: Dict[str, Tensor] = None,
        precomputed_zero_points: Dict[str, Tensor] = None,
        lora_correction_algo: LoraCorrectionAlgorithm = None,
        compression_format: CompressionFormat = CompressionFormat.DQ,
        advanced_parameters: AdvancedCompressionParameters = AdvancedCompressionParameters(),
    ) -> onnx.ModelProto:
        # TODO 完成 ONNXWeightCompressionAlgoBackend 类中的函数， 只考虑最简单的 weightcompression
        compressed_model = onnx.ModelProto()
        compressed_model.CopyFrom(model)
        onnx_graph = compressed_model.graph
        
        # Create mappings for easier lookup
        initializers = {init.name: (i, init) for i, init in enumerate(onnx_graph.initializer)}
        nodes = {node.name: node for node in onnx_graph.node}
        
        # Track newly added nodes and initializers
        new_nodes = []
        new_initializers = []
        
        for wc_params in weight_compression_parameters:
            # Get weight node information
            weight_node = get_const_node(wc_params.node_with_weight, wc_params.weight_port_id, graph)
            weight_name = weight_node.layer_attributes.name
            
            # Check if weight exists in initializers
            if weight_name not in initializers:
                msg = f"Could not find weight tensor '{weight_name}' in ONNX model initializers."
                raise nncf.InternalError(msg)
            
            # Get weight data from ONNX model
            _, weight_initializer = initializers[weight_name]
            weight_np = numpy_helper.to_array(weight_initializer)
            weight = Tensor(weight_np)
            
            # Check if compression mode is supported
            compression_config = wc_params.compression_config
            if compression_config.mode in [
                CompressWeightsMode.NF4,
                CompressWeightsMode.E2M1,
            ]:
                msg = f"{compression_config.mode.value} is not supported for ONNX backend."
                raise nncf.ParameterNotSupportedError(msg)
            
            # Find nodes that use this weight
            consumer_nodes = []
            for node in onnx_graph.node:
                if weight_name in node.input:
                    consumer_nodes.append(node)
            
            if not consumer_nodes:
                continue  # Skip if no nodes use this weight
            
            # Compress weight
            compressed_weight = compress_weight(
                weight,
                wc_params.reduction_axes,
                compression_config,
                None if precomputed_scales is None else precomputed_scales.get(wc_params.weight_name),
                None if precomputed_zero_points is None else precomputed_zero_points.get(wc_params.weight_name),
            )
            
            # Create decompression subgraph based on compression mode
            if compression_config.mode == CompressWeightsMode.INT8_SYM:
                # Create compressed weight initializer (int8)
                compressed_data = compressed_weight.tensor.data.numpy().astype(np.int8)
                compressed_weight_name = f"{weight_name}_compressed"
                compressed_initializer = numpy_helper.from_array(compressed_data, compressed_weight_name)
                new_initializers.append(compressed_initializer)
                
                # Create scale initializer
                scale_data = compressed_weight.scale.data.numpy()
                scale_name = f"{weight_name}_scale"
                scale_initializer = numpy_helper.from_array(scale_data, scale_name)
                new_initializers.append(scale_initializer)
                
                # Create cast node (int8 -> float)
                cast_output_name = f"{weight_name}_casted"
                cast_node = onnx.helper.make_node(
                    'Cast',
                    inputs=[compressed_weight_name],
                    outputs=[cast_output_name],
                    name=f"{weight_name}_cast",
                    to=onnx.TensorProto.FLOAT
                )
                new_nodes.append(cast_node)
                
                # Create multiplication node (apply scale)
                dequantized_name = f"{weight_name}_dequantized"
                mul_node = onnx.helper.make_node(
                    'Mul',
                    inputs=[cast_output_name, scale_name],
                    outputs=[dequantized_name],
                    name=f"{weight_name}_dequant"
                )
                new_nodes.append(mul_node)
                
                # Replace weight usage in consumer nodes
                for node in consumer_nodes:
                    for i, input_name in enumerate(node.input):
                        if input_name == weight_name:
                            node.input[i] = dequantized_name
                
            elif compression_config.mode == CompressWeightsMode.INT8_ASYM:
                # Create compressed weight initializer (uint8)
                compressed_data = compressed_weight.tensor.data.numpy().astype(np.uint8)
                compressed_weight_name = f"{weight_name}_compressed"
                compressed_initializer = numpy_helper.from_array(compressed_data, compressed_weight_name)
                new_initializers.append(compressed_initializer)
                
                # Create scale initializer
                scale_data = compressed_weight.scale.data.numpy()
                scale_name = f"{weight_name}_scale"
                scale_initializer = numpy_helper.from_array(scale_data, scale_name)
                new_initializers.append(scale_initializer)
                
                # Create zero point initializer
                zero_point_data = compressed_weight.zero_point.data.numpy()
                zero_point_name = f"{weight_name}_zero_point"
                zero_point_initializer = numpy_helper.from_array(zero_point_data, zero_point_name)
                new_initializers.append(zero_point_initializer)
                
                # Create cast node (uint8 -> float)
                cast_output_name = f"{weight_name}_casted"
                cast_node = onnx.helper.make_node(
                    'Cast',
                    inputs=[compressed_weight_name],
                    outputs=[cast_output_name],
                    name=f"{weight_name}_cast",
                    to=onnx.TensorProto.FLOAT
                )
                new_nodes.append(cast_node)
                
                # Create subtraction node (subtract zero point)
                sub_output_name = f"{weight_name}_sub_zp"
                sub_node = onnx.helper.make_node(
                    'Sub',
                    inputs=[cast_output_name, zero_point_name],
                    outputs=[sub_output_name],
                    name=f"{weight_name}_sub_zp"
                )
                new_nodes.append(sub_node)
                
                # Create multiplication node (apply scale)
                dequantized_name = f"{weight_name}_dequantized"
                mul_node = onnx.helper.make_node(
                    'Mul',
                    inputs=[sub_output_name, scale_name],
                    outputs=[dequantized_name],
                    name=f"{weight_name}_dequant"
                )
                new_nodes.append(mul_node)
                
                # Replace weight usage in consumer nodes
                for node in consumer_nodes:
                    for i, input_name in enumerate(node.input):
                        if input_name == weight_name:
                            node.input[i] = dequantized_name
                            
            elif compression_config.mode in [CompressWeightsMode.INT4_SYM, CompressWeightsMode.INT4_ASYM]:
                # For INT4 formats, since ONNX doesn't natively support int4,
                # we need to pack int4 values into int8 or uint8 tensors
                # This requires careful handling of the unpacking logic
                
                # Implementation would be similar to INT8 cases but with additional handling
                # for the packed format and unpacking operations
                
                # For this implementation, let's assume INT4 is not fully supported yet
                msg = f"{compression_config.mode.value} is not fully implemented for ONNX backend yet."
                raise nncf.ParameterNotSupportedError(msg)
        
        # Add new initializers and nodes to the graph
        for initializer in new_initializers:
            onnx_graph.initializer.append(initializer)
        
        for node in new_nodes:
            onnx_graph.node.append(node)
        
        # Remove original weight initializers that have been replaced
        # This is optional but helps reduce model size
        original_initializers_to_keep = []
        for i, initializer in enumerate(onnx_graph.initializer):
            if any(initializer.name in node.input for node in onnx_graph.node):
                original_initializers_to_keep.append(initializer)
        
        # Clear and re-add initializers
        del onnx_graph.initializer[:]
        for initializer in original_initializers_to_keep + new_initializers:
            if initializer not in onnx_graph.initializer:
                onnx_graph.initializer.append(initializer)
        
        # Verify the model is valid
        try:
            onnx.checker.check_model(compressed_model)
        except Exception as e:
            raise nncf.InternalError(f"Generated ONNX model is invalid: {str(e)}")
        
        return compressed_model

    @staticmethod
    def dump_parameters(
        model: onnx.ModelProto, parameters: Dict, algo_name: Optional[str] = "quantization", path: Optional[List] = None
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

