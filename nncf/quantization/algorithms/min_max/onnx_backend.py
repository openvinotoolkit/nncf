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

from typing import Dict, List, Optional, Set, Union

import numpy as np

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.hardware.config import HWConfig
from nncf.common.quantization.structs import QuantizationMode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.onnx.graph.metatypes import onnx_metatypes as om
from nncf.onnx.graph.metatypes.groups import MATMUL_METATYPES
from nncf.onnx.graph.node_utils import get_input_edges_mapping
from nncf.onnx.graph.node_utils import get_quantization_axis
from nncf.onnx.graph.node_utils import get_quantized_tensor_shape
from nncf.onnx.graph.node_utils import get_reduction_shape
from nncf.onnx.graph.transformations.commands import ONNXQuantizerInsertionCommand
from nncf.onnx.graph.transformations.commands import ONNXTargetPoint
from nncf.onnx.hardware.config import ONNXHWConfig
from nncf.onnx.quantization.default_quantization import DEFAULT_ONNX_QUANT_TRAIT_TO_OP_DICT
from nncf.onnx.quantization.quantizer_parameters import convert_fq_params_to_onnx_params
from nncf.onnx.statistics.collectors import ONNXMeanMinMaxStatisticCollector
from nncf.onnx.statistics.collectors import ONNXMinMaxStatisticCollector
from nncf.onnx.statistics.statistics import ONNXMinMaxTensorStatistic
from nncf.parameters import ModelType
from nncf.parameters import TargetDevice
from nncf.quantization.advanced_parameters import AggregatorType
from nncf.quantization.advanced_parameters import StatisticsType
from nncf.quantization.algorithms.min_max.backend import MinMaxAlgoBackend
from nncf.quantization.fake_quantize import FakeQuantizeParameters
from nncf.quantization.range_estimator import RangeEstimatorParameters


class ONNXMinMaxAlgoBackend(MinMaxAlgoBackend):
    @property
    def mat_mul_metatypes(self) -> List[OperatorMetatype]:
        return MATMUL_METATYPES

    @property
    def post_processing_metatypes(self) -> List[OperatorMetatype]:
        return [om.ONNXTopKMetatype, om.ONNXNonMaxSuppressionMetatype]

    @property
    def conv_metatypes(self) -> List[OperatorMetatype]:
        return [om.ONNXConvolutionMetatype]

    @property
    def overflow_fix_metatypes(self) -> List[OperatorMetatype]:
        return [om.ONNXConvolutionMetatype, om.ONNXConvolutionTransposeMetatype, *MATMUL_METATYPES]

    @property
    def add_metatypes(self) -> List[OperatorMetatype]:
        return [om.ONNXAddLayerMetatype]

    @property
    def group_conv_metatypes(self) -> List[OperatorMetatype]:
        return self.conv_metatypes

    @property
    def shapeof_metatypes(self) -> List[OperatorMetatype]:
        return [om.ONNXShapeMetatype]

    @property
    def dropout_metatypes(self) -> List[OperatorMetatype]:
        return []

    @property
    def read_variable_metatypes(self) -> List[OperatorMetatype]:
        return []

    @property
    def scales_unification_map(self) -> Dict[OperatorMetatype, OperatorMetatype]:
        return {om.ONNXConcatMetatype: self.overflow_fix_metatypes}

    @property
    def hw_config(self) -> HWConfig:
        return ONNXHWConfig

    @property
    def quant_trait_op_dict(self) -> Dict[int, OperatorMetatype]:
        return DEFAULT_ONNX_QUANT_TRAIT_TO_OP_DICT

    @staticmethod
    def target_point(target_type: TargetType, target_node_name: str, port_id: int) -> ONNXTargetPoint:
        return ONNXTargetPoint(target_type, target_node_name, port_id)

    @staticmethod
    def create_quantizer_insertion_command(
        nncf_graph: NNCFGraph,
        target_point: ONNXTargetPoint,
        quantizer_config: QuantizerConfig,
        parameters: FakeQuantizeParameters,
    ):
        tensor_type = np.int8 if np.any(parameters.input_low.data < 0) else np.uint8
        if target_point.is_weight_target_point():
            tensor_type = np.int8  # The weight is restricted to have only signed range
        nncf_input_node_next_nodes = ONNXMinMaxAlgoBackend._get_input_edges_mapping(nncf_graph)
        node = nncf_graph.get_node_by_name(target_point.target_node_name)
        axis = get_quantization_axis(quantizer_config.per_channel, node, target_point)
        onnx_parameters = convert_fq_params_to_onnx_params(parameters, quantizer_config.num_bits, tensor_type, axis)
        return ONNXQuantizerInsertionCommand(target_point, nncf_input_node_next_nodes, onnx_parameters)

    @staticmethod
    def unify_statistics(statistics: List[ONNXMinMaxTensorStatistic]) -> ONNXMinMaxTensorStatistic:
        max_values, min_values = [], []
        for statistic in statistics:
            max_values.append(np.array(statistic.max_values).flatten())
            min_values.append(np.array(statistic.min_values).flatten())
        max_values = np.max(max_values, axis=0)
        min_values = np.min(min_values, axis=0)
        return ONNXMinMaxTensorStatistic(min_values=min_values, max_values=max_values)

    @staticmethod
    def _get_input_edges_mapping(nncf_graph: NNCFGraph):
        return get_input_edges_mapping(nncf_graph)

    @staticmethod
    def get_statistic_collector(
        range_estimator_params: RangeEstimatorParameters,
        nncf_graph: NNCFGraph,
        target_point: ONNXTargetPoint,
        quantizer_config: QuantizerConfig,
        inplace: bool,
        num_samples: int = None,
    ) -> Union[ONNXMinMaxStatisticCollector, ONNXMeanMinMaxStatisticCollector]:
        is_per_channel = quantizer_config.per_channel
        node = nncf_graph.get_node_by_name(target_point.target_node_name)
        use_abs_max = quantizer_config.mode == QuantizationMode.SYMMETRIC
        reduction_shape = None  # Per-Tensor
        quantization_axis = get_quantization_axis(is_per_channel, node, target_point)
        quantized_tensor_shape = get_quantized_tensor_shape(nncf_graph, node, target_point)
        if quantization_axis is not None and quantized_tensor_shape is not None:  # Per-Channel
            reduction_shape = get_reduction_shape(quantized_tensor_shape, quantization_axis)

        if (
            range_estimator_params.min.statistics_type == StatisticsType.MIN
            and range_estimator_params.min.aggregator_type == AggregatorType.MIN
            and range_estimator_params.max.statistics_type == StatisticsType.MAX
            and range_estimator_params.max.aggregator_type == AggregatorType.MAX
        ):
            return ONNXMinMaxStatisticCollector(use_abs_max, reduction_shape, num_samples)

        if (
            range_estimator_params.min.statistics_type == StatisticsType.MIN
            and range_estimator_params.min.aggregator_type == AggregatorType.MEAN
            and range_estimator_params.max.statistics_type == StatisticsType.MAX
            and range_estimator_params.max.aggregator_type == AggregatorType.MEAN
        ):
            return ONNXMeanMinMaxStatisticCollector(
                use_per_sample_stats=False,
                use_abs_max=use_abs_max,
                reduction_shape=reduction_shape,
                num_samples=num_samples,
                window_size=None,
            )
        raise RuntimeError(
            "The following range estimator parameters are not supported by ONNX backend by now: "
            f"{str(range_estimator_params)}"
        )

    @staticmethod
    def get_weight_tensor_port_ids(node: NNCFNode) -> List[Optional[int]]:
        return list(node.layer_attributes.weight_attrs.keys())

    @staticmethod
    def get_ignored_metatypes(model_type: ModelType, device: TargetDevice) -> List[OperatorMetatype]:
        types = []
        if model_type == ModelType.TRANSFORMER:
            types = [
                om.ONNXAddLayerMetatype,
                om.ONNXPowMetatype,
                om.ONNXSqueezeMetatype,
                om.ONNXSubMetatype,
                om.ONNXAveragePoolMetatype,
                om.ONNXGlobalAveragePoolMetatype,
                om.ONNXReduceMeanMetatype,
                om.ONNXReduceL2Metatype,
                om.ONNXReduceSumMetatype,
                om.ONNXDivLayerMetatype,
                om.ONNXMaximumMetatype,
                om.ONNXSqrtMetatype,
                om.ONNXReciprocalMetatype,
                om.ONNXBatchNormMetatype,
            ]
            if device != TargetDevice.CPU_SPR:
                types.append(om.ONNXMulLayerMetatype)
        return types

    @staticmethod
    def get_ignored_names_by_layer_attributes(nncf_graph: NNCFGraph) -> List[str]:
        return []

    @staticmethod
    def get_weight_nodes(nncf_graph: NNCFGraph) -> List[NNCFNode]:
        return [node for node in nncf_graph.get_all_nodes() if node.layer_attributes.has_weight()]

    @staticmethod
    def get_weight_name(nncf_graph: NNCFGraph, target_point: ONNXTargetPoint) -> str:
        node_name, port_id = target_point.target_node_name, target_point.port_id
        return nncf_graph.get_node_by_name(node_name).layer_attributes.weight_attrs[port_id]["name"]

    @staticmethod
    def should_quantize_weight(weight_name: str, quantized_weight_names: Set[str]) -> bool:
        # If the nodes share one weight tensor, we should have only one quantizer on that
        return weight_name not in quantized_weight_names
