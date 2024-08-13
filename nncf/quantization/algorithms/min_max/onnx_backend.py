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

from typing import Dict, List, Optional, Set, Tuple

import numpy as np

import nncf
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationCommand
from nncf.common.hardware.config import HWConfig
from nncf.common.quantization.structs import QuantizerConfig
from nncf.experimental.common.tensor_statistics.collectors import AGGREGATORS_MAP
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.experimental.common.tensor_statistics.statistics import MinMaxTensorStatistic
from nncf.onnx.graph.metatypes import onnx_metatypes as om
from nncf.onnx.graph.metatypes.groups import MATMUL_METATYPES
from nncf.onnx.graph.node_utils import get_input_edges_mapping
from nncf.onnx.graph.node_utils import get_quantized_tensor_shape
from nncf.onnx.graph.node_utils import get_weight_quantization_axis
from nncf.onnx.graph.transformations.commands import ONNXQuantizerInsertionCommand
from nncf.onnx.graph.transformations.commands import ONNXTargetPoint
from nncf.onnx.hardware.config import ONNXHWConfig
from nncf.onnx.quantization.default_quantization import DEFAULT_ONNX_QUANT_TRAIT_TO_OP_DICT
from nncf.onnx.quantization.quantizer_parameters import convert_fq_params_to_onnx_params
from nncf.onnx.statistics.collectors import ONNX_REDUCERS_MAP
from nncf.parameters import ModelType
from nncf.parameters import TargetDevice
from nncf.quantization.advanced_parameters import StatisticsType
from nncf.quantization.algorithms.min_max.backend import MinMaxAlgoBackend
from nncf.quantization.fake_quantize import FakeConvertParameters
from nncf.quantization.fake_quantize import FakeQuantizeParameters
from nncf.quantization.range_estimator import AggregatorType
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
        return [
            om.ONNXConvolutionMetatype,
            om.ONNXConvolutionTransposeMetatype,
            *MATMUL_METATYPES,
        ]

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
    def scaled_dot_product_attention_metatypes(self) -> List[OperatorMetatype]:
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
    def get_start_nodes_for_activation_path_tracing(
        nncf_graph: NNCFGraph,
    ) -> List[NNCFNode]:
        return nncf_graph.get_input_nodes()

    @staticmethod
    def target_point(target_type: TargetType, target_node_name: str, port_id: int) -> ONNXTargetPoint:
        return ONNXTargetPoint(target_type, target_node_name, port_id)

    @staticmethod
    def create_quantizer_insertion_command(
        nncf_graph: NNCFGraph,
        target_point: ONNXTargetPoint,
        quantizer_config: QuantizerConfig,
        parameters: FakeQuantizeParameters,
    ) -> ONNXQuantizerInsertionCommand:
        tensor_type = np.int8 if np.any(parameters.input_low.data < 0) else np.uint8
        is_weight = target_point.is_weight_target_point()
        if is_weight:
            tensor_type = np.int8  # The weight is restricted to have only signed range
        nncf_input_node_next_nodes = ONNXMinMaxAlgoBackend._get_input_edges_mapping(nncf_graph)
        node = nncf_graph.get_node_by_name(target_point.target_node_name)
        axis = ()
        if quantizer_config.per_channel:
            axis = ONNXMinMaxAlgoBackend.get_weight_quantization_axes(node, target_point, None) if is_weight else (1,)
        onnx_parameters = convert_fq_params_to_onnx_params(parameters, quantizer_config.num_bits, tensor_type, axis)
        return ONNXQuantizerInsertionCommand(target_point, nncf_input_node_next_nodes, onnx_parameters)

    @staticmethod
    def create_unified_scales_quantizers_insertion_commands(
        nncf_graph: NNCFGraph,
        target_points: List[ONNXTargetPoint],
        quantizer_config: QuantizerConfig,
        parameters: FakeQuantizeParameters,
    ) -> List[ONNXQuantizerInsertionCommand]:
        return [
            ONNXMinMaxAlgoBackend.create_quantizer_insertion_command(
                nncf_graph, target_point, quantizer_config, parameters
            )
            for target_point in target_points
        ]

    @staticmethod
    def create_convert_insertion_command(
        target_point: ONNXTargetPoint,
        parameters: FakeConvertParameters,
    ) -> TransformationCommand:
        raise nncf.InternalError("FakeConvert insertion not implemented in ONNX backend!")

    @staticmethod
    def _get_input_edges_mapping(nncf_graph: NNCFGraph):
        return get_input_edges_mapping(nncf_graph)

    @staticmethod
    def get_target_point_shape(nncf_graph: NNCFGraph, node: NNCFNode, target_point: ONNXTargetPoint) -> Tuple[int, ...]:
        return get_quantized_tensor_shape(nncf_graph, node, target_point)

    @staticmethod
    def get_weight_quantization_axes(node: NNCFNode, target_point: ONNXTargetPoint, ndims: int) -> Tuple[int]:
        return (get_weight_quantization_axis(node, target_point.port_id),)

    @staticmethod
    def get_statistic_collector(
        range_estimator_params: RangeEstimatorParameters,
        use_abs_max: bool,
        reduction_axes: Optional[Tuple[int, ...]],
        aggregation_axes: Optional[Tuple[int, ...]],
        inplace: bool,
        num_samples: Optional[int] = None,
    ) -> TensorCollector:
        collector = TensorCollector(MinMaxTensorStatistic)
        for params, container_key in zip(
            [range_estimator_params.min, range_estimator_params.max],
            [MinMaxTensorStatistic.MIN_STAT, MinMaxTensorStatistic.MAX_STAT],
        ):
            if params.statistics_type not in ONNX_REDUCERS_MAP:
                raise nncf.InternalError(
                    f"Statistic type: {params.statistics_type} is not supported for ONNX PTQ backend yet."
                )
            if params.aggregator_type not in AGGREGATORS_MAP:
                raise nncf.InternalError(
                    f"Aggregator type: {params.aggregator_type} is not supported for ONNX PTQ backend yet."
                )
            kwargs = {"reduction_axes": reduction_axes, "inplace": False}
            if params.statistics_type in [StatisticsType.QUANTILE, StatisticsType.ABS_QUANTILE]:
                if container_key == MinMaxTensorStatistic.MIN_STAT:
                    quantile = params.quantile_outlier_prob
                else:
                    quantile = 1 - params.quantile_outlier_prob
                kwargs.update({"quantile": [quantile]})
            # TODO(dlyakhov): merge two quantile aggregators in one
            statistic_type = params.statistics_type
            if use_abs_max and statistic_type == StatisticsType.MAX:
                statistic_type = StatisticsType.ABS_MAX
            reducer = ONNX_REDUCERS_MAP[statistic_type](**kwargs)

            kwargs = {
                "num_samples": num_samples,
                "aggregation_axes": aggregation_axes,
            }
            if params.aggregator_type in [AggregatorType.MEAN_NO_OUTLIERS, AggregatorType.MEDIAN_NO_OUTLIERS]:
                kwargs.update({"quantile": params.quantile_outlier_prob})
            aggregator = AGGREGATORS_MAP[params.aggregator_type](**kwargs)

            collector.register_statistic_branch(container_key, reducer, aggregator)
        return collector

    @staticmethod
    def get_weight_tensor_port_ids(node: NNCFNode, graph: NNCFGraph) -> List[Optional[int]]:
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
                # Сomparison operations
                om.ONNXGreaterMetatype,
                om.ONNXLessMetatype,
                om.ONNXEqualMetatype,
            ]
            if device != TargetDevice.CPU_SPR:
                types.append(om.ONNXMulLayerMetatype)
        return types

    @staticmethod
    def get_ignored_names_by_layer_attributes(nncf_graph: NNCFGraph) -> Set[str]:
        return set()

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
