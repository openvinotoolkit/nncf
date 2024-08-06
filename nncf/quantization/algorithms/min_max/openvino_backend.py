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

import nncf
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.hardware.config import HWConfig
from nncf.common.quantization.structs import QuantizerConfig
from nncf.experimental.common.tensor_statistics.collectors import AGGREGATORS_MAP
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.experimental.common.tensor_statistics.statistics import MinMaxTensorStatistic
from nncf.openvino.graph.layer_attributes import OVLayerAttributes
from nncf.openvino.graph.metatypes import openvino_metatypes as om
from nncf.openvino.graph.metatypes.groups import OPERATIONS_WITH_WEIGHTS
from nncf.openvino.graph.model_utils import get_start_nodes_for_activation_path_tracing
from nncf.openvino.graph.node_utils import get_weight_channel_axes
from nncf.openvino.graph.transformations.commands import OVConvertInsertionCommand
from nncf.openvino.graph.transformations.commands import OVQuantizerInsertionCommand
from nncf.openvino.graph.transformations.commands import OVTargetPoint
from nncf.openvino.hardware.config import OVHWConfig
from nncf.openvino.quantization.default_quantization import DEFAULT_OV_QUANT_TRAIT_TO_OP_DICT
from nncf.openvino.statistics.collectors import OV_REDUCERS_MAP
from nncf.parameters import ModelType
from nncf.parameters import TargetDevice
from nncf.quantization.advanced_parameters import RangeEstimatorParameters
from nncf.quantization.advanced_parameters import StatisticsType
from nncf.quantization.algorithms.min_max.backend import MinMaxAlgoBackend
from nncf.quantization.fake_quantize import FakeConvertParameters
from nncf.quantization.fake_quantize import FakeQuantizeParameters
from nncf.quantization.range_estimator import AggregatorType


class OVMinMaxAlgoBackend(MinMaxAlgoBackend):
    @property
    def mat_mul_metatypes(self) -> List[OperatorMetatype]:
        return [om.OVMatMulMetatype]

    @property
    def post_processing_metatypes(self) -> List[OperatorMetatype]:
        return [om.OVTopKMetatype, om.OVNonMaxSuppressionMetatype]

    @property
    def conv_metatypes(self) -> List[OperatorMetatype]:
        return [om.OVConvolutionMetatype]

    @property
    def overflow_fix_metatypes(self) -> List[OperatorMetatype]:
        return [
            om.OVConvolutionMetatype,
            om.OVGroupConvolutionMetatype,
            om.OVConvolutionBackpropDataMetatype,
            om.OVGroupConvolutionBackpropDataMetatype,
            om.OVMatMulMetatype,
        ]

    @property
    def add_metatypes(self) -> List[OperatorMetatype]:
        return [om.OVAddMetatype]

    @property
    def group_conv_metatypes(self) -> List[OperatorMetatype]:
        return [om.OVGroupConvolutionMetatype]

    @property
    def shapeof_metatypes(self) -> List[OperatorMetatype]:
        return [om.OVShapeOfMetatype]

    @property
    def dropout_metatypes(self) -> List[OperatorMetatype]:
        return []

    @property
    def read_variable_metatypes(self) -> List[OperatorMetatype]:
        return [om.OVReadValueMetatype]

    @property
    def scaled_dot_product_attention_metatypes(self) -> List[OperatorMetatype]:
        return [om.OVScaledDotProductAttentionMetatype]

    @property
    def scales_unification_map(self) -> Dict[OperatorMetatype, OperatorMetatype]:
        return {om.OVConcatMetatype: self.overflow_fix_metatypes}

    @property
    def hw_config(self) -> HWConfig:
        return OVHWConfig

    @property
    def quant_trait_op_dict(self) -> Dict[int, OperatorMetatype]:
        return DEFAULT_OV_QUANT_TRAIT_TO_OP_DICT

    @staticmethod
    def get_start_nodes_for_activation_path_tracing(nncf_graph: NNCFGraph) -> List[NNCFNode]:
        return get_start_nodes_for_activation_path_tracing(nncf_graph)

    @staticmethod
    def target_point(target_type: TargetType, target_node_name: str, port_id: int) -> OVTargetPoint:
        return OVTargetPoint(target_type, target_node_name, port_id)

    @staticmethod
    def create_quantizer_insertion_command(
        nncf_graph: NNCFGraph,
        target_point: OVTargetPoint,
        quantizer_config: QuantizerConfig,
        parameters: FakeQuantizeParameters,
    ) -> OVQuantizerInsertionCommand:
        return OVQuantizerInsertionCommand(target_point, parameters)

    @staticmethod
    def create_unified_scales_quantizers_insertion_commands(
        nncf_graph: NNCFGraph,
        target_points: List[OVTargetPoint],
        quantizer_config: QuantizerConfig,
        parameters: FakeQuantizeParameters,
    ) -> List[OVQuantizerInsertionCommand]:
        return [OVQuantizerInsertionCommand(target_point, parameters) for target_point in target_points]

    @staticmethod
    def create_convert_insertion_command(
        target_point: OVTargetPoint,
        parameters: FakeConvertParameters,
    ) -> OVQuantizerInsertionCommand:
        return OVConvertInsertionCommand(target_point, parameters)

    @staticmethod
    def get_target_point_shape(nncf_graph: NNCFGraph, node: NNCFNode, target_point: OVTargetPoint) -> Tuple[int, ...]:
        if target_point.is_weight_target_point():
            return node.layer_attributes.constant_attributes[target_point.port_id]["shape"]

        if target_point.type == TargetType.PRE_LAYER_OPERATION:
            edge = nncf_graph.get_input_edge_by_port_id(node, target_point.port_id)
            return edge.tensor_shape
        elif target_point.type == TargetType.POST_LAYER_OPERATION:
            # NOTE: Assumes that all output edges for the `node` with `output_port_id`
            # equal to `target_point.port_id` should have the same `tensor_shape` value.
            edges = nncf_graph.get_output_edges_by_port_id(node, target_point.port_id)
            return edges[0].tensor_shape

        raise NotImplementedError(f"Unsupported target point type {target_point.type}.")

    @staticmethod
    def get_weight_quantization_axes(node: NNCFNode, target_point: OVTargetPoint, ndims: int) -> Tuple[int]:
        return tuple(get_weight_channel_axes(node))

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
            if params.statistics_type not in OV_REDUCERS_MAP:
                raise nncf.InternalError(
                    f"Statistic type: {params.statistics_type} is not supported for OpenVino PTQ backend yet."
                )
            if params.aggregator_type not in AGGREGATORS_MAP:
                raise nncf.InternalError(
                    f"Aggregator type: {params.aggregator_type} is not supported for OpenVino PTQ backend yet."
                )
            kwargs = {"reduction_axes": reduction_axes, "inplace": inplace}
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
            reducer = OV_REDUCERS_MAP[statistic_type](**kwargs)

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
        return node.layer_attributes.get_const_port_ids()

    @staticmethod
    def get_ignored_metatypes(model_type: ModelType, device: TargetDevice) -> List[OperatorMetatype]:
        types = []
        if model_type == ModelType.TRANSFORMER:
            types = [
                om.OVAddMetatype,
                om.OVPowerMetatype,
                om.OVSqueezeMetatype,
                om.OVSubtractMetatype,
                om.OVAvgPoolMetatype,
                om.OVReduceMeanMetatype,
                om.OVReduceL2Metatype,
                om.OVSumMetatype,
                om.OVSquaredDifferenceMetatype,
                om.OVMVNMetatype,
                om.OVBatchNormMetatype,
                om.OVDivideMetatype,
                om.OVSqrtMetatype,
                om.OVMaximumMetatype,
                # Сomparison operations
                om.OVGreaterEqualMetatype,
                om.OVGreaterMetatype,
                om.OVLessEqualMetatype,
                om.OVLessMetatype,
                om.OVNotEqualMetatype,
                om.OVEqualMetatype,
            ]
            if device != TargetDevice.CPU_SPR:
                types.append(om.OVMultiplyMetatype)
        return types

    @staticmethod
    def get_ignored_names_by_layer_attributes(nncf_graph: NNCFGraph) -> Set[str]:
        ignored_names = set()
        target_nodes = nncf_graph.get_nodes_by_metatypes([om.OVGRUSequenceMetatype])
        for node in target_nodes:
            if (
                isinstance(node.layer_attributes, OVLayerAttributes)
                and node.layer_attributes.input_attributes["linear_before_reset"]
            ):
                ignored_names.add(node.node_name)
        return ignored_names

    @staticmethod
    def get_weight_nodes(nncf_graph: NNCFGraph) -> List[NNCFNode]:
        return [
            node
            for node in nncf_graph.get_all_nodes()
            if isinstance(node.layer_attributes, OVLayerAttributes) and node.metatype in OPERATIONS_WITH_WEIGHTS
        ]

    @staticmethod
    def get_weight_name(nncf_graph: NNCFGraph, target_point: OVTargetPoint) -> str:
        node = nncf_graph.get_node_by_name(target_point.target_node_name)
        return node.layer_attributes.constant_attributes[target_point.port_id]["name"]

    @staticmethod
    def should_quantize_weight(weight_name: str, quantized_weight_names: Set[str]) -> bool:
        return True
