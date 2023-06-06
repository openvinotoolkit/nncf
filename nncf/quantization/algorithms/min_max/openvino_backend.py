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

from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.hardware.config import HWConfig
from nncf.common.quantization.structs import QuantizationMode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.tensor_statistics.collectors import ReductionShape
from nncf.common.utils.backend import BackendType
from nncf.experimental.common.tensor_statistics.collectors import AGGREGATORS_MAP
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.openvino.graph.metatypes.openvino_metatypes import GENERAL_WEIGHT_LAYER_METATYPES
from nncf.openvino.graph.metatypes.openvino_metatypes import OVAddMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVConvolutionBackpropDataMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVConvolutionMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVDivideMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVGroupConvolutionBackpropDataMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVGroupConvolutionMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVMatMulMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVMultiplyMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVMVNMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVNonMaxSuppressionMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVPowerMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVReadValueMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVReduceMeanMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVShapeOfMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVSqrtMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVSquaredDifferenceMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVSqueezeMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVSubtractMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVTopKMetatype
from nncf.openvino.graph.nncf_graph_builder import OVConstantLayerAttributes
from nncf.openvino.graph.node_utils import get_weight_channel_axes
from nncf.openvino.graph.transformations.commands import OVQuantizerInsertionCommand
from nncf.openvino.graph.transformations.commands import OVTargetPoint
from nncf.openvino.hardware.config import OVHWConfig
from nncf.openvino.quantization.default_quantization import DEFAULT_OV_QUANT_TRAIT_TO_OP_DICT
from nncf.openvino.statistics.collectors import OV_REDUCERS_MAP
from nncf.openvino.statistics.collectors import OVNNCFCollectorTensorProcessor
from nncf.openvino.statistics.statistics import OVMinMaxTensorStatistic
from nncf.parameters import ModelType
from nncf.parameters import TargetDevice
from nncf.quantization.advanced_parameters import RangeEstimatorParameters
from nncf.quantization.advanced_parameters import StatisticsType
from nncf.quantization.algorithms.min_max.backend import ALGO_BACKENDS
from nncf.quantization.algorithms.min_max.backend import MinMaxAlgoBackend
from nncf.quantization.fake_quantize import FakeQuantizeParameters
from nncf.scopes import IgnoredScope


@ALGO_BACKENDS.register(BackendType.OPENVINO)
class OVMinMaxAlgoBackend(MinMaxAlgoBackend):
    @property
    def mat_mul_metatype(self) -> OperatorMetatype:
        return OVMatMulMetatype

    @property
    def post_processing_metatypes(self) -> List[OperatorMetatype]:
        return [OVTopKMetatype, OVNonMaxSuppressionMetatype]

    @property
    def shapeof_metatypes(self) -> List[OperatorMetatype]:
        return [OVShapeOfMetatype]

    @property
    def conv_metatype(self) -> List[OperatorMetatype]:
        return [OVConvolutionMetatype]

    @property
    def overflow_fix_metatypes(self) -> List[OperatorMetatype]:
        return [
            OVConvolutionMetatype,
            OVGroupConvolutionMetatype,
            OVConvolutionBackpropDataMetatype,
            OVGroupConvolutionBackpropDataMetatype,
            OVMatMulMetatype,
        ]

    @property
    def read_variable_metatypes(self) -> List[OperatorMetatype]:
        return [OVReadValueMetatype]

    @property
    def hw_config(self) -> HWConfig:
        return OVHWConfig

    @property
    def quant_trait_op_dict(self) -> Dict[int, OperatorMetatype]:
        return DEFAULT_OV_QUANT_TRAIT_TO_OP_DICT

    @staticmethod
    def target_point(target_type: TargetType, target_node_name: str, port_id: int) -> OVTargetPoint:
        return OVTargetPoint(target_type, target_node_name, port_id)

    @staticmethod
    def create_activation_quantizer_insertion_command(
        nncf_graph: NNCFGraph,
        target_point: OVTargetPoint,
        quantizer_config: QuantizerConfig,
        parameters: FakeQuantizeParameters,
    ) -> OVQuantizerInsertionCommand:
        return OVQuantizerInsertionCommand(target_point, parameters)

    @staticmethod
    def create_weight_quantizer_insertion_command(
        nncf_graph: NNCFGraph,
        target_point: OVTargetPoint,
        quantizer_config: QuantizerConfig,
        parameters: FakeQuantizeParameters,
    ) -> OVQuantizerInsertionCommand:
        return OVQuantizerInsertionCommand(target_point, parameters)

    @staticmethod
    def unify_statistics(statistics: List[OVMinMaxTensorStatistic]) -> OVMinMaxTensorStatistic:
        max_values, min_values = [], []
        for statistic in statistics:
            max_values.append(statistic.max_values)
            min_values.append(statistic.min_values)
        max_values = np.max(max_values, axis=0)
        min_values = np.min(min_values, axis=0)
        return OVMinMaxTensorStatistic(min_values=min_values, max_values=max_values)

    @staticmethod
    def _get_reduction_shape_and_use_abs_max(
        nncf_graph: NNCFGraph, target_point: OVTargetPoint, quantizer_config: QuantizerConfig
    ) -> Tuple[ReductionShape, bool]:
        use_abs_max = quantizer_config.mode == QuantizationMode.SYMMETRIC
        if not quantizer_config.per_channel:
            return None, use_abs_max

        node = nncf_graph.get_node_by_name(target_point.target_node_name)
        if not target_point.is_weight_target_point():
            if target_point.type == TargetType.PRE_LAYER_OPERATION:
                shape = nncf_graph.get_input_edges(node)[target_point.port_id].tensor_shape
            elif target_point.type == TargetType.POST_LAYER_OPERATION:
                shape = nncf_graph.get_output_edges(node)[target_point.port_id].tensor_shape
            else:
                raise NotImplementedError(f"Unsupported target point type {target_point.type}.")

            # TODO (l-bat): Disable quantizer propogation through layout changing operations
            channel_axis = 1  # OpenVINO activations have channel first layout: [N, C, Z, Y, X]
            axes = tuple(i for i in range(len(shape)) if i != channel_axis)
            return axes, use_abs_max

        assert isinstance(node.layer_attributes, OVConstantLayerAttributes)
        const_shape = node.layer_attributes.const_attrs[target_point.port_id]["shape"]

        if quantizer_config.per_channel:
            channel_axes = get_weight_channel_axes(node, target_point.port_id)
            axes = tuple(i for i in range(len(const_shape)) if i not in channel_axes)
        else:
            axes = tuple(range(len(const_shape)))

        return axes, use_abs_max

    @staticmethod
    def _get_num_samples(num_samples, target_point: OVTargetPoint):
        if target_point.is_weight_target_point():
            return 1
        return num_samples

    @staticmethod
    def get_statistic_collector(
        range_estimator_params: RangeEstimatorParameters,
        nncf_graph: NNCFGraph,
        target_point: OVTargetPoint,
        quantizer_config: QuantizerConfig,
        inplace: bool,
        num_samples: int = None,
    ) -> TensorCollector:
        reduction_shape, use_abs_max = OVMinMaxAlgoBackend._get_reduction_shape_and_use_abs_max(
            nncf_graph, target_point, quantizer_config
        )
        _num_samples = OVMinMaxAlgoBackend._get_num_samples(num_samples, target_point)

        collector = TensorCollector(OVMinMaxTensorStatistic)
        for params, container_key in zip(
            [range_estimator_params.min, range_estimator_params.max],
            [OVMinMaxTensorStatistic.MIN_STAT, OVMinMaxTensorStatistic.MAX_STAT],
        ):
            if not params.statistics_type in OV_REDUCERS_MAP:
                raise RuntimeError(
                    f"Statistic type: {params.statistics_type} is not supported for OpenVino PTQ backend yet."
                )

            if not params.aggregator_type in AGGREGATORS_MAP:
                raise RuntimeError(
                    f"Aggregator type: {params.aggregator_type} is not supported for OpenVino PTQ backend yet."
                )

            kwargs = {"reduction_shape": reduction_shape, "inplace": inplace}
            if params.statistics_type in [StatisticsType.QUANTILE, StatisticsType.ABS_QUANTILE]:
                if container_key == OVMinMaxTensorStatistic.MIN_STAT:
                    quantile = params.quantile_outlier_prob
                else:
                    quantile = 1 - params.quantile_outlier_prob
                kwargs.update({"quantile": [quantile]})
            # TODO(dlyakhov): merge two quantile aggregators in one
            statistic_type = params.statistics_type
            if use_abs_max and statistic_type == StatisticsType.MAX:
                statistic_type = StatisticsType.ABS_MAX
            reducer = OV_REDUCERS_MAP[statistic_type](**kwargs)

            kwargs = {"num_samples": _num_samples, "tensor_processor": OVNNCFCollectorTensorProcessor}
            aggregator = AGGREGATORS_MAP[params.aggregator_type](**kwargs)

            collector.register_statistic_branch(container_key, reducer, aggregator)
        return collector

    @staticmethod
    def get_weight_tensor_port_ids(node: NNCFNode) -> List[Optional[int]]:
        return node.layer_attributes.get_const_port_ids()

    @staticmethod
    def get_ignored_scope(model_type: ModelType, device: TargetDevice) -> IgnoredScope:
        if model_type == ModelType.TRANSFORMER:
            types = []
            metatypes_to_add = [
                OVAddMetatype,
                OVPowerMetatype,
                OVSqueezeMetatype,
                OVSubtractMetatype,
                OVReduceMeanMetatype,
                OVSquaredDifferenceMetatype,
                OVMVNMetatype,
                OVDivideMetatype,
                OVSqrtMetatype,
            ]
            if device != TargetDevice.CPU_SPR:
                metatypes_to_add.append(OVMultiplyMetatype)
            for metatype in metatypes_to_add:
                types.extend(metatype.get_all_aliases())
            return IgnoredScope(types=types)
        return IgnoredScope()

    @staticmethod
    def get_weight_nodes(nncf_graph: NNCFGraph) -> List[NNCFNode]:
        return [
            node
            for node in nncf_graph.get_all_nodes()
            if isinstance(node.layer_attributes, OVConstantLayerAttributes)
            and node.metatype in GENERAL_WEIGHT_LAYER_METATYPES
        ]

    @staticmethod
    def get_weight_name(nncf_graph: NNCFGraph, target_point: OVTargetPoint) -> str:
        node = nncf_graph.get_node_by_name(target_point.target_node_name)
        return node.layer_attributes.const_attrs[target_point.port_id]["name"]

    @staticmethod
    def should_quantize_weight(weight_name: str, quantized_weight_names: Set[str]) -> bool:
        return True
