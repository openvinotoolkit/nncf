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

from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.hardware.config import HWConfig
from nncf.common.quantization.structs import QuantizationMode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.utils.backend import BackendType
from nncf.onnx.graph.metatypes import onnx_metatypes as om
from nncf.onnx.graph.node_utils import get_input_edges_mapping
from nncf.onnx.graph.node_utils import transpose_axis
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
from nncf.quantization.algorithms.min_max.backend import ALGO_BACKENDS
from nncf.quantization.algorithms.min_max.backend import MinMaxAlgoBackend
from nncf.quantization.fake_quantize import FakeQuantizeParameters
from nncf.quantization.range_estimator import RangeEstimatorParameters
from nncf.scopes import IgnoredScope


@ALGO_BACKENDS.register(BackendType.ONNX)
class ONNXMinMaxAlgoBackend(MinMaxAlgoBackend):
    @property
    def mat_mul_metatype(self) -> OperatorMetatype:
        return om.MATMUL_METATYPES

    @property
    def post_processing_metatypes(self) -> List[OperatorMetatype]:
        return [om.ONNXTopKMetatype, om.ONNXNonMaxSuppressionMetatype]

    @property
    def shapeof_metatypes(self) -> List[OperatorMetatype]:
        return [om.ONNXShapeMetatype]

    @property
    def conv_metatype(self) -> List[OperatorMetatype]:
        return [om.ONNXConvolutionMetatype]

    @property
    def overflow_fix_metatypes(self) -> List[OperatorMetatype]:
        return [om.ONNXConvolutionMetatype, om.ONNXConvolutionTransposeMetatype, *om.MATMUL_METATYPES]

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
    def create_activation_quantizer_insertion_command(
        nncf_graph: NNCFGraph,
        target_point: ONNXTargetPoint,
        quantizer_config: QuantizerConfig,
        parameters: FakeQuantizeParameters,
    ) -> ONNXQuantizerInsertionCommand:
        nncf_input_node_next_nodes = ONNXMinMaxAlgoBackend._get_input_edges_mapping(nncf_graph)
        axis = ONNXMinMaxAlgoBackend._get_axis(nncf_graph, target_point, quantizer_config)
        tensor_type = np.int8 if np.any(parameters.input_low < 0) else np.uint8
        onnx_parameters = convert_fq_params_to_onnx_params(parameters, quantizer_config.num_bits, tensor_type, axis)
        return ONNXQuantizerInsertionCommand(target_point, nncf_input_node_next_nodes, onnx_parameters)

    @staticmethod
    def create_weight_quantizer_insertion_command(
        nncf_graph: NNCFGraph,
        target_point: ONNXTargetPoint,
        quantizer_config: QuantizerConfig,
        parameters: FakeQuantizeParameters,
    ) -> ONNXQuantizerInsertionCommand:
        if quantizer_config.signedness_to_force is False:
            raise ValueError(
                "The HW expects to have signed quantization of weights, "
                "while the quantizer configuration for weights contains signedness_to_force=False."
            )

        tensor_type = np.int8  # The weight is restricted to have only signed range
        nncf_input_node_next_nodes = ONNXMinMaxAlgoBackend._get_input_edges_mapping(nncf_graph)
        axis = ONNXMinMaxAlgoBackend._get_axis(nncf_graph, target_point, quantizer_config)
        onnx_parameters = convert_fq_params_to_onnx_params(parameters, quantizer_config.num_bits, tensor_type, axis)
        return ONNXQuantizerInsertionCommand(target_point, nncf_input_node_next_nodes, onnx_parameters)

    @staticmethod
    def unify_statistics(statistics: List[ONNXMinMaxTensorStatistic]) -> ONNXMinMaxTensorStatistic:
        max_values, min_values = [], []
        for statistic in statistics:
            max_values.append(statistic.max_values)
            min_values.append(statistic.min_values)
        max_values = np.max(max_values, axis=0)
        min_values = np.min(min_values, axis=0)
        return ONNXMinMaxTensorStatistic(min_values=min_values, max_values=max_values)

    @staticmethod
    def _get_input_edges_mapping(nncf_graph: NNCFGraph):
        return get_input_edges_mapping(nncf_graph)

    @staticmethod
    def _get_axis(
        nncf_graph: NNCFGraph, target_point: ONNXTargetPoint, quantizer_config: QuantizerConfig
    ) -> Optional[int]:
        if not quantizer_config.per_channel:
            return None
        if not target_point.is_weight_target_point():
            return 1
        node = nncf_graph.get_node_by_name(target_point.target_node_name)

        weight_channel_axis = node.metatype.weight_channel_axis
        if node.layer_attributes.has_node_attrs():
            if node.metatype == om.ONNXGemmMetatype:
                weight_shape = node.layer_attributes.weight_attrs[target_point.port_id]["shape"]
                if (
                    target_point.port_id == 0
                    and node.layer_attributes.node_attrs["transA"] == 1
                    or target_point.port_id == 1
                    and node.layer_attributes.node_attrs["transB"] == 1
                ):
                    weight_channel_axis = transpose_axis(weight_shape, weight_channel_axis)
        return weight_channel_axis

    @staticmethod
    def _get_reduction_shape_and_use_abs_max(
        nncf_graph: NNCFGraph, target_point: ONNXTargetPoint, quantizer_config: QuantizerConfig
    ) -> Tuple[Optional[Tuple[int, ...]], bool]:
        use_abs_max = quantizer_config.mode == QuantizationMode.SYMMETRIC
        if not quantizer_config.per_channel:
            return None, use_abs_max

        if not target_point.is_weight_target_point():
            # TODO: support reduction shapes for 3D-5D conv cases
            return (0, 2, 3), use_abs_max

        # Calculate reduction shape for weight statistic collector
        node = nncf_graph.get_node_by_name(target_point.target_node_name)
        assert node.layer_attributes.has_weight()
        weight_shape = node.layer_attributes.weight_attrs[target_point.port_id]["shape"]
        reduction_shape = list(range(len(weight_shape)))

        axis = ONNXMinMaxAlgoBackend._get_axis(nncf_graph, target_point, quantizer_config)
        reduction_shape.pop(axis)
        return tuple(reduction_shape), use_abs_max

    @staticmethod
    def get_statistic_collector(
        range_estimator_params: RangeEstimatorParameters,
        nncf_graph: NNCFGraph,
        target_point: ONNXTargetPoint,
        quantizer_config: QuantizerConfig,
        inplace: bool,
        num_samples: int = None,
    ) -> Union[ONNXMinMaxStatisticCollector, ONNXMeanMinMaxStatisticCollector]:
        reduction_shape, use_abs_max = ONNXMinMaxAlgoBackend._get_reduction_shape_and_use_abs_max(
            nncf_graph, target_point, quantizer_config
        )

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
    def get_ignored_scope(model_type: ModelType, device: TargetDevice) -> IgnoredScope:
        if model_type == ModelType.TRANSFORMER:
            types = []
            metatypes_to_add = [
                om.ONNXAddLayerMetatype,
                om.ONNXPowMetatype,
                om.ONNXSqueezeMetatype,
                om.ONNXSubMetatype,
                om.ONNXReduceMeanMetatype,
                om.ONNXReduceL2Metatype,
                om.ONNXReduceSumMetatype,
                om.ONNXDivLayerMetatype,
                om.ONNXMaximumMetatype,
                om.ONNXSqrtMetatype,
                om.ONNXReciprocalMetatype,
            ]
            if device != TargetDevice.CPU_SPR:
                metatypes_to_add.append(om.ONNXMulLayerMetatype)
            for metatype in metatypes_to_add:
                types.extend(metatype.get_all_aliases())
            return IgnoredScope(types=types)
        return IgnoredScope()

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
