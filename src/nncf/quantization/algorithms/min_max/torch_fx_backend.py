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

from typing import Optional

import torch
from torch.quantization.fake_quantize import FakeQuantize

import nncf
import nncf.torch.graph.operator_metatypes as om
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationCommand
from nncf.common.hardware.config import HWConfig
from nncf.common.quantization.quantizer_propagation.structs import QuantizationTrait
from nncf.common.quantization.structs import QuantizationScheme
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import TypedQuantizerConfig
from nncf.experimental.common.tensor_statistics.collectors import REDUCERS_MAP
from nncf.experimental.common.tensor_statistics.collectors import TensorReducerBase
from nncf.experimental.torch.fx.commands import FXApplyTransformationCommand
from nncf.experimental.torch.fx.model_utils import get_target_point
from nncf.experimental.torch.fx.transformations import qdq_insertion_transformation_builder
from nncf.parameters import ModelType
from nncf.parameters import TargetDevice
from nncf.quantization.algorithms.min_max.backend import MinMaxAlgoBackend
from nncf.quantization.fake_quantize import FakeConvertParameters
from nncf.quantization.fake_quantize import FakeQuantizeParameters
from nncf.quantization.range_estimator import StatisticsType
from nncf.tensor.definitions import TensorDataType
from nncf.torch.graph.graph import PTNNCFGraph
from nncf.torch.graph.graph import PTTargetPoint
from nncf.torch.graph.operator_metatypes import ELEMENTWISE_OPERATIONS
from nncf.torch.graph.operator_metatypes import MATMUL_METATYPES
from nncf.torch.graph.transformations.commands import PTSharedFnInsertionCommand
from nncf.torch.hardware.config import PTHWConfig
from nncf.torch.model_graph_manager import get_weight_nodes
from nncf.torch.model_graph_manager import get_weight_tensor_port_ids
from nncf.torch.model_graph_manager import is_matmul_with_constant
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.quantization.default_quantization import DEFAULT_PT_QUANT_TRAIT_TO_OP_DICT
from nncf.torch.quantization.quantize_functions import get_scale_zp_from_input_low_input_high


class FXMinMaxAlgoBackend(MinMaxAlgoBackend):
    @property
    def preserved_metatypes(self) -> list[OperatorMetatype]:
        return []

    @property
    def mat_mul_metatypes(self) -> list[OperatorMetatype]:
        return MATMUL_METATYPES

    @property
    def post_processing_metatypes(self) -> list[OperatorMetatype]:
        return []

    @property
    def shapeof_metatypes(self) -> list[OperatorMetatype]:
        return []

    @property
    def dropout_metatypes(self) -> list[OperatorMetatype]:
        return [om.PTDropoutMetatype]

    @property
    def read_variable_metatypes(self) -> list[OperatorMetatype]:
        return []

    @property
    def conv_metatypes(self) -> list[OperatorMetatype]:
        return [om.PTConv1dMetatype, om.PTConv2dMetatype, om.PTConv3dMetatype]

    @property
    def elementwise_metatypes(self) -> list[OperatorMetatype]:
        return ELEMENTWISE_OPERATIONS

    @property
    def overflow_fix_metatypes(self) -> list[OperatorMetatype]:
        return [
            om.PTConv1dMetatype,
            om.PTConv2dMetatype,
            om.PTConv3dMetatype,
            om.PTLinearMetatype,
            om.PTConvTranspose1dMetatype,
            om.PTConvTranspose2dMetatype,
            om.PTConvTranspose3dMetatype,
        ]

    @property
    def add_metatypes(self) -> list[OperatorMetatype]:
        return [om.PTAddMetatype]

    @property
    def group_conv_metatypes(self) -> list[OperatorMetatype]:
        return self.conv_metatypes

    @property
    def scaled_dot_product_attention_metatypes(self) -> list[OperatorMetatype]:
        return [om.PTScaledDotProductAttentionMetatype]

    @property
    def scales_unification_map(self) -> dict[OperatorMetatype, OperatorMetatype]:
        return {om.PTCatMetatype: self.overflow_fix_metatypes + self.scaled_dot_product_attention_metatypes}

    @property
    def hw_config(self) -> HWConfig:
        return PTHWConfig

    @property
    def quant_trait_op_dict(self) -> dict[int, OperatorMetatype]:
        return DEFAULT_PT_QUANT_TRAIT_TO_OP_DICT

    @property
    def reducer_map(self) -> dict[StatisticsType, TensorReducerBase]:
        return REDUCERS_MAP

    @property
    def supports_inplace_statistics(self) -> bool:
        return False

    @staticmethod
    def get_start_nodes_for_activation_path_tracing(nncf_graph: PTNNCFGraph) -> list[NNCFNode]:
        return nncf_graph.get_input_nodes() + nncf_graph.get_nodes_by_metatypes(
            DEFAULT_PT_QUANT_TRAIT_TO_OP_DICT[QuantizationTrait.OUTPUT_QUANTIZATION_AS_WEIGHTS]
        )

    @staticmethod
    def target_point(target_type: TargetType, target_node_name: str, port_id: int) -> PTTargetPoint:
        return get_target_point(target_type, target_node_name, port_id)

    @staticmethod
    def create_convert_insertion_command(
        target_point: PTTargetPoint,
        parameters: FakeConvertParameters,
    ) -> TransformationCommand:
        msg = "FakeConvert insertion not implemented in PyTorch backend!"
        raise nncf.InternalError(msg)

    @staticmethod
    def get_target_point_shape(nncf_graph: PTNNCFGraph, node: NNCFNode, target_point: PTTargetPoint) -> tuple[int, ...]:
        return nncf_graph.get_input_shape_for_insertion_point(target_point)

    @staticmethod
    def get_weight_quantization_axes(node: NNCFNode, target_point: PTTargetPoint, ndims: int) -> tuple[int]:
        # TODO(dlyakhov): support transpose conv and other cases
        return (0,)

    @staticmethod
    def get_weight_tensor_port_ids(node: NNCFNode, graph: NNCFGraph) -> list[Optional[int]]:
        return get_weight_tensor_port_ids(node, graph)

    @staticmethod
    def get_weight_name(nncf_graph: NNCFGraph, target_point: PTTargetPoint) -> str:
        weighted_node = nncf_graph.get_node_by_name(target_point.target_node_name)
        weight_edge = nncf_graph.get_input_edge_by_port_id(weighted_node, target_point.input_port_id)
        weight = weight_edge.from_node
        return weight.node_name

    @staticmethod
    def should_quantize_weight(weight_name: str, quantized_weight_names: set[str]) -> bool:
        # If the nodes share one weight tensor, we should have only one quantizer on that
        return weight_name not in quantized_weight_names

    @staticmethod
    def get_weight_config(config: QuantizerConfig, model: NNCFNetwork) -> QuantizerConfig:
        return config

    @staticmethod
    def _get_channel_axis(is_weight_quantizer: bool) -> int:
        if is_weight_quantizer:
            # TODO(dlyakhov): support transpose conv/ make channel_idx common
            return 0
        return 1

    @staticmethod
    def _create_quantizer(
        quantizer_config: QuantizerConfig,
        parameters: FakeQuantizeParameters,
        is_weight_quantizer: bool,
    ) -> FakeQuantize:
        per_channel = quantizer_config.per_channel
        dtype = None
        if isinstance(quantizer_config, TypedQuantizerConfig):
            dtype = quantizer_config.dest_dtype

            if dtype not in [TensorDataType.int8, TensorDataType.uint8]:
                msg = f"Quantization configurations with dest_dtype=={dtype} are not supported."
                raise nncf.ParameterNotSupportedError(msg)

        elif quantizer_config.mode != QuantizationScheme.SYMMETRIC:
            dtype = TensorDataType.uint8
        else:
            dtype = (
                TensorDataType.int8
                if quantizer_config.signedness_to_force or torch.any(parameters.input_low.data < 0.0)
                else TensorDataType.uint8
            )

        if per_channel:
            observer = torch.ao.quantization.observer.PerChannelMinMaxObserver
        else:
            observer = torch.ao.quantization.observer.MinMaxObserver

        if dtype is TensorDataType.int8:
            level_high = 127
            level_low = -128
        else:
            level_high = 255
            level_low = 0

        if quantizer_config.narrow_range:
            if level_low < 0:
                level_low += 1
            else:
                level_high -= 1

        if quantizer_config.mode == QuantizationScheme.SYMMETRIC:
            qscheme = torch.per_channel_symmetric if per_channel else torch.per_tensor_symmetric
        else:
            qscheme = torch.per_channel_affine if per_channel else torch.per_tensor_affine

        scale, zero_point = get_scale_zp_from_input_low_input_high(
            level_low, level_high, parameters.input_low.data, parameters.input_high.data
        )

        scale = scale.view(-1)
        zero_point = zero_point.view(-1)

        fakequantizer = FakeQuantize(
            observer=observer,
            quant_max=level_high,
            quant_min=level_low,
            dtype=torch.qint8 if dtype is TensorDataType.int8 else torch.quint8,
            qscheme=qscheme,
            eps=1e-16,
        )

        fakequantizer.scale = scale
        fakequantizer.zero_point = zero_point
        if per_channel:
            fakequantizer.ch_axis = FXMinMaxAlgoBackend._get_channel_axis(is_weight_quantizer)

        # Disable observer to save parameters
        fakequantizer.disable_observer()
        return fakequantizer

    @staticmethod
    def create_quantizer_insertion_command(
        nncf_graph: NNCFGraph,
        target_point: PTTargetPoint,
        quantizer_config: QuantizerConfig,
        parameters: FakeQuantizeParameters,
    ) -> FXApplyTransformationCommand:
        quantizer = FXMinMaxAlgoBackend._create_quantizer(
            quantizer_config, parameters, target_point.is_weight_target_point()
        )
        transformation = qdq_insertion_transformation_builder(quantizer, [target_point])
        return FXApplyTransformationCommand(transformation)

    @staticmethod
    def create_unified_scales_quantizers_insertion_commands(
        nncf_graph: NNCFGraph,
        target_points: list[PTTargetPoint],
        quantizer_config: QuantizerConfig,
        parameters: FakeQuantizeParameters,
    ) -> list[PTSharedFnInsertionCommand]:
        quantizer = FXMinMaxAlgoBackend._create_quantizer(
            quantizer_config, parameters, target_points[0].is_weight_target_point()
        )

        transformations = []
        for tp in target_points:
            transformation = qdq_insertion_transformation_builder(quantizer, [tp])
            transformations.append(FXApplyTransformationCommand(transformation))
        return transformations

    @staticmethod
    def get_ignored_metatypes(model_type: ModelType, device: TargetDevice) -> list[OperatorMetatype]:
        types = []
        if model_type == ModelType.TRANSFORMER:
            types = [
                om.PTAddMetatype,
                om.PTPowerMetatype,
                om.PTSubMetatype,
                om.PTAvgPool2dMetatype,
                om.PTAvgPool3dMetatype,
                om.PTMeanMetatype,
                om.PTSumMetatype,
                om.PTReduceL2,
                om.PTDivMetatype,
                om.PTMaxMetatype,
                om.PTSqueezeMetatype,
                om.PTLayerNormMetatype,
                om.PTModuleLayerNormMetatype,
                om.PTGroupNormMetatype,
                om.PTModuleGroupNormMetatype,
                # Batchnorm
                om.PTBatchNormMetatype,
                om.PTModuleBatchNormMetatype,
            ]
            if device != TargetDevice.CPU_SPR:
                types.append(om.PTMulMetatype)
        return types

    @staticmethod
    def get_ignored_names_by_layer_attributes(nncf_graph: NNCFGraph) -> set[str]:
        return set()

    def get_weight_nodes(self, nncf_graph: NNCFGraph, inference_nncf_graph: NNCFGraph) -> list[NNCFNode]:
        return get_weight_nodes(nncf_graph, inference_nncf_graph)

    def is_matmul_with_constant(self, node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
        return is_matmul_with_constant(node, nncf_graph)
