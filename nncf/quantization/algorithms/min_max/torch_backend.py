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

from typing import Any, Optional, Union

import torch

import nncf
import nncf.torch.graph.operator_metatypes as om
from nncf.common.graph.definitions import NNCFGraphNodeType
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationCommand
from nncf.common.hardware.config import HWConfig
from nncf.common.quantization.quantizer_propagation.structs import QuantizationTrait
from nncf.common.quantization.structs import QuantizerConfig
from nncf.experimental.common.tensor_statistics.collectors import REDUCERS_MAP
from nncf.experimental.common.tensor_statistics.collectors import TensorReducerBase
from nncf.parameters import ModelType
from nncf.parameters import TargetDevice
from nncf.quantization.algorithms.min_max.backend import MinMaxAlgoBackend
from nncf.quantization.fake_quantize import FakeConvertParameters
from nncf.quantization.fake_quantize import FakeQuantizeParameters
from nncf.quantization.range_estimator import StatisticsType
from nncf.torch.function_hook.commands import PT2InsertionCommand
from nncf.torch.graph.graph import PTNNCFGraph
from nncf.torch.graph.graph import PTTargetPoint
from nncf.torch.graph.operator_metatypes import ELEMENTWISE_OPERATIONS
from nncf.torch.graph.operator_metatypes import MATMUL_METATYPES
from nncf.torch.graph.transformations.commands import PTInsertionCommand
from nncf.torch.graph.transformations.commands import PTSharedFnInsertionCommand
from nncf.torch.hardware.config import PTHWConfig
from nncf.torch.model_graph_manager import get_weight_channel_axes
from nncf.torch.model_graph_manager import get_weight_nodes
from nncf.torch.model_graph_manager import get_weight_tensor_port_ids
from nncf.torch.model_graph_manager import is_matmul_with_constant
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.quantization.default_quantization import DEFAULT_PT_QUANT_TRAIT_TO_OP_DICT
from nncf.torch.quantization.layers import QUANTIZATION_MODULES
from nncf.torch.quantization.layers import AsymmetricQuantizer
from nncf.torch.quantization.layers import BaseQuantizer
from nncf.torch.quantization.layers import PTQuantizerSpec
from nncf.torch.quantization.layers import get_scale_shape


class PTMinMaxAlgoBackend(MinMaxAlgoBackend):
    @property
    def preserved_metatypes(self) -> list[OperatorMetatype]:
        return []

    TARGET_TYPE_TO_PT_INS_TYPE_MAP = {
        TargetType.PRE_LAYER_OPERATION: TargetType.OPERATOR_PRE_HOOK,
        TargetType.POST_LAYER_OPERATION: TargetType.OPERATOR_POST_HOOK,
    }

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
        return (
            nncf_graph.get_nodes_with_missed_input_edges()
            + nncf_graph.get_input_nodes()
            + nncf_graph.get_nodes_by_metatypes(
                DEFAULT_PT_QUANT_TRAIT_TO_OP_DICT[QuantizationTrait.OUTPUT_QUANTIZATION_AS_WEIGHTS]
            )
        )

    @staticmethod
    def target_point(target_type: TargetType, target_node_name: str, port_id: int) -> PTTargetPoint:
        input_port_id: Optional[int] = port_id
        if NNCFGraphNodeType.INPUT_NODE in target_node_name or target_type == TargetType.POST_LAYER_OPERATION:
            input_port_id = None
        return PTTargetPoint(target_type, target_node_name, input_port_id=input_port_id)

    @staticmethod
    def create_convert_insertion_command(
        target_point: PTTargetPoint,
        parameters: FakeConvertParameters,
        extra_params: dict[str, Any],
    ) -> TransformationCommand:
        msg = "FakeConvert insertion not implemented in PyTorch backend!"
        raise nncf.InternalError(msg)

    @staticmethod
    def get_target_point_shape(nncf_graph: PTNNCFGraph, node: NNCFNode, target_point: PTTargetPoint) -> tuple[int, ...]:
        return nncf_graph.get_input_shape_for_insertion_point(target_point)

    @staticmethod
    def get_weight_quantization_axes(node: NNCFNode, target_point: PTTargetPoint, ndims: int) -> tuple[int]:
        return get_weight_channel_axes(node.metatype, ndims, target_point.input_port_id)

    @staticmethod
    def get_weight_tensor_port_ids(node: NNCFNode, graph: NNCFGraph) -> list[Optional[int]]:
        return get_weight_tensor_port_ids(node, graph)

    @staticmethod
    def get_weight_name(nncf_graph: NNCFGraph, target_point: PTTargetPoint) -> str:
        return nncf_graph.get_node_by_name(target_point.target_node_name).layer_name

    @staticmethod
    def should_quantize_weight(weight_name: str, quantized_weight_names: set[str]) -> bool:
        # If the nodes share one weight tensor, we should have only one quantizer on that
        return weight_name not in quantized_weight_names

    @staticmethod
    def get_weight_config(config: QuantizerConfig, model: NNCFNetwork) -> QuantizerConfig:
        return config

    @staticmethod
    def _get_input_scale_shape(
        nncf_graph: PTNNCFGraph, target_point: PTTargetPoint, per_channel: bool
    ) -> tuple[int, ...]:
        """
        Determines the scale shape for the input data at a given target point in the NNCF graph.

        :param nncf_graph: The graph representing the neural network for compression.
        :param target_point: The target point in the graph where the input scale shape is required.
        :param per_channel: Whether the scaling is per-channel or per-tensor.

        :return: Tuple[int, ...]: The shape of the scale to be applied to the input.
        """
        is_weights = target_point.is_weight_target_point()
        input_shape = nncf_graph.get_input_shape_for_insertion_point(target_point)

        if is_weights:
            node_with_weight = nncf_graph.get_node_by_name(target_point.target_node_name)
            channel_axes = get_weight_channel_axes(
                node_with_weight.metatype, len(input_shape), target_point.input_port_id
            )
        else:
            channel_axes = (1,)  # channel dim for activations

        if len(channel_axes):
            scale_shape = tuple(
                get_scale_shape(
                    input_shape, is_weights=is_weights, per_channel=per_channel, channel_idx=channel_axes[0]
                )
            )
        else:
            # For cases where weights are vectors that should be quantized as per-tensor
            scale_shape = (1,)

        return scale_shape

    @staticmethod
    def _create_quantizer(
        quantizer_config: QuantizerConfig,
        scale_shape: tuple,
        parameters: FakeQuantizeParameters,
        target_type: TargetType,
    ) -> BaseQuantizer:
        mode = quantizer_config.mode
        quantizer_cls = QUANTIZATION_MODULES.get(mode)
        quantizer_spec = PTQuantizerSpec.from_config(
            quantizer_config,
            narrow_range=quantizer_config.narrow_range,
            scale_shape=scale_shape,
            half_range=False,
            logarithm_scale=False,
            is_quantized_on_export=False,
            compression_lr_multiplier=None,
        )
        quantizer = quantizer_cls(quantizer_spec)

        # Fill it with minmax
        PTMinMaxAlgoBackend._fill_quantizer_parameters(quantizer, parameters, quantizer_spec.scale_shape)
        return quantizer

    @staticmethod
    def _fill_quantizer_parameters(
        quantizer: BaseQuantizer, parameters: FakeQuantizeParameters, scale_shape: tuple[int, ...]
    ) -> None:
        if isinstance(quantizer, AsymmetricQuantizer):
            quantizer.input_low = torch.nn.Parameter(parameters.input_low.data.reshape(scale_shape))
            input_range = parameters.input_high - parameters.input_low
            # Subtract eps from the input_range to make quantizer parameters equal to
            # original parameters on the forward call.
            quantizer.input_range = torch.nn.Parameter((input_range.data - quantizer.eps).reshape(scale_shape))
        else:
            quantizer.signed = bool(torch.any(parameters.input_low.data < 0))
            # Subtract eps from the scale to make quantizer parameters equal to
            # original parameters on the forward call.
            quantizer.scale = torch.nn.Parameter((parameters.input_high.data - quantizer.eps).reshape(scale_shape))

    @staticmethod
    def create_quantizer_insertion_command(
        nncf_graph: NNCFGraph,
        target_point: PTTargetPoint,
        quantizer_config: QuantizerConfig,
        parameters: FakeQuantizeParameters,
    ) -> Union[PTInsertionCommand, PTSharedFnInsertionCommand]:
        scale_shape = PTMinMaxAlgoBackend._get_input_scale_shape(nncf_graph, target_point, quantizer_config.per_channel)

        quantizer = PTMinMaxAlgoBackend._create_quantizer(
            quantizer_config, scale_shape, parameters, target_point.target_type
        )
        return PT2InsertionCommand(target_points=[target_point], hook_module=quantizer)

    @staticmethod
    def create_unified_scales_quantizers_insertion_commands(
        nncf_graph: NNCFGraph,
        target_points: list[PTTargetPoint],
        quantizer_config: QuantizerConfig,
        parameters: FakeQuantizeParameters,
        extra_params: dict[str, Any],
    ) -> list[PTSharedFnInsertionCommand]:
        scale_shape = PTMinMaxAlgoBackend._get_input_scale_shape(
            nncf_graph, target_points[0], quantizer_config.per_channel
        )

        quantizer = PTMinMaxAlgoBackend._create_quantizer(
            quantizer_config, scale_shape, parameters, target_points[0].target_type
        )
        return [PT2InsertionCommand(target_points=target_points, hook_module=quantizer)]

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
                # Comparison operations
                om.PTGreaterEqualMetatype,
                om.PTGreaterMetatype,
                om.PTLessEqualMetatype,
                om.PTLessMetatype,
                om.PTNotEqualMetatype,
                om.PTEqualsMetatype,
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
