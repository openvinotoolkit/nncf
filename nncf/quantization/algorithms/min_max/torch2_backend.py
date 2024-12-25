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

from typing import Dict, List, Optional, Set, Tuple, Type, cast

import torch

import nncf
import nncf.tensor.functions as fns
import nncf.torch.graph.operator_metatypes as om
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.transformations.commands import Command
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationCommand
from nncf.common.hardware.config import HWConfig
from nncf.common.quantization.quantizer_propagation.structs import QuantizationTrait
from nncf.common.quantization.structs import QuantizationScheme as QuantizationMode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.experimental.common.tensor_statistics.collectors import REDUCERS_MAP
from nncf.experimental.common.tensor_statistics.collectors import TensorReducerBase
from nncf.experimental.torch2.commands import PT2InsertionCommand
from nncf.parameters import ModelType
from nncf.parameters import TargetDevice
from nncf.quantization.algorithms.min_max.backend import MinMaxAlgoBackend
from nncf.quantization.fake_quantize import FakeConvertParameters
from nncf.quantization.fake_quantize import FakeQuantizeParameters
from nncf.quantization.range_estimator import StatisticsType
from nncf.torch.graph.graph import PTNNCFGraph
from nncf.torch.graph.operator_metatypes import ELEMENTWISE_OPERATIONS
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.hardware.config import PTHWConfig
from nncf.torch.model_graph_manager import get_weight_channel_axes
from nncf.torch.model_graph_manager import get_weight_tensor_port_ids
from nncf.torch.quantization.default_quantization import DEFAULT_PT_QUANT_TRAIT_TO_OP_DICT
from nncf.torch.quantization.layers import QUANTIZATION_MODULES
from nncf.torch.quantization.layers import AsymmetricQuantizer
from nncf.torch.quantization.layers import BaseQuantizer
from nncf.torch.quantization.layers import PTQuantizerSpec
from nncf.torch.quantization.layers import SymmetricQuantizer
from nncf.torch.quantization.layers import get_scale_shape


class PT2MinMaxAlgoBackend(MinMaxAlgoBackend):
    @property
    def preserved_metatypes(self) -> List[Type[OperatorMetatype]]:
        return []

    @property
    def mat_mul_metatypes(self) -> List[Type[OperatorMetatype]]:
        return [om.PTLinearMetatype, om.PTMatMulMetatype, om.PTAddmmMetatype]

    @property
    def post_processing_metatypes(self) -> List[Type[OperatorMetatype]]:
        return []

    @property
    def shapeof_metatypes(self) -> List[Type[OperatorMetatype]]:
        return []

    @property
    def dropout_metatypes(self) -> List[Type[OperatorMetatype]]:
        return [om.PTDropoutMetatype]

    @property
    def read_variable_metatypes(self) -> List[Type[OperatorMetatype]]:
        return []

    @property
    def conv_metatypes(self) -> List[Type[OperatorMetatype]]:
        return [om.PTConv1dMetatype, om.PTConv2dMetatype, om.PTConv3dMetatype]

    @property
    def elementwise_metatypes(self) -> List[Type[OperatorMetatype]]:
        return cast(List[Type[OperatorMetatype]], ELEMENTWISE_OPERATIONS)

    @property
    def overflow_fix_metatypes(self) -> List[Type[OperatorMetatype]]:
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
    def add_metatypes(self) -> List[Type[OperatorMetatype]]:
        return [om.PTAddMetatype]

    @property
    def group_conv_metatypes(self) -> List[Type[OperatorMetatype]]:
        return self.conv_metatypes

    @property
    def scaled_dot_product_attention_metatypes(self) -> List[Type[OperatorMetatype]]:
        return [om.PTScaledDotProductAttentionMetatype]

    @property
    def scales_unification_map(self) -> Dict[Type[OperatorMetatype], List[Type[OperatorMetatype]]]:
        ret_map: Dict[Type[OperatorMetatype], List[Type[OperatorMetatype]]] = {
            om.PTCatMetatype: self.overflow_fix_metatypes,
        }
        return ret_map

    @property
    def hw_config(self) -> Type[HWConfig]:
        return PTHWConfig

    @property
    def quant_trait_op_dict(self) -> Dict[QuantizationTrait, List[Type[OperatorMetatype]]]:
        return cast(Dict[QuantizationTrait, List[Type[OperatorMetatype]]], DEFAULT_PT_QUANT_TRAIT_TO_OP_DICT)

    @property
    def reducer_map(self) -> Dict[StatisticsType, TensorReducerBase]:
        return cast(Dict[StatisticsType, TensorReducerBase], REDUCERS_MAP)

    @property
    def supports_inplace_statistics(self) -> bool:
        return False

    @staticmethod
    def get_start_nodes_for_activation_path_tracing(nncf_graph: NNCFGraph) -> List[NNCFNode]:
        if not isinstance(nncf_graph, PTNNCFGraph):
            raise nncf.InternalError(f"Only PTNNCFGraph is supported: {type(nncf_graph)}")
        return nncf_graph.get_nodes_with_missed_input_edges() + nncf_graph.get_input_nodes()

    @staticmethod
    def target_point(target_type: TargetType, target_node_name: str, port_id: int) -> PTTargetPoint:
        if target_type in (TargetType.POST_LAYER_OPERATION, TargetType.OPERATOR_POST_HOOK):
            port_id = None
        return PTTargetPoint(target_type, target_node_name, input_port_id=port_id)

    @staticmethod
    def create_convert_insertion_command(
        target_point: TargetPoint,
        parameters: FakeConvertParameters,
    ) -> TransformationCommand:
        raise nncf.InternalError("FakeConvert insertion not implemented in PyTorch backend!")

    @staticmethod
    def get_target_point_shape(nncf_graph: NNCFGraph, node: NNCFNode, target_point: TargetPoint) -> Tuple[int, ...]:
        if not isinstance(nncf_graph, PTNNCFGraph):
            raise nncf.InternalError(f"Only PTNNCFGraph is supported: {type(nncf_graph)}")
        if not isinstance(target_point, PTTargetPoint):
            raise nncf.InternalError(f"Only PTTargetPoint is supported: {type(target_point)}")
        return nncf_graph.get_input_shape_for_insertion_point(target_point)

    @staticmethod
    def get_weight_quantization_axes(node: NNCFNode, target_point: TargetPoint, ndims: int) -> Tuple[int, ...]:
        if not isinstance(target_point, PTTargetPoint):
            raise nncf.InternalError(f"Only PTTargetPoint is supported: {type(target_point)}")
        if target_point.input_port_id is None:
            raise nncf.InternalError("Weight target point must have input port id")
        return get_weight_channel_axes(node.metatype, ndims, target_point.input_port_id)

    @staticmethod
    def get_weight_tensor_port_ids(node: NNCFNode, graph: NNCFGraph) -> List[Optional[int]]:
        return cast(List[Optional[int]], get_weight_tensor_port_ids(node, graph))

    @staticmethod
    def get_weight_name(nncf_graph: NNCFGraph, target_point: TargetPoint) -> str:
        if not isinstance(target_point, PTTargetPoint):
            raise nncf.InternalError(f"Only PTTargetPoint is supported: {type(target_point)}")
        return target_point.target_node_name

    @staticmethod
    def should_quantize_weight(weight_name: str, quantized_weight_names: Set[str]) -> bool:
        # If the nodes share one weight tensor, we should have only one quantizer on that
        return weight_name not in quantized_weight_names

    @staticmethod
    def _get_input_scale_shape(nncf_graph: NNCFGraph, target_point: TargetPoint, per_channel: bool) -> Tuple[int, ...]:
        """
        Determines the scale shape for the input data at a given target point in the NNCF graph.

        :param nncf_graph: The graph representing the neural network for compression.
        :param target_point: The target point in the graph where the input scale shape is required.
        :param per_channel: Whether the scaling is per-channel or per-tensor.

        :return: Tuple[int, ...]: The shape of the scale to be applied to the input.
        """

        if not isinstance(nncf_graph, PTNNCFGraph):
            raise nncf.ValidationError(f"Only PTNNCFGraph is supported: {type(nncf_graph)}")
        if not isinstance(target_point, PTTargetPoint):
            raise nncf.ValidationError(f"Only PTTargetPoint is supported: {type(target_point)}")

        is_weights = target_point.is_weight_target_point()
        input_shape = nncf_graph.get_input_shape_for_insertion_point(target_point)
        if is_weights:
            if target_point.input_port_id is None:
                raise nncf.InternalError("Weight target point must have input port id")
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
        scale_shape: Tuple[int, ...],
        parameters: FakeQuantizeParameters,
        target_type: TargetType,
    ) -> BaseQuantizer:
        # TODO(AlexanderDokuchaev): remove cast after use Enum as parent class for QuantizationScheme
        mode = cast(str, quantizer_config.mode)
        quantizer_cls = QUANTIZATION_MODULES.get(mode)
        narrow_range = bool(target_type == TargetType.OPERATION_WITH_WEIGHTS and mode == QuantizationMode.SYMMETRIC)
        quantizer_spec = PTQuantizerSpec.from_config(
            quantizer_config,
            narrow_range=narrow_range,
            scale_shape=scale_shape,
            half_range=False,
            logarithm_scale=False,
            is_quantized_on_export=False,
            compression_lr_multiplier=None,
        )
        quantizer = cast(BaseQuantizer, quantizer_cls(quantizer_spec))

        # Fill it with minmax
        PT2MinMaxAlgoBackend._fill_quantizer_parameters(quantizer, parameters, quantizer_spec.scale_shape)
        return quantizer

    @staticmethod
    def _fill_quantizer_parameters(
        quantizer: BaseQuantizer, parameters: FakeQuantizeParameters, scale_shape: Tuple[int, ...]
    ) -> None:
        if isinstance(quantizer, AsymmetricQuantizer):
            input_low = cast(torch.Tensor, parameters.input_low.data)
            input_high = cast(torch.Tensor, parameters.input_high.data)
            # Subtract eps from the input_range to make quantizer parameters equal to
            # original parameters on the forward call.

            input_range = input_high - input_low - quantizer.eps

            quantizer.input_low.data = input_low.reshape(scale_shape)
            quantizer.input_range.data = input_range.reshape(scale_shape)
        elif isinstance(quantizer, SymmetricQuantizer):
            quantizer.signed = bool(fns.any(parameters.input_low < 0))
            # Subtract eps from the scale to make quantizer parameters equal to
            # original parameters on the forward call.
            input_high = cast(torch.Tensor, (parameters.input_high.data))
            scale = input_high - quantizer.eps
            quantizer.scale = torch.nn.Parameter(scale.reshape(scale_shape))
        else:
            raise nncf.ValidationError(f"Unsupported quantizer type: {type(quantizer)}")

    @staticmethod
    def create_quantizer_insertion_command(
        nncf_graph: NNCFGraph,
        target_point: TargetPoint,
        quantizer_config: QuantizerConfig,
        parameters: FakeQuantizeParameters,
    ) -> Command:
        if not isinstance(target_point, PTTargetPoint):
            raise nncf.InternalError(f"Only PTTargetPoint is supported: {type(target_point)}")
        scale_shape = PT2MinMaxAlgoBackend._get_input_scale_shape(
            nncf_graph, target_point, quantizer_config.per_channel
        )
        quantizer = PT2MinMaxAlgoBackend._create_quantizer(
            quantizer_config, scale_shape, parameters, target_point.target_type
        )
        return PT2InsertionCommand(target_points=[target_point], hook_module=quantizer)

    @staticmethod
    def create_unified_scales_quantizers_insertion_commands(
        nncf_graph: NNCFGraph,
        target_points: List[TargetPoint],
        quantizer_config: QuantizerConfig,
        parameters: FakeQuantizeParameters,
    ) -> List[Command]:
        if not isinstance(target_points, list) or not all(isinstance(tp, PTTargetPoint) for tp in target_points):
            raise nncf.InternalError("Only PTTargetPoint is supported")
        pt_target_points = cast(List[PTTargetPoint], target_points)
        scale_shape = PT2MinMaxAlgoBackend._get_input_scale_shape(
            nncf_graph, pt_target_points[0], quantizer_config.per_channel
        )

        quantizer = PT2MinMaxAlgoBackend._create_quantizer(
            quantizer_config, scale_shape, parameters, pt_target_points[0].target_type
        )

        return [PT2InsertionCommand(pt_target_points, quantizer)]

    @staticmethod
    def get_ignored_metatypes(model_type: ModelType, device: TargetDevice) -> List[Type[OperatorMetatype]]:
        types: List[Type[OperatorMetatype]] = []
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
                om.PTGroupNormMetatype,
                # Batchnorm
                om.PTBatchNormMetatype,
                # Сomparison operations
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
    def get_ignored_names_by_layer_attributes(nncf_graph: NNCFGraph) -> Set[str]:
        return set()

    def get_weight_nodes(self, nncf_graph: NNCFGraph) -> List[NNCFNode]:
        weight_nodes_candidates = [
            node
            for node in nncf_graph.get_all_nodes()
            if issubclass(node.metatype, om.PTOperatorMetatype) and node.metatype.weight_port_ids
        ]
        weight_nodes = []
        for node in weight_nodes_candidates:
            if node.metatype in self.mat_mul_metatypes and not self.is_matmul_with_constant(node, nncf_graph):
                continue
            weight_nodes.append(node)
        return weight_nodes

    def is_matmul_with_constant(self, node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
        return node.metatype in self.mat_mul_metatypes and len(get_weight_tensor_port_ids(node, nncf_graph)) > 0
