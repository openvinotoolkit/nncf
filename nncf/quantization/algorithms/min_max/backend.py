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

from abc import ABC
from abc import abstractmethod
from typing import Dict, List, Optional, Set, Tuple, TypeVar

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationCommand
from nncf.common.hardware.config import HWConfig
from nncf.common.quantization.structs import QuantizerConfig
from nncf.experimental.common.tensor_statistics.collectors import TensorReducerBase
from nncf.parameters import ModelType
from nncf.parameters import TargetDevice
from nncf.quantization.fake_quantize import FakeConvertParameters
from nncf.quantization.fake_quantize import FakeQuantizeParameters
from nncf.quantization.range_estimator import StatisticsType

TModel = TypeVar("TModel")


class MinMaxAlgoBackend(ABC):
    @property
    @abstractmethod
    def preserved_metatypes(self) -> List[OperatorMetatype]:
        """
        Property for backend-specific metatypes that require preserving float subgraphs
        when removing the ShapeOf subgraph.
        """

    @property
    @abstractmethod
    def mat_mul_metatypes(self) -> List[OperatorMetatype]:
        """
        Property for the backend-specific MatMul metatypes.
        """

    @property
    @abstractmethod
    def post_processing_metatypes(self) -> List[OperatorMetatype]:
        """
        Property for the backend-specific post-processing metatypes (NonMaximumSupression, TopK, etc.).
        """

    @property
    @abstractmethod
    def conv_metatypes(self) -> List[OperatorMetatype]:
        """
        Property for the backend-specific Convolution metatypes.
        """

    @property
    @abstractmethod
    def shapeof_metatypes(self) -> List[OperatorMetatype]:
        """
        Property for the backend-specific ShapeOf metatypes.
        """

    @property
    @abstractmethod
    def dropout_metatypes(self) -> List[OperatorMetatype]:
        """
        Property for the backend-specific Dropout metatypes.
        """

    @property
    @abstractmethod
    def elementwise_metatypes(self) -> List[OperatorMetatype]:
        """
        Property for the backend-specific Elementwises metatypes.
        """

    @property
    @abstractmethod
    def overflow_fix_metatypes(self) -> List[OperatorMetatype]:
        """
        Property for the backend-specific metatypes for which overflow_fix is applicable.
        """

    @property
    @abstractmethod
    def add_metatypes(self) -> List[OperatorMetatype]:
        """
        Property for the backend-specific metatypes that also can be interpreted as Add layer.
        """

    @property
    @abstractmethod
    def group_conv_metatypes(self) -> List[OperatorMetatype]:
        """
        Property for the backend-specific Grouped Convolution metatypes.
        """

    @property
    @abstractmethod
    def scaled_dot_product_attention_metatypes(self) -> List[OperatorMetatype]:
        """
        Property for the backend-specific Scaled Dot Product Attention metatypes.
        """

    @property
    @abstractmethod
    def scales_unification_map(self) -> Dict[OperatorMetatype, OperatorMetatype]:
        """
        Property for the backend-specific metatypes that produces quantizers that might be unified.
        """

    @property
    @abstractmethod
    def hw_config(self) -> HWConfig:
        """
        Property for the hardware backend-specific configuration.
        """

    @property
    @abstractmethod
    def quant_trait_op_dict(self) -> Dict[int, OperatorMetatype]:
        """
        Property for the backend-specific dictionary that contains QuantizationTrait-specific metatypes.
        """

    @property
    @abstractmethod
    def reducer_map(self) -> Dict[StatisticsType, TensorReducerBase]:
        """
        Property for the backend-specific dictionary that contains backend-specific tensor reducers.
        """

    @property
    @abstractmethod
    def supports_inplace_statistics(self) -> bool:
        """
        Property for the backend-specific flag that specifies whether the backend supports inplace statistics.
        """

    @staticmethod
    @abstractmethod
    def target_point(target_type: TargetType, target_node_name: str, port_id: int) -> TargetPoint:
        """
        Returns backend-specific target point.

        :param target_type: Type of the location that should be modified.
        :param target_node_name: Name of the located node.
        :param port_id: Port ID of the tensor for the statistics distribution.
        :return: Backend-specific TargetPoint.
        """

    @staticmethod
    @abstractmethod
    def create_quantizer_insertion_command(
        nncf_graph: NNCFGraph,
        target_point: TargetPoint,
        quantizer_config: QuantizerConfig,
        parameters: FakeQuantizeParameters,
    ) -> TransformationCommand:
        """
        Returns backend-specific quantizer insertion command.

        :param nncf_graph: NNCFGraph to get input/output shapes for the target point.
        :param target_point: Target location for the quantizer insertion.
        :param quantizer_config: QuantizerConfig instance for the current layer.
        :param parameters: FakeQuantizeParameters to calculate activation quantization parameters.
        :return: Backend-specific TransformationCommand for the quantizer insertion operation.
        """

    @staticmethod
    @abstractmethod
    def create_unified_scales_quantizers_insertion_commands(
        nncf_graph: NNCFGraph,
        target_points: List[TargetPoint],
        quantizer_config: QuantizerConfig,
        parameters: FakeQuantizeParameters,
    ) -> List[TransformationCommand]:
        """
        Returns backend-specific unified scales quantizers insertion commands.

        :param nncf_graph: NNCFGraph to get input/output shapes for the target point.
        :param target_points: List of target locations for the quantizers insertion.
        :param quantizer_config: QuantizerConfig instance for the current layer.
        :param parameters: FakeQuantizeParameters to calculate activation quantization parameters.
        :return: List of backend-specific TransformationCommands
            for the quantizers with unified scales insertion operations.
        """

    @staticmethod
    @abstractmethod
    def create_convert_insertion_command(
        target_point: TargetPoint,
        parameters: FakeConvertParameters,
    ) -> TransformationCommand:
        """
        Returns backend-specific convert insertion command.

        :param target_point: Target location for the correction.
        :param parameters: FakeConvertParameters to calculate activation quantization parameters.
        :return: Backend-specific TransformationCommand for the quantizer insertion operation.
        """

    @staticmethod
    @abstractmethod
    def get_start_nodes_for_activation_path_tracing(nncf_graph: NNCFGraph) -> List[NNCFNode]:
        """
        Returns a list of NNCFNodes to use as start nodes for activation path tracing.

        :param nncf_graph: NNCFGraph to get the start nodes.
        :return: List of NNCFNodes to use as start nodes for activation path tracing.

        """

    @staticmethod
    @abstractmethod
    def get_target_point_shape(nncf_graph: NNCFGraph, node: NNCFNode, target_point: TargetPoint) -> Tuple[int, ...]:
        """
        Returns shape of a target point tensor.

        :param nncf_graph: NNCFGraph instance.
        :param node: NNCFNode.
        :param target_point: Target point of which tensor shape is seeked.
        :return: Shape of target point tensor.
        """

    @staticmethod
    @abstractmethod
    def get_weight_quantization_axes(node: NNCFNode, target_point: TargetPoint, ndims: int) -> Tuple[int, ...]:
        """
        Returns axes for per-channel quantization of weights of the node placed on a input port_id.

        :param node: Quantized node with the weight.
        :param target_point: Corresponding target point.
        :param ndims: Number of dimensions of weight.
        :return: Axes for per-channel quantization of weights.
        """

    @staticmethod
    @abstractmethod
    def get_weight_tensor_port_ids(node: NNCFNode, graph: NNCFGraph) -> List[Optional[int]]:
        """
        Returns node's input port indices with weight tensors.

        :param node: NNCFNode to find its weight input port indices.
        :param graph: NNCFGraph instance.
        :return: Weights input port indices.
        """

    @staticmethod
    def get_weight_name(nncf_graph: NNCFGraph, target_point: TargetPoint) -> str:
        """
        Returns node's weight name corresponding to port ID.

        :param nncf_graph: NNCFGraph instance.
        :param target_point: The TargetPoint instance that contains layer's information.
        :return: Weight name.
        """

    @staticmethod
    def should_quantize_weight(weight_name: str, quantized_weight_names: Set[str]) -> bool:
        """
        Return True if weight should be quantized.

        :param weight_name: Weight name.
        :param quantized_weight_names: Set containing already quantized weight names.
        :return: A boolean value specifying whether a weight should be quantized.
        """

    @staticmethod
    @abstractmethod
    def get_ignored_metatypes(model_type: ModelType, device: TargetDevice) -> List[OperatorMetatype]:
        """
        Returns ignored metatypes based on a model type and device parameters.

        :param model_type: Model type parameter.
        :param device: Target device.
        :return: List of ignored metatypes.
        """

    @staticmethod
    @abstractmethod
    def get_ignored_names_by_layer_attributes(nncf_graph: NNCFGraph) -> Set[str]:
        """
        Returns names of ignored nodes based on layer_attributes.

        :param nncf_graph: NNCFGraph instance.
        :return: List of ignored names.
        """

    @abstractmethod
    def get_weight_nodes(self, nncf_graph: NNCFGraph) -> List[NNCFNode]:
        """
        Returns nodes that have weights.

        :param nncf_graph: Instance of NNCFGraph.
        :return: All nodes with weights.
        """

    @abstractmethod
    def is_matmul_with_constant(self, node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
        """
        Returns true if given nncf matmul node is a matmul with a constant, False otherwise.

        :param Node: Instance of NNCFNode.
        :param nncf_graph: Instance of NNCFGraph.
        :return: True if given nncf matmul node is a matmul with a constant, False otherwise.
        """
