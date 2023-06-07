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

from abc import ABC
from abc import abstractmethod
from typing import Dict, List, Optional, Set, TypeVar

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationCommand
from nncf.common.hardware.config import HWConfig
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.tensor_statistics.collectors import TensorStatisticCollectorBase
from nncf.common.tensor_statistics.statistics import MinMaxTensorStatistic
from nncf.common.utils.registry import Registry
from nncf.parameters import ModelType
from nncf.parameters import TargetDevice
from nncf.quantization.fake_quantize import FakeQuantizeParameters
from nncf.quantization.range_estimator import RangeEstimatorParameters
from nncf.scopes import IgnoredScope

TModel = TypeVar("TModel")
ALGO_BACKENDS = Registry("algo_backends")


class MinMaxAlgoBackend(ABC):
    @property
    @abstractmethod
    def mat_mul_metatype(self) -> OperatorMetatype:
        """
        Property for the backend-specific MatMul metatype.
        """

    @property
    @abstractmethod
    def post_processing_metatypes(self) -> List[OperatorMetatype]:
        """
        Property for the backend-specific post-processing metatypes (NonMaximumSupression, TopK, etc.).
        """

    @property
    @abstractmethod
    def shapeof_metatypes(self) -> List[OperatorMetatype]:
        """
        Property for the backend-specific ShapeOf metatypes.
        """

    @property
    @abstractmethod
    def conv_metatype(self) -> List[OperatorMetatype]:
        """
        Property for the backend-specific Convolution metatypes.
        """

    @property
    @abstractmethod
    def overflow_fix_metatypes(self) -> List[OperatorMetatype]:
        """
        Property for the backend-specific metatypes for which overflow_fix is applicable.
        """

    @property
    @abstractmethod
    def read_variable_metatypes(self) -> List[OperatorMetatype]:
        """
        Property for the backend-specific metatypes that also can be interpreted as inputs (ReadValue).
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
    def create_activation_quantizer_insertion_command(
        nncf_graph: NNCFGraph,
        target_point: TargetPoint,
        quantizer_config: QuantizerConfig,
        parameters: FakeQuantizeParameters,
    ) -> TransformationCommand:
        """
        Returns backend-specific quantizer insertion command.

        :param nncf_graph: NNCFGraph to get input/output shapes for the target point.
        :param target_point: Target location for the correction.
        :param quantizer_config: QuantizerConfig instance for the current layer.
        :param parameters: FakeQuantizeParameters to calculate activation quantization parameters.
        :return: Backend-specific TransformationCommand for the quantizer insertion operation.
        """

    @staticmethod
    @abstractmethod
    def create_weight_quantizer_insertion_command(
        nncf_graph: NNCFGraph,
        target_point: TargetPoint,
        quantizer_config: QuantizerConfig,
        parameters: FakeQuantizeParameters,
    ) -> TransformationCommand:
        """
        Returns backend-specific quantizer insertion command.

        :param nncf_graph: NNCFGraph to get input/output shapes for the target point.
        :param target_point: Target location for the correction.
        :param quantizer_config: QuantizerConfig instance for the current layer.
        :param parameters: FakeQuantizeParameters to calculate activation quantization parameters.
        :return: Backend-specific TransformationCommand for the quantizer insertion operation.
        """

    @staticmethod
    @abstractmethod
    def unify_statistics(statistics: List[MinMaxTensorStatistic]) -> MinMaxTensorStatistic:
        """
        Returns backend-specific unified statistics.

        :param statistics: List of MinMaxTensorStatistic instances.
        :return: Unified MinMaxTensorStatistic value.
        """

    @staticmethod
    @abstractmethod
    def get_statistic_collector(
        range_estimator_params: RangeEstimatorParameters,
        nncf_graph: NNCFGraph,
        target_point: TargetPoint,
        quantizer_config: QuantizerConfig,
        inplace: bool,
        num_samples: int = None,
    ) -> TensorStatisticCollectorBase:
        """
        Returns backend-specific statistic collector.

        :param range_estimator_params: Parameters that specify estimators types.
        :param nncf_graph: NNCFGraph to get input/output shapes for the target point.
        :param target_point: Target location for the correction.
        :param quantizer_config: QuantizerConfig instance for the current layer.
        :param inplace: Whether to calculate statistic inplace or not.
        :param num_samples: Maximum number of samples to collect.
        :return: Backend-specific TensorStatisticCollectorBase for the statistics calculation.
        """

    @staticmethod
    @abstractmethod
    def get_weight_tensor_port_ids(node: NNCFNode) -> List[Optional[int]]:
        """
        Returns node's input port indices with weight tensors.

        :param node: NNCFNode to find its weight input port indices.
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
    def get_ignored_scope(model_type: ModelType, device: TargetDevice) -> IgnoredScope:
        """
        Returns ignores scope based on a model type and device parameters.

        :param model_type: Model type parameter.
        :param device: Target device.
        :return: Instance of ignored scope.
        """

    @staticmethod
    @abstractmethod
    def get_weight_nodes(nncf_graph: NNCFGraph) -> List[NNCFNode]:
        """
        Returns nodes that have weights.

        :param nncf_graph: Instance of NNCFGraph.
        :return: All nodes with weights.
        """
