"""
 Copyright (c) 2023 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from abc import ABC
from abc import abstractmethod
from typing import Dict, TypeVar, Tuple, List

import numpy as np
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.patterns import HWFusedPatterns
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationCommand
from nncf.common.hardware.config import HWConfig
from nncf.common.tensor_statistics.collectors import ReductionShape
from nncf.common.tensor_statistics.collectors import TensorStatisticCollectorBase
from nncf.common.tensor_statistics.statistics import MinMaxTensorStatistic
from nncf.common.utils.registry import Registry
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.graph.model_transformer import ModelTransformer

TModel = TypeVar('TModel')
ALGO_BACKENDS = Registry('algo_backends')


class MinMaxAlgoBackend(ABC):

    @property
    @abstractmethod
    def layers_with_weights_metatypes(self) -> List[OperatorMetatype]:
        """
        Property for the backend-specific metatypes with weights.
        """

    @property
    @abstractmethod
    def post_processing_metatypes(self) -> List[OperatorMetatype]:
        """
        Property for the backend-specific post-processing metatypes (NonMaximumSupression, TopK, etc.).
        """

    @property
    @abstractmethod
    def hw_fused_patterns(self) -> HWFusedPatterns:
        """
        Property for the hardware & backend-specific layers patterns.
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
    def model_transformer(model: TModel) -> ModelTransformer:
        """
        Returns backend-specific ModelTransformer instance.

        :param model: Backend-specific model to create ModelTransformer.
        :return: ModelTransformer instance.
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
    def create_activation_quantizer_insertion_command(target_point: TargetPoint,
                                                      quantizer_config: QuantizerConfig,
                                                      statistics: MinMaxTensorStatistic) -> TransformationCommand:
        """
        Returns backend-specific quantizer insertion command.

        :param target_point: Target location for the correction.
        :param quantizer_config: QuantizerConfig instance for the current layer.
        :param statistics: MinMaxTensorStatistic to calculate activation quantization parameters.
        :return: Backend-specific TransformationCommand for the quantizer insertion operation.
        """

    @staticmethod
    @abstractmethod
    def create_weight_quantizer_insertion_command(target_point: TargetPoint,
                                                  quantizer_config: QuantizerConfig,
                                                  weight_tensor: np.ndarray,
                                                  node: NNCFNode) -> TransformationCommand:
        """
        Returns backend-specific quantizer insertion command.

        :param target_point: Target location for the correction.
        :param quantizer_config: QuantizerConfig instance for the current layer.
        :param weight_tensor: weight tensor to calculate weight quantization parameters.
        :param node: NNCFNode with the attributes.
        :return: Backend-specific TransformationCommand for the quantizer insertion operation.
        """

    @staticmethod
    @abstractmethod
    def minmax_statistic_collector(use_abs_max: bool,
                                   reduction_shape: ReductionShape,
                                   num_samples: int = None) -> TensorStatisticCollectorBase:
        """
        Returns backend-specific min max statistic collector.

        :param use_abs_max: Whether to use absolute maximum value or not.
        :param reduction_shape: Channel axes for the statistics aggregation.
        :param num_samples: Maximum number of samples to collect.
        :return: Backend-specific TensorStatisticCollectorBase for the statistics calculation.
        """

    @staticmethod
    @abstractmethod
    def mean_minmax_statistic_collector(use_per_sample_stats: bool,
                                        use_abs_max: bool,
                                        reduction_shape: ReductionShape,
                                        num_samples: int = None,
                                        window_size: int = None) -> TensorStatisticCollectorBase:
        """
        Returns backend-specific min max statistic collector.

        :param use_abs_max: Whether to use absolute maximum value or not.
        :param reduction_shape: Channel axes for the statistics aggregation.
        :param num_samples: Maximum number of samples to collect.
        :param window_size: The maximum size of the samples queue.
        :return: Backend-specific TensorStatisticCollectorBase for the statistics calculation.
        """

    @staticmethod
    @abstractmethod
    def get_weight_tensor(model: TModel, target_point: TargetPoint) -> Tuple[str, np.ndarray]:
        """
        Returns node's weight tensor name and its value.

        :param model: Backend-specific model for the initializer finding.
        :param target_point: Backend-specific TargetPoint to find its weight.
        :return: Weight tensor name and its value.
        """

    @staticmethod
    def get_weight_tensor_port_id(model: TModel, node: NNCFNode) -> int:
        """
        Returns node's weight tensor input port ID.

        :param node: NNCFNode to find its weight input port ID.
        :return: The input port ID of the weight.
        """

    @staticmethod
    @abstractmethod
    def get_weight_config(config: QuantizerConfig, model: TModel) -> QuantizerConfig:
        """
        Returns backend-specific configuration based on the input model attributes.

        :param config: Base QuantizerConfig from the algo.
        :param model: Backend-specific model instance.
        :return: The updated QuantizerConfig.
        """
