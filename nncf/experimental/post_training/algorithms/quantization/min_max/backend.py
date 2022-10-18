"""
 Copyright (c) 2022 Intel Corporation
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
from typing import Dict
from typing import TypeVar
from typing import Tuple
from typing import List

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
from nncf.common.utils.registry import Registry
from nncf.common.quantization.structs import QuantizerConfig
from nncf.experimental.post_training.algorithms.quantization.min_max.utils import QuantizerLayerParameters
from nncf.experimental.post_training.graph.model_transformer import StaticModelTransformerBase

ModelType = TypeVar('ModelType')
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
    def model_transformer(model: ModelType) -> StaticModelTransformerBase:
        """
        Returns backend-specific ModelTransformer instance.

        :param model: Backend-specific model to create ModelTransformer.
        :return: ModelTransformer instance.
        """

    @staticmethod
    @abstractmethod
    def target_point(target_type: TargetType, target_node_name: str, edge_name: str = None) -> TargetPoint:
        """
        Returns backend-specific target point.

        :param target_type: Type of the location that should be modified.
        :param target_node_name: Name of the located node.
        :param edge_name: Name of the tensor for the statistics disctribution.
        :return: Backend-specific TargetPoint.
        """

    @staticmethod
    @abstractmethod
    def quantizer_insertion_command(target_point: TargetPoint,
                                    parameters: QuantizerLayerParameters) -> TransformationCommand:
        """
        Returns backend-specific quantizer insertion command.

        :param target_point: Target location for the correction.
        :param parameters: QuantizerLayerParameters instance for the command.
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
    def get_initializer_value(model: ModelType, initializer_name: str) -> np.ndarray:
        """
        Returns initializer value in the NumPy format.

        :param model: Backend-specific model for the initializer finding.
        :param initializer_name: Name of the tensor/initializer to find in the model.
        :return: Initializer value in the NumPy format.
        """

    @staticmethod
    @abstractmethod
    def get_tensor_names(node: NNCFNode) -> Tuple[List[str], List[str]]:
        """
        Returns tuple of the lists with the input & output tensor names respectively.

        :param node: NNCFNode with the layer_attributes.
        :return: Tuple of the lists with the names.
        """

    @staticmethod
    @abstractmethod
    def get_weight_config(config: QuantizerConfig, model: ModelType) -> QuantizerConfig:
        """
        Returns backend-specific configuration based on the input model attributes.

        :param config: Base QuantizerConfig from the algo.
        :param model: Backend-specific model instance.
        :return: The updated QuantizerConfig.
        """
