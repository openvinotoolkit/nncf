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
from typing import List
from typing import TypeVar

import numpy as np
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.commands import TransformationCommand
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.tensor import NNCFTensor
from nncf.common.tensor import TensorType
from nncf.common.tensor_statistics.collectors import TensorStatisticCollectorBase
from nncf.common.tensor_statistics.collectors import ReductionShape
from nncf.common.utils.registry import Registry
from nncf.experimental.post_training.graph.model_transformer import StaticModelTransformerBase

ModelType = TypeVar('ModelType')
ALGO_BACKENDS = Registry('algo_backends')


class FBCAlgoBackend(ABC):

    @property
    @abstractmethod
    def operation_metatypes(self):
        """
        Property for the backend-specific metatypes
        """

    @property
    @abstractmethod
    def layers_with_bias_metatypes(self):
        """
        Property for the backend-specific metatypes with bias
        """

    @property
    @abstractmethod
    def channel_axis_by_types(self):
        """
        Property for the backend-specific info about channels placement in the layout
        """

    @property
    @abstractmethod
    def tensor_processor(self):
        """
        Returns backend-specific instance of the NNCFCollectorTensorProcessor
        """
    
    @staticmethod
    @abstractmethod
    def model_transformer(model: ModelType) -> StaticModelTransformerBase:
        """
        Returns backend-specific ModelTransformer instance

        :param model: backend-specific model to create ModelTransformer
        :return: ModelTransformer instance
        """

    @staticmethod
    @abstractmethod
    def target_point(target_type: TargetType, target_node_name: str, edge_name: str = None) -> TargetPoint:
        """
        Returns backend-specific target point

        :param target_type: type of the location that should be modified
        :param target_node_name: the name of the located node
        :param edge_name: name of the tensor for the statistics disctribution
        :return: backend-specific TargetPoint
        """

    @staticmethod
    @abstractmethod
    def bias_correction_command(target_point: TargetPoint,
                                bias_shift: np.ndarray,
                                threshold: float) -> TransformationCommand:
        """
        Returns backend-specific bias correction command

        :param target_point: target location for the correction
        :param bias_shift: shift value for the bias
        :param threshold: parametrized threshold for the shift magnitude comparison
        :return: backend-specific TransformationCommand for the bias correction
        """

    @staticmethod
    @abstractmethod
    def model_extraction_command(inputs: List[str], outputs: List[str]) -> TransformationCommand:
        """
        Returns backend-specific bias correction

        :param inputs: list of the input names for sub-model beggining
        :param outputs: list of the output names for sub-model end
        :return: backend-specific TransformationCommand for the model extraction
        """

    @staticmethod
    @abstractmethod
    def mean_statistic_collector(reduction_shape: ReductionShape,
                                 num_samples: int = None,
                                 window_size: int = None) -> TensorStatisticCollectorBase:
        """
        Returns backend-specific mean statistic collector

        :param reduction_shape: channel axes for the statistics aggregation
        :param num_samples: maximum number of samples to collect.
        :param window_size:
        :return: backend-specific TensorStatisticCollectorBase for the statistics calculation
        """

    @staticmethod
    @abstractmethod
    def nncf_tensor(tensor: TensorType) -> NNCFTensor:
        """
        Returns backend-specific NNCFTensor

        :param tensor: tensor data for the wrapping
        :return: NNCFTensor
        """
