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
from typing import List, Optional, Tuple, TypeVar

import numpy as np

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationCommand
from nncf.common.tensor import NNCFTensor
from nncf.common.tensor_statistics.collectors import ReductionShape
from nncf.common.tensor_statistics.collectors import TensorStatisticCollectorBase
from nncf.common.utils.registry import Registry

TModel = TypeVar("TModel")
OutputType = TypeVar("OutputType")
ALGO_BACKENDS = Registry("algo_backends")


# pylint:disable=too-many-public-methods
class BiasCorrectionAlgoBackend(ABC):
    @property
    @abstractmethod
    def tensor_processor(self):
        """
        Returns backend-specific instance of the NNCFCollectorTensorProcessor.
        """

    @property
    @abstractmethod
    def quantizer_types(self):
        """
        Returns backend-specific list of the quantizer metatypes.
        """

    @staticmethod
    @abstractmethod
    def target_point(target_type: TargetType, target_node_name: str, port_id: int) -> TargetPoint:
        """
        Returns backend-specific target point.

        :param target_type: Type of the location that should be modified.
        :param target_node_name: Name of the located node.
        :param port_id: id of the port for the statistics disctribution.
        :return: Backend-specific TargetPoint.
        """

    @staticmethod
    @abstractmethod
    def create_bias_correction_command(node: NNCFNode, bias_value: np.ndarray) -> TransformationCommand:
        """
        Creates backend-specific command to update bias value.

        :param node: The node for which bias should be updated.
        :param bias_value: New value for the bias.
        :return: Backend-specific command to update bias value.
        """

    @staticmethod
    @abstractmethod
    def model_extraction_command(inputs: List[str], outputs: List[str]) -> TransformationCommand:
        """
        Returns backend-specific command to extract sub-model based on input & output names.

        :param inputs: List of the input names for sub-model beginning.
        :param outputs: List of the output names for sub-model end.
        :return: Backend-specific TransformationCommand for the model extraction.
        """

    @staticmethod
    @abstractmethod
    def output_insertion_command(nncf_graph: NNCFGraph, target_point: TargetPoint) -> TransformationCommand:
        """
        Returns backend-specific command that inserts output.

        :param nncf_graph: NNCFGraph instance.
        :param target_point: TargetPoint instance.
        :return: Backend-specific command that inserts output.
        """

    @staticmethod
    @abstractmethod
    def node_removing_command(target_point: TargetPoint) -> TransformationCommand:
        """
        Returns backend-specific command that removes node.

        :param target_point: TargetPoint instance.
        :return: Backend-specific command that remove node.
        """

    @staticmethod
    @abstractmethod
    def mean_statistic_collector(
        reduction_shape: ReductionShape,
        inplace: bool,
        num_samples: Optional[int] = None,
        window_size: Optional[int] = None,
    ) -> TensorStatisticCollectorBase:
        """
        Returns backend-specific mean statistic collector.

        :param reduction_shape: Channel axis for the statistics aggregation.
        :param inplace: Whether to calculate statistic inplace or not.
        :param num_samples: Maximum number of samples to collect.
        :param window_size: The maximum size of the samples queue.
        :return: Backend-specific TensorStatisticCollectorBase for the statistics calculation.
        """

    @staticmethod
    @abstractmethod
    def batch_statistic_collector(inplace: bool, num_samples: int = None) -> TensorStatisticCollectorBase:
        """
        Returns backend-specific batch statistic collector.

        :param inplace: Whether to calculate statistic inplace or not.
        :param num_samples: Maximum number of samples to collect.
        :return: Backend-specific TensorStatisticCollectorBase for the statistics calculation.
        """

    @staticmethod
    @abstractmethod
    def process_model_output(raw_data: OutputType, output_name: str) -> NNCFTensor:
        """
        Returns backend-specific processed output from the model.

        :param raw_data: Backend-specific output from the model.
        :param output_name: Name of the output layer or tensor name.
        :return: Processed output as NNCFTensor.
        """

    @staticmethod
    @abstractmethod
    def get_activation_port_ids_for_bias_node(node: NNCFNode) -> Tuple[int, int]:
        """
        Returns Input Port ID and Output Port ID corresponding to activation input and output edges for
        the node.
        Supports only nodes that could have bias value.

        :param node: Node of NNCFGraph with bias value.
        """

    @staticmethod
    @abstractmethod
    def get_bias_value(node: NNCFNode, model: TModel, nncf_graph: NNCFGraph) -> np.ndarray:
        """
        Returns bias value in the NumPy format of provided node.

        :param node: Node of NNCFGraph with bias value.
        :param model: Backend-specific model for the initializer finding.
        :param nncf_graph: NNCFGraph instance with the node.
        :return: Bias value in the NumPy format.
        """

    @staticmethod
    @abstractmethod
    def get_input_name(model: TModel, node_name: str) -> str:
        """
        Returns input tensor name for the specific node.

        :param model: Backend-specific model for the initializer finding.
        :param node_name: Name of the backend-specific node.
        :return: Input tensor name.
        """

    @staticmethod
    @abstractmethod
    def get_output_name(model: TModel, node_name: str) -> str:
        """
        Returns output tensor name for the specific node.

        :param model: Backend-specific model.
        :param node_name: Name of the backend-specific node.
        :return: Output tensor name.
        """

    @staticmethod
    @abstractmethod
    def is_quantized_weights(node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
        """
        Checks whether the node is quantized or not.

        :param node: NNCFNode to check.
        :param nncf_graph: NNCFGraph instance with the node.
        :return: boolean indicating whether the node has a quantized weights or not.
        """

    @staticmethod
    @abstractmethod
    def is_node_with_bias(node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
        """
        Checks whether the node has a bias or not.

        :param node: NNCFNode with the attributes.
        :param nncf_graph: NNCFGraph instance with the node.
        :return: Boolean indicating whether the node has a bias or not.
        """

    @staticmethod
    @abstractmethod
    def insert_null_biases(model: TModel) -> TModel:
        """
        This method finds and inserts zero biases for the layers that should have it.

        :param model: TModel instance.
        :return: TModel instance with zero biases
        """
