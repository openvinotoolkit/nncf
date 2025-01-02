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
from typing import Optional, Set, Tuple, TypeVar, Union

import numpy as np

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationCommand
from nncf.common.tensor import NNCFTensor
from nncf.common.tensor_statistics.collectors import TensorStatisticCollectorBase
from nncf.tensor import Tensor

TModel = TypeVar("TModel")
OutputType = TypeVar("OutputType")


class BiasCorrectionAlgoBackend(ABC):
    @staticmethod
    @abstractmethod
    def target_point(target_type: TargetType, target_node_name: str, port_id: int) -> TargetPoint:
        """
        Returns backend-specific target point.

        :param target_type: Type of the location that should be modified.
        :param target_node_name: Name of the located node.
        :param port_id: id of the port for the statistics distribution.
        :return: Backend-specific TargetPoint.
        """

    @staticmethod
    @abstractmethod
    def create_bias_correction_command(node: NNCFNode, bias_value: Tensor) -> TransformationCommand:
        """
        Creates backend-specific command to update bias value.

        :param node: The node for which bias should be updated.
        :param bias_value: New value for the bias.
        :return: Backend-specific command to update bias value.
        """

    @staticmethod
    @abstractmethod
    def model_extraction_command(
        input_ids: Set[Tuple[str, int]], output_ids: Set[Tuple[str, int]]
    ) -> TransformationCommand:
        """
        Returns backend-specific command to extract sub-model based on input & output names.

        :param input_ids: Set of the input IDs: pairs of node names and correspondent input port ids.
            Each pair denotes the sub-graph beginning.
        :param output_ids: Set of the output IDs: pairs of node names and correspondent output port ids.
            Each pair denotes the sub-graph ending.
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
    def mean_statistic_collector(
        channel_axis: int,
        inplace: bool,
        num_samples: Optional[int] = None,
        window_size: Optional[int] = None,
    ) -> TensorStatisticCollectorBase:
        """
        Returns backend-specific mean statistic collector.

        :param channel_axis: Channel axis for the statistics aggregation.
        :param inplace: Whether to calculate statistic inplace or not.
        :param num_samples: Maximum number of samples to collect.
        :param window_size: The maximum size of the samples queue.
        :return: Backend-specific TensorStatisticCollectorBase for the statistics calculation.
        """

    @staticmethod
    @abstractmethod
    def raw_statistic_collector(num_samples: Optional[int] = None) -> TensorStatisticCollectorBase:
        """
        Returns backend-specific raw statistic collector.
        This statistic collector is used for raw data calculation, without aggregating.

        :param num_samples: Maximum number of samples to collect.
        :return: Backend-specific TensorStatisticCollectorBase for the statistics calculation.
        """

    @staticmethod
    @abstractmethod
    def process_model_output(raw_data: OutputType, output_name: Union[str, int]) -> NNCFTensor:
        """
        Returns backend-specific processed output from the model.

        :param raw_data: Backend-specific output from the model.
        :param output_name: Name of the output layer or tensor name.
        :return: Processed output as NNCFTensor.
        """

    @staticmethod
    @abstractmethod
    def get_activation_port_id(node: NNCFNode, nncf_graph: NNCFGraph) -> int:
        """
        Returns input port id corresponding to activation input edge for
        the node.
        Supports only nodes that could have bias value.

        :param node: Node of NNCFGraph with bias value.
        :param nncf_graph: NNCFGraph instance with the node.
        :return: target input port id.
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
    def get_input_name(model: TModel, node_name: str, input_port_id: int) -> str:
        """
        Returns input tensor name for the specific node.

        :param model: Backend-specific model for the initializer finding.
        :param node_name: Name of the backend-specific node.
        :param input_port_id: Port Id for input.
        :return: Input tensor name.
        """

    @staticmethod
    @abstractmethod
    def get_output_name(model: TModel, node_name: str, output_port_id: int) -> str:
        """
        Returns output tensor name for the specific node.

        :param model: Backend-specific model.
        :param node_name: Name of the backend-specific node.
        :param output_port_id: Port Id for output.
        :return: Output tensor name.
        """

    @staticmethod
    @abstractmethod
    def is_quantized_weights(node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
        """
        Checks whether the node is quantized or not.

        :param node: NNCFNode to check.
        :param nncf_graph: NNCFGraph instance with the node.
        :return: Boolean indicating whether the node has a quantized weights or not.
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
    def remove_fq_from_inputs(model: TModel, nncf_graph: NNCFGraph) -> TModel:
        """
        This method removes the activation Fake Quantize nodes (or Quantize-Dequantize pairs) from the model.
        It's needed for the further bias shift calculation that relates on quantized weights.

        :param model: TModel instance.
        :param nncf_graph: NNCFGraph instance.
        :return: TModel without activation Fake Quantize nodes (or Quantize-Dequantize pairs).
        """

    @staticmethod
    @abstractmethod
    def get_port_id(target_point: TargetPoint) -> int:
        """
        Returns port id from the given backend-specific target point.
        Port id is an input port id in case target point target type is
        TargetType.PRE_LAYER_OPERATION or TargetType.OPERATOR_PRE_HOOK and
        is an output port id in case target point target type is
        TargetType.POST_LAYER_OPERATION or TargetType.OPERATOR_POST_HOOK.

        :param target_point: TargetPoint instance.
        :return: Port id from the given backend-specific target point
        """
