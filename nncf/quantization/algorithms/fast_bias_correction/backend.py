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
from typing import Dict, List, Optional, Tuple, TypeVar, Union

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.model_transformer import ModelTransformer
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationCommand
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.tensor_statistics.collectors import TensorStatisticCollectorBase
from nncf.tensor import Tensor

TModel = TypeVar("TModel")
TTensor = TypeVar("TTensor")
OutputType = TypeVar("OutputType")


class FastBiasCorrectionAlgoBackend(ABC):
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
    def create_bias_correction_command(
        node: NNCFNode, bias_value: Tensor, nncf_graph: NNCFGraph
    ) -> TransformationCommand:
        """
        Creates backend-specific command to update bias value.

        :param node: The node for which bias should be updated.
        :param bias_value: New value for the bias.
        :param nncf_graph: NNCFGraph instance that contains the node.
        :return: Backend-specific command to update bias value.
        """

    @staticmethod
    @abstractmethod
    def model_extraction_command(
        input_ids: List[Tuple[str, int]], output_ids: List[Tuple[str, int]]
    ) -> TransformationCommand:
        """
        Returns backend-specific command to extract sub-model based on input & output names.

        :param input_ids: List of the input IDs: pairs of node names and correspondent input port ids.
            Each pair denotes the sub-graph beginning.
        :param output_ids: List of the output IDs: pairs of node names and correspondent output port ids.
            Each pair denotes the sub-graph ending.
        :return: Backend-specific TransformationCommand for the model extraction.
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

        :param channel_axis: Channel axes for the statistics aggregation.
        :param inplace: Whether to calculate statistic inplace or not.
        :param num_samples: Maximum number of samples to collect.
        :param window_size: The maximum size of the samples queue.
        :return: Backend-specific TensorStatisticCollectorBase for the statistics calculation.
        """

    @staticmethod
    @abstractmethod
    def get_sub_input_output_names(subgraph: TModel) -> Tuple[Union[str, int], Union[str]]:
        """
        Returns tuple of the subgraph's the input & output tensor names respectively.

        :param node: NNCFNode instance.
        :return: Tuple of the names.
        """

    @staticmethod
    @abstractmethod
    def create_input_data(
        shape: Tuple[int], data: List[TTensor], input_name: str, channel_axis: int
    ) -> Union[Dict[str, TTensor], TTensor]:
        """
        Creates input data for the bias shift calculation.

        :param shape: Shape of the blob.
        :param data: Data to fill the blob.
        :param input_name: Name for the output dictionary.
        :param channel_axis: Axis to fill the blob with provided data.
        :return: Created data.
        """

    @staticmethod
    @abstractmethod
    def get_bias_value(node: NNCFNode, nncf_graph: NNCFGraph, model: TModel) -> Tensor:
        """
        Returns bias value in the NumPy format of provided node.

        :param node: Node of NNCFGraph with bias value.
        :param nncf_graph: NNCFGraph instance.
        :param model: Backend-specific model for the initializer finding.
        :return: Bias value in the NumPy format.
        """

    @staticmethod
    @abstractmethod
    def get_activation_port_ids_for_bias_node(node: NNCFNode) -> Tuple[int, int]:
        """
        Returns Input Port ID and Output Port ID corresponding to activation input and output edges for
        the node.
        Supports only nodes that could have bias value.

        :param node: Node of NNCFGraph with bias value.
        :param model: Backend-specific model.
        """

    @staticmethod
    @abstractmethod
    def is_quantized_weights(node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
        """
        Checks whether the node is quantized or not.

        :param node: NNCFNode to check.
        :param nncf_graph: NNCFGraph instance.

        :return: Boolean indicating whether the node has a quantized weights or not
        """

    @staticmethod
    @abstractmethod
    def process_model_output(raw_data: OutputType, output_name: Union[str, int]) -> Tensor:
        """
        Returns backend-specific processed output from the model.

        :param raw_data: Backend-specific output from the model.
        :param output_name: Name of the output layer or tensor name.
        :return: Processed output as NNCFTensor.
        """

    @staticmethod
    @abstractmethod
    def is_node_with_bias(node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
        """
        Checks whether the node has a bias or not.

        :param node: NNCFNode with the attributes.
        :param nncf_graph: NNCFGraph that contains node.
        :return: Boolean indicating whether the node has a bias or not.
        """

    @staticmethod
    @abstractmethod
    def get_node_names_for_input_output_statistics(node: NNCFNode, nncf_graph: NNCFGraph) -> Tuple[str, str]:
        """
        Return name of nodes to collect statistics.

        :param node: NNCFNode to check.
        :param nncf_graph: NNCFGraph instance.

        :return:
            Name of node to collect input statistics
            Name of node to collect output statistics
        """

    @staticmethod
    @abstractmethod
    def get_activation_channel_axis(node: NNCFNode, port_id: int, input_shape: Tuple[int]) -> int:
        """
        Returns axis number of the activation tensor which correspond to it channel.

        :param node: NNCFNode instance.
        :param port_id: Port ID for input.
        :param input_shape: Shape of the input.
        :return: Channel axis number.
        """

    def extract_submodel(
        self, model_transformer: ModelTransformer, input_id: Tuple[str, int], output_id: Tuple[str, int]
    ) -> TModel:
        """
        Extracts sub-model using backend-specific ModelTransformer.

        :param model_transformer: Backend-specific ModelTransformer.
        :param input_id: Input ID.
        :param output_id: Output ID.
        :return: Backend-specific sub-model.
        """
        model_extraction_command = self.model_extraction_command([input_id], [output_id])
        me_transformation_layout = TransformationLayout()
        me_transformation_layout.register(model_extraction_command)
        extracted_model = model_transformer.transform(me_transformation_layout)
        return extracted_model
