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
from typing import Any, List, Optional, TypeVar

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype

TModel = TypeVar("TModel")


class AccuracyControlAlgoBackend(ABC):
    # Metatypes

    @staticmethod
    @abstractmethod
    def get_quantizer_metatypes() -> List[OperatorMetatype]:
        """
        Returns a list of quantizer metatypes.

        :return: The list of quantizer metatypes.
        """

    @staticmethod
    @abstractmethod
    def get_const_metatypes() -> List[OperatorMetatype]:
        """
        Returns a list of constant metatypes.

        :return: The list of constant metatypes.
        """

    @staticmethod
    @abstractmethod
    def get_quantizable_metatypes() -> List[OperatorMetatype]:
        """
        Returns a list of metatypes for operations that may be quantized.

        :return: The list of metatypes for operations that may be quantized.
        """

    @staticmethod
    @abstractmethod
    def get_quantize_agnostic_metatypes() -> List[OperatorMetatype]:
        """
        Returns a list of quantize agnostic metatypes.

        :return: The list of quantize agnostic metatypes.
        """

    @staticmethod
    @abstractmethod
    def get_shapeof_metatypes() -> List[OperatorMetatype]:
        """
        Returns a list of shape of metatypes.

        :return: The list of shape of metatypes.
        """

    # Manipulations with bias value and weights

    @staticmethod
    @abstractmethod
    def is_node_with_bias(node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
        """
        Checks if the node has a bias or not.

        :param node: The node to check.
        :param nncf_graph: The NNCF graph.
        :return: True` if `node` corresponds to the operation with bias
            (bias is added to the output tensor of that operation), `False` otherwise.
        """

    @staticmethod
    @abstractmethod
    def is_node_with_weight(node: NNCFNode) -> bool:
        """
        Checks if the node has a weight or not.

        :param node: The node to check.
        :param nncf_graph: The NNCF graph.
        :return: True` if `node` corresponds to the operation with weights, `False` otherwise.
        """

    @staticmethod
    @abstractmethod
    def get_bias_value(node_with_bias: NNCFNode, nncf_graph: NNCFGraph, model: TModel) -> Any:
        """
        Returns the bias value for the biased node.

        :param node_with_bias: The node that corresponds to the operation with bias.
        :param nncf_graph: The NNCF graph.
        :param model: The model that contains this operation.
        :return: The bias value that is applied to the output tensor of the node's operation.
        """

    @staticmethod
    @abstractmethod
    def get_weight_value(node_with_weight: NNCFNode, model: TModel, port_id: int) -> Any:
        """
        Returns the weight value for the node with weight.

        :param node_with_weight: The node with weight.
        :param model: The model that contains this operation.
        :param port_id: The input port ID to get weight input.
        :return: The weight value.
        """

    @staticmethod
    @abstractmethod
    def get_weight_tensor_port_ids(node: NNCFNode) -> List[Optional[int]]:
        """
        Returns node's input port indices with weights tensors.

        :param node: NNCFNode to find its weights input port indices.
        :return: Weights input port indices.
        """

    # Preparation of model

    @staticmethod
    @abstractmethod
    def prepare_for_inference(model: TModel) -> Any:
        """
        Prepares model for inference.

        :param model: A model that should be prepared.
        :retunr: Prepared model for inference.
        """
