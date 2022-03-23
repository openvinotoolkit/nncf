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

from typing import List

from nncf.common.graph.layer_attributes import BaseLayerAttributes


class TFNodeAttributes(BaseLayerAttributes):
    """
    Contains the TF-specific attributes of the NNCFNode.
    """

    def __init__(self, data_format: str):
        """
        Initializes the TF-specific attributes of the NNCFNode.

        :param data_format: The data format of the input and
            output data of the node. One of the following:
            `channels_last` or `channels_first`.
        """
        self._data_format = data_format

    def get_data_format(self) -> str:
        """
        Returns the data format of the input and output data of the node.

        :return: The data format of the input and output data of the node.
        """
        return self._data_format


class TFWeightedNodeAttributes(TFNodeAttributes):
    """
    Contains the TF-specific attributes of the NNCFNode with weight.
    """

    def __init__(self, data_format: str, weight_shape: List[int]):
        """
        Initializes the TF-specific attributes of the NNCFNode.

        :param data_format: The data format of the input and
            output data of the node. One of the following:
            `channels_last` or `channels_first`.
        :param weight_shape: The shape of the weight tensor.
        """
        super().__init__(data_format)
        self._weight_shape = weight_shape

    def get_weight_shape(self) -> List[int]:
        """
        Returns shape of the weight tensor.

        :return: Shape of the weight tensor.
        """
        return self._weight_shape
