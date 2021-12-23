"""
 Copyright (c) 2021 Intel Corporation
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

from nncf.tensorflow.layers.operation import NNCFOperation
from nncf.experimental.tensorflow.nncf_network import NNCFNetwork


class NNCFOperationV2(NNCFOperation):
    """
    Represents the main building block for adding compression
    extensions to the NNCF network.
    """

    def build(self, nncf_network: NNCFNetwork) -> None:
        """
        Creates operation's weights and adds them to the NNCF network.

        Adding operation weights to the NNCF network is required
        because this is the only way to save it to a checkpoint.

        :param nncf_network: NNCF network, where this operation was added.
        """

    def call(self, inputs, *args, **kwargs):
        """
        Performs the logic of applying the operation to the input tensor.

        :param inputs: Input tensor.
        :return: Result of operation.
        """
