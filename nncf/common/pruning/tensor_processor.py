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

from abc import abstractmethod
from typing import List, Union

from nncf.common.tensor import DeviceType
from nncf.common.tensor import NNCFTensor


class NNCFPruningBaseTensorProcessor:
    """
    An interface of the processing methods for NNCFTensors.
    """

    @classmethod
    @abstractmethod
    def concatenate(cls, tensors: List[NNCFTensor], axis: int) -> NNCFTensor:
        """
        Join a list of NNCFTensors along an existing axis.

        :param tensors: List of NNCFTensors.
        :param axis: The axis, along which the tensors will be joined.
        :returns: The concatenated List of the tensors.
        """

    @classmethod
    @abstractmethod
    def ones(cls, shape: Union[int, List[int]], device: DeviceType) -> NNCFTensor:
        """
        Return a new float tensor of given shape, filled with ones.

        :param shape: Shape of the new tensor.
        :param device: Device to put created tensor in.
        :returns: Float tensor of ones with the given shape.
        """

    @classmethod
    @abstractmethod
    def assert_allclose(cls, tensors: List[NNCFTensor]) -> None:
        """
        Raises an AssertionError if any two tensors are not equal.

        :param tensors: List of tensors to check pairwise equality.
        """

    @classmethod
    @abstractmethod
    def repeat(cls, tensor: NNCFTensor, repeats: int) -> NNCFTensor:
        """
        Successively repeat each element of given NNCFTesnor.

        :param tensor: Given NNCFTensor.
        :param repeats: The number of repetitions for each element.
        :return: NNCFTensor with repited elements.
        """

    @classmethod
    @abstractmethod
    def elementwise_mask_propagation(cls, input_masks: List[NNCFTensor]) -> NNCFTensor:
        """
        Assemble output mask for elementwise pruning operation from given input masks.
        Raises an AssertionError if input masks are not pairwise equal.

        :param input_masks: Given input masks.
        :return: Elementwise pruning operation output mask.
        """

    @classmethod
    @abstractmethod
    def split(cls, tensor: NNCFTensor, output_shapes: List[int]) -> List[NNCFTensor]:
        """
        Split/chunk NNCFTensor into chunks along an exsiting dimension.

        :param tensor: Given NNCFTensor.
        :param output_shapes: Given shapes of the output masks
        :returns: The list of NNCFTensor which is split
        """
