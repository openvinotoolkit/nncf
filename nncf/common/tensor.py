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

from abc import abstractmethod
from typing import TypeVar, List

TensorType = TypeVar('TensorType')
DeviceType = TypeVar('DeviceType')


class NNCFTensor:
    """
    An interface of framework specific tensors for common NNCF algorithms.
    """

    def __init__(self, tensor: TensorType,
                 tensor_processor: 'NNCFBaseTensorProcessor'):
        self._tensor = tensor
        self._tensor_processor = tensor_processor

    @property
    def tensor(self) -> TensorType:
        return self._tensor

    @property
    def tensor_processor(self) -> 'NNCFBaseTensorProcessor':
        return self._tensor_processor

    @property
    @abstractmethod
    def device(self) -> DeviceType:
        pass


class NNCFBaseTensorProcessor:
    """
    An interface of the processing methods set for NNCFTensors.
    """

    @classmethod
    @abstractmethod
    def concatenate(cls, tensors: List[NNCFTensor], axis: int) -> NNCFTensor:
        """
        Join a list of NNCFTensors along an existing axis.

        :param tensors: List of NNCFTensors.
        :param axis: The axis along which the tensors will be joined.
        :returns: The concatenated List of the tensors.
        """

    @classmethod
    @abstractmethod
    def ones(cls, shape: List[int], device: DeviceType) -> NNCFTensor:
        """
        Return a new float tensor of given shape, filled with ones.

        :param shape: Shape of the new tensor.
        :param device: Device to put created tensor in.
        :returns: Float tensor of ones with the given shape.
        """

    @classmethod
    @abstractmethod
    def check_all_close(cls, tensors: List[NNCFTensor]) -> None:
        """
        Raises an AssertionError if two objects are not equal.

        :param tensors: List of tensors to check pairwise equality.
        """

    @classmethod
    @abstractmethod
    def repeat(cls, tensor: NNCFTensor, repeats: int) -> NNCFTensor:
        """
        TODO
        Args:
            tensor:
            repeats:

        Returns:

        """

    @classmethod
    @abstractmethod
    def elementwise_output_mask_from_input_masks(cls, tensors: List[NNCFTensor]) -> NNCFTensor:
        """
        TODO
        Args:
            tensors:

        Returns:

        """
