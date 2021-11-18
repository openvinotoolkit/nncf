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
from typing import TypeVar, List, Optional, Union, Deque

TensorType = TypeVar('TensorType')
DeviceType = TypeVar('DeviceType')


class NNCFTensor:
    """
    An interface of framework specific tensors for common NNCF algorithms.
    """

    def __init__(self, tensor: Optional[TensorType],
                 tensor_processor: 'NNCFBaseTensorProcessor'):
        self._tensor = tensor
        self._tensor_processor = tensor_processor

    @property
    def tensor(self) -> TensorType:
        return self._tensor

    @property
    def shape(self) -> List[int]:
        if self._tensor is None:
            raise RuntimeError('Attempt to get shape of empty NNCFTensor')
        return self._tensor.shape

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
    def reduce_min(cls, x: NNCFTensor, axis: Union[int, tuple, list]) -> NNCFTensor:
        """
        Computes minimum of elements across dimensions of NNCFTensor.

        :param x: NNCFTensor to reduce
        :param axis: The dimensions to reduce.
        :return: Reduced NNCFTensor.
        """

    @classmethod
    @abstractmethod
    def reduce_max(cls, x: NNCFTensor, axis: Union[int, tuple, list]) -> NNCFTensor:
        """
        Computes maximum of elements across dimensions of NNCFTensor.

        :param x: NNCFTensor to reduce
        :param axis: The dimensions to reduce.
        :return: Reduced NNCFTensor.
        """

    @classmethod
    @abstractmethod
    def abs(cls, x: NNCFTensor) -> NNCFTensor:
        """
        Computes the absolute value of a NNCFTensor.

        :param x: NNCFTensor
        :return: Absolute value of a NNCFTensor
        """

    @classmethod
    @abstractmethod
    def min(cls, x1: NNCFTensor, x2: NNCFTensor) -> NNCFTensor:
        """
        Returns the min of x1 and x2.

        :param x1: NCFTensor to compare.
        :param x2: NCFTensor to compare.
        :return: Compared NNCFTensor.
        """

    @classmethod
    @abstractmethod
    def max(cls, x1: NNCFTensor, x2: NNCFTensor) -> NNCFTensor:
        """
        Returns the max of x1 and x2.

        :param x1: NNCFTensor to compare.
        :param x2: NNCFTensor to compare.
        :return: Compared NNCFTensor.
        """

    @classmethod
    @abstractmethod
    def mean(cls, x: NNCFTensor, axis: Union[int, tuple, list]) -> NNCFTensor:
        """
        Computes the mean of elements across given dimensions of NNCFTensor.

        :param x: NNCFTensor to reduce.
        :param axis:
        :return: Reduced NNCFTensor.
        """

    @classmethod
    @abstractmethod
    def stack(cls, x: Union[List[NNCFTensor], Deque[NNCFTensor]], axis: int = 0) -> NNCFTensor:
        """
        Stacks a list or deque of NNCFTensors rank-R tensors into one NNCFTensor rank-(R+1) tensor.

        :param x: List or deque of NNCFTensors.
        :param axis: The axis to stack along.
        :return: Stacked NNCFTensor.
        """

    @classmethod
    @abstractmethod
    def unstack(cls, x: NNCFTensor, axis: int = 0) -> list:
        """
        Unstack a tensor into list.

        :param x: NNCFTensor to unstack.
        :param axis: The axis to unstack along.
        :return: List of NNCFTensors.
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
