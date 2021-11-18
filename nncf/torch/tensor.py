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

from typing import List, Union, Deque

import torch

from nncf.common.tensor import NNCFTensor
from nncf.common.tensor import NNCFBaseTensorProcessor


class PTNNCFTensorProcessor(NNCFBaseTensorProcessor):
    """
    A realization of the processing methods set for PTNNCFTensors.
    """

    @classmethod
    def reduce_min(cls, x: NNCFTensor, axis: Union[int, tuple]) -> NNCFTensor:
        return PTNNCFTensor(torch.amin(x.tensor, dim=axis))

    @classmethod
    def reduce_max(cls, x: NNCFTensor, axis: Union[int, tuple]) -> NNCFTensor:
        return PTNNCFTensor(torch.amax(x.tensor, dim=axis))

    @classmethod
    def abs(cls, x: NNCFTensor) -> NNCFTensor:
        return PTNNCFTensor(torch.abs(x.tensor))

    @classmethod
    def min(cls, x1: NNCFTensor, x2: NNCFTensor) -> NNCFTensor:
        return PTNNCFTensor(torch.min(x1.tensor, x2.tensor))

    @classmethod
    def max(cls, x1: NNCFTensor, x2: NNCFTensor) -> NNCFTensor:
        return PTNNCFTensor(torch.max(x1.tensor, x2.tensor))

    @classmethod
    def mean(cls, x: NNCFTensor, axis: Union[int, tuple]) -> NNCFTensor:
        return PTNNCFTensor(x.tensor.mean(dim=axis))

    @classmethod
    def stack(cls, x: Union[List[NNCFTensor], Deque[NNCFTensor]], axis: int = 0) -> NNCFTensor:
        x = [t.tensor for t in x]
        return PTNNCFTensor(torch.stack(x, dim=axis))

    @classmethod
    def unstack(cls, x: NNCFTensor, axis: int = 0) -> List[NNCFTensor]:
        tensor_list = torch.unbind(x.tensor, dim=axis)
        return [PTNNCFTensor(t) for t in tensor_list]

    @classmethod
    def concatenate(cls, tensors: List[NNCFTensor], axis: int) -> NNCFTensor:
        ret_tensor = torch.cat([t.tensor for t in tensors], dim=axis)
        return PTNNCFTensor(ret_tensor)

    @classmethod
    def ones(cls, shape: Union[int, List[int]], device: torch.device) -> NNCFTensor:
        return PTNNCFTensor(torch.ones(shape, device=device))

    @classmethod
    def assert_allclose(cls, tensors: List[NNCFTensor]) -> None:
        for input_mask in tensors[1:]:
            assert torch.allclose(tensors[0].tensor, input_mask.tensor)

    @classmethod
    def repeat(cls, tensor: NNCFTensor, repeats: int) -> NNCFTensor:
        ret_tensor = torch.repeat_interleave(tensor.tensor, repeats)
        return PTNNCFTensor(ret_tensor)

    @classmethod
    def elementwise_mask_propagation(cls, input_masks: List[NNCFTensor]) -> NNCFTensor:
        cls.assert_allclose(input_masks)
        return input_masks[0]


class PTNNCFTensor(NNCFTensor):
    """
    A realisation of torch tensors wrapper for common NNCF algorithms.
    """

    def __init__(self, tensor: torch.tensor):
        # In case somebody attempts to wrap
        # tensor twice
        if isinstance(tensor, self.__class__):
            tensor = tensor.tensor

        super().__init__(tensor, PTNNCFTensorProcessor)

    @property
    def device(self) -> torch.device:
        return self._tensor.device
