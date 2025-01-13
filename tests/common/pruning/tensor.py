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

from typing import List, Optional, Union

import numpy as np

from nncf.common.pruning.tensor_processor import NNCFPruningBaseTensorProcessor
from nncf.common.tensor import NNCFTensor


class NPNNCFTensorProcessor(NNCFPruningBaseTensorProcessor):
    @classmethod
    def concatenate(cls, tensors: List[NNCFTensor], axis: int) -> NNCFTensor:
        for tensor in tensors[1:]:
            assert tensors[0].device == tensor.device

        ret_tensor = np.concatenate([t.tensor for t in tensors], axis=axis)
        return NPNNCFTensor(ret_tensor, tensors[0].device)

    @classmethod
    def ones(cls, shape: Union[int, List[int]], device: Optional[str]) -> NNCFTensor:
        return NPNNCFTensor(np.ones(shape), device)

    @classmethod
    def assert_allclose(cls, tensors: List[NNCFTensor]) -> None:
        for input_mask in tensors[1:]:
            np.testing.assert_allclose(tensors[0].tensor, input_mask.tensor)

    @classmethod
    def repeat(cls, tensor: NNCFTensor, repeats: int) -> NNCFTensor:
        ret_tensor = np.repeat(tensor.tensor, repeats)
        return NPNNCFTensor(ret_tensor)

    @classmethod
    def elementwise_mask_propagation(cls, input_masks: List[NNCFTensor]) -> NNCFTensor:
        cls.assert_allclose(input_masks)
        return input_masks[0]

    @classmethod
    def split(cls, tensor: NNCFTensor, output_shapes: List[int]) -> List[NNCFTensor]:
        chunks = len(output_shapes)
        ret_tensors = np.split(tensor.tensor, chunks)
        return [NPNNCFTensor(ret_tensor) for ret_tensor in ret_tensors]


class NPNNCFTensor(NNCFTensor):
    def __init__(self, tensor: np.array, dummy_device: Optional[str] = None):
        # In case somebody attempts to wrap
        # tensor twice
        if isinstance(tensor, self.__class__):
            tensor = tensor.tensor

        super().__init__(tensor)
        self.dummy_device = dummy_device

    @property
    def device(self) -> Optional[str]:
        return self.dummy_device
