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

import numpy as np

from abc import abstractproperty
from typing import TypeVar, List


TensorType = TypeVar('TensorType')


class NNCFTensor:
    def __init__(self, tensor: TensorType,
                 mask_processor: 'NNCFBaseTensorProcessor'):
        self._tensor = tensor
        self._mask_processor = mask_processor

    @property
    def tensor(self):
        return self._tensor

    @property
    def mask_processor(self):
        return self._mask_processor

    @abstractproperty
    def device(self):
        pass


class NNCFBaseTensorProcessor:
    @classmethod
    def concatenate(cls, tensors: List[NNCFTensor], axis: int) -> NNCFTensor:
        ret_tensor = np.concatenate([t.tensor for t in tensors], axis=axis)
        return NNCFTensor(ret_tensor, cls)

    @classmethod
    def ones(cls, shape: List[int], device) -> NNCFTensor:
        ret_tensor = np.ones(shape)
        return NNCFTensor(ret_tensor, cls)

    @classmethod
    def check_all_close(cls, tensors: List[NNCFTensor]) -> None:
        for tensor in tensors[1:]:
            np.testing.assert_allclose(tensors[0].tensor, tensor.tensor)