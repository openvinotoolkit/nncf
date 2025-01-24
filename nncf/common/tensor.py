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

from typing import List, TypeVar

import nncf

TensorType = TypeVar("TensorType")
DeviceType = TypeVar("DeviceType")
TensorElementsType = TypeVar("TensorElementsType")


class NNCFTensor:
    """
    An interface of framework specific tensors for common NNCF algorithms.
    """

    def __init__(self, tensor: TensorType):
        self._tensor = tensor

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NNCFTensor):
            raise nncf.InternalError("Attempt to compare NNCFTensor with a non-NNCFTensor object")
        return self._tensor == other.tensor

    @property
    def tensor(self) -> TensorType:  # type: ignore
        return self._tensor

    @property
    def shape(self) -> List[int]:
        if self._tensor is None:
            raise nncf.InternalError("Attempt to get shape of empty NNCFTensor")
        return self._tensor.shape  # type: ignore

    @property
    def device(self) -> DeviceType:  # type: ignore
        raise NotImplementedError

    def is_empty(self) -> bool:
        raise NotImplementedError
