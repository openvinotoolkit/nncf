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

import torch

from nncf.common.tensor import NNCFTensor


class PTNNCFTensor(NNCFTensor):
    """
    A realisation of torch tensors wrapper for common NNCF algorithms.
    """

    def __init__(self, tensor: torch.tensor):
        # In case somebody attempts to wrap
        # tensor twice
        if isinstance(tensor, self.__class__):
            tensor = tensor.tensor

        super().__init__(tensor)

    @property
    def device(self) -> torch.device:
        return self._tensor.device

    def is_empty(self) -> bool:
        return self.tensor.numel() == 0
