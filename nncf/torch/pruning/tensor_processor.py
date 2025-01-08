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

from typing import List, Union

import torch

from nncf.common.pruning.tensor_processor import NNCFPruningBaseTensorProcessor
from nncf.common.tensor import NNCFTensor
from nncf.torch.tensor import PTNNCFTensor


class PTNNCFPruningTensorProcessor(NNCFPruningBaseTensorProcessor):
    """
    A realization of the processing methods set for PTNNCFTensors.
    """

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

    @classmethod
    def split(cls, tensor: NNCFTensor, output_shapes: List[int] = None) -> List[NNCFTensor]:
        chunks = len(output_shapes)
        ret_tensors = torch.chunk(tensor.tensor, chunks)
        return [PTNNCFTensor(ret_tensor) for ret_tensor in ret_tensors]
