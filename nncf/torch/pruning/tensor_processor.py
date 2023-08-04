# Copyright (c) 2023 Intel Corporation
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
from nncf.experimental.tensor import Tensor


class PTNNCFPruningTensorProcessor(NNCFPruningBaseTensorProcessor):
    """
    A realization of the processing methods set for Tensors.
    """

    @classmethod
    def concatenate(cls, tensors: List[Tensor], axis: int) -> Tensor:
        ret_tensor = torch.cat([t.data for t in tensors], dim=axis)
        return Tensor(ret_tensor)

    @classmethod
    def ones(cls, shape: Union[int, List[int]], device: torch.device) -> Tensor:
        return Tensor(torch.ones(shape, device=device))

    @classmethod
    def assert_allclose(cls, tensors: List[Tensor]) -> None:
        for input_mask in tensors[1:]:
            assert torch.allclose(tensors[0].data, input_mask.data)

    @classmethod
    def repeat(cls, tensor: Tensor, repeats: int) -> Tensor:
        ret_tensor = torch.repeat_interleave(tensor.data, repeats)
        return Tensor(ret_tensor)

    @classmethod
    def elementwise_mask_propagation(cls, input_masks: List[Tensor]) -> Tensor:
        cls.assert_allclose(input_masks)
        return input_masks[0]

    @classmethod
    def split(cls, tensor: Tensor, output_shapes: List[int] = None) -> List[Tensor]:
        chunks = len(output_shapes)
        ret_tensors = torch.chunk(tensor.data, chunks)
        return [Tensor(ret_tensor) for ret_tensor in ret_tensors]
