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

from typing import List, Union

from nncf.common.tensor import NNCFTensor
from nncf.common.pruning.tensor_processor import NNCFPruningBaseTensorProcessor


class SymbolicMask(NNCFTensor):
    """
    Framework agnostic 1D NNCFTensor representation which only uses given dimension and do not uses value
    of the tensor. Keeps additional attribute - symbolic mask producer, pointer to NNCFNode which produced
    this mask during symbolic mask propagation algorithm. NNCFNode produced a (symbolic or not) mask means
    this mask was set as an output mask to this NNCFNode during (symbolic or not) mask propagation.
    Tensor shape and mask producer attributes are correctly propagating during
    symbolic mask propagation by SymbolicMaskProcessor.
    """

    def __init__(self, dimension: int, mask_producers: List[int] = None):
        super().__init__(None)
        self._mask_producers = mask_producers if mask_producers else []
        self._shape = dimension

    @property
    def shape(self) -> List[int]:
        return [self._shape]

    @property
    def mask_producers(self) -> List[int]:
        return self._mask_producers

    @property
    def device(self) -> None:
        return None


class AmbiguousSymbolicMask(SymbolicMask):
    """
    Special case of symbolic mask used when pruning operation
    receive inconsistent set of masks and should produce mask which
    certainly mark all producers of such mask as unprunable by dimension mismatch.
    """

    def __init__(self, mask_producers: List[int] = None):
        super().__init__(-1, mask_producers)


class SymbolicMaskProcessor(NNCFPruningBaseTensorProcessor):
    """
    Implementation of processing methods set for SymbolicMask.
    Responsible for correct mask dimension and mask producer attributes propagation.
    For methods like concatenate and elementwise_mask_propagation unions
    mask producers of input masks.
    """

    @classmethod
    def concatenate(cls, tensors: List[SymbolicMask], axis: int) -> SymbolicMask:
        ret_shape = sum([t.shape[0] for t in tensors])
        producers = []
        for tensor in tensors:
            if tensor.mask_producers is not None:
                producers.extend(tensor.mask_producers)
        if not producers:
            producers = None

        return SymbolicMask(ret_shape, producers)

    @classmethod
    def ones(cls, shape: Union[int, List[int]], device) -> SymbolicMask:
        if isinstance(shape, list):
            if len(shape) != 1:
                raise RuntimeError(f'Unexpected shape = {shape} for 1D symbolic mask')
            shape = shape[0]

        return SymbolicMask(shape)

    @classmethod
    def assert_allclose(cls, tensors: List[SymbolicMask]) -> None:
        for input_mask in tensors[1:]:
            assert tensors[0].shape == input_mask.shape

    @classmethod
    def repeat(cls, tensor: SymbolicMask, repeats: int) -> SymbolicMask:
        return SymbolicMask(tensor.shape[0] * repeats, tensor.mask_producers)

    @classmethod
    def elementwise_mask_propagation(cls, input_masks: List[SymbolicMask]) -> SymbolicMask:
        """
        Assemble output mask for elementwise pruning operation from given input masks.
        In case input_masks have different shape don't propagate any masks.

        :param input_masks: Given input masks.
        :return: Elementwise pruning operation output mask.
        """
        producers = list({p for t in input_masks for p in t.mask_producers})
        for input_mask in input_masks[1:]:
            if not input_masks[0].shape == input_mask.shape:
                return AmbiguousSymbolicMask(producers)

        return SymbolicMask(input_masks[0].shape[0], producers)
