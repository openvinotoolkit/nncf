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

import tensorflow as tf

from nncf.common.tensor import NNCFTensor
from nncf.common.tensor import NNCFBaseTensorProcessor


class TFNNCFTensorProcessor(NNCFBaseTensorProcessor):
    """
    A realization of the processing methods set for TFNNCFTensors.
    """

    @classmethod
    def reduce_min(cls, x: NNCFTensor, axis: Union[int, tuple, list]) -> NNCFTensor:
        return TFNNCFTensor(tf.reduce_min(x.tensor, axis=axis))

    @classmethod
    def reduce_max(cls, x: NNCFTensor, axis: Union[int, tuple, list]) -> NNCFTensor:
        return TFNNCFTensor(tf.reduce_max(x.tensor, axis=axis))

    @classmethod
    def abs(cls, x: NNCFTensor) -> NNCFTensor:
        return TFNNCFTensor(tf.math.abs(x.tensor))

    @classmethod
    def min(cls, x1: NNCFTensor, x2: NNCFTensor) -> NNCFTensor:
        return TFNNCFTensor(tf.math.minimum(x1.tensor, x2.tensor))

    @classmethod
    def max(cls, x1: NNCFTensor, x2: NNCFTensor) -> NNCFTensor:
        return TFNNCFTensor(tf.math.maximum(x1.tensor, x2.tensor))

    @classmethod
    def mean(cls, x: NNCFTensor, axis: Union[int, tuple, list]) -> NNCFTensor:
        return TFNNCFTensor(tf.math.reduce_mean(x.tensor, axis=axis))

    @classmethod
    def stack(cls, x: Union[List[NNCFTensor], Deque[NNCFTensor]], axis: int = 0) -> NNCFTensor:
        x = [t.tensor for t in x]
        return TFNNCFTensor(tf.stack(x, axis=axis))

    @classmethod
    def unstack(cls, x: NNCFTensor, axis: int = 0) -> List[NNCFTensor]:
        tensor_list = tf.unstack(x.tensor, axis=axis)
        return [TFNNCFTensor(t) for t in tensor_list]

    @classmethod
    def concatenate(cls, tensors: List[NNCFTensor], axis: int) -> NNCFTensor:
        # pylint: disable=E1120,E1123
        ret_tensor = tf.concat([t.tensor for t in tensors], axis=axis)
        return TFNNCFTensor(ret_tensor)

    @classmethod
    def ones(cls, shape: Union[int, List[int]], device: tf.device) -> NNCFTensor:
        with tf.device(device):
            return TFNNCFTensor(tf.ones(shape))

    @classmethod
    def assert_allclose(cls, tensors: List[NNCFTensor]) -> None:
        for input_mask in tensors[1:]:
            tf.debugging.assert_near(tensors[0].tensor, input_mask.tensor)

    @classmethod
    def repeat(cls, tensor: NNCFTensor, repeats: int) -> NNCFTensor:
        ret_tensor = tf.repeat(tensor, repeats=repeats)
        return TFNNCFTensor(ret_tensor)

    @classmethod
    def elementwise_mask_propagation(cls, input_masks: List[NNCFTensor]) -> NNCFTensor:
        cls.assert_allclose(input_masks)
        return input_masks[0]


class TFNNCFTensor(NNCFTensor):
    """
    A realisation of tensorflow tensors wrapper for common NNCF algorithms.
    """

    def __init__(self, tensor: tf.Tensor):
        # In case somebody attempts to wrap
        # tensor twice
        if isinstance(tensor, self.__class__):
            tensor = tensor.tensor

        super().__init__(tensor, TFNNCFTensorProcessor)

    @property
    def device(self) -> tf.device:
        return self._tensor.device
