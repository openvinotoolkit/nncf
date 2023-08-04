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

import tensorflow as tf

from nncf.common.pruning.tensor_processor import NNCFPruningBaseTensorProcessor
from nncf.experimental.tensor import Tensor


class TFNNCFPruningTensorProcessor(NNCFPruningBaseTensorProcessor):
    """
    A realization of the processing methods set for TFNNCFTensors.
    """

    @classmethod
    def concatenate(cls, tensors: List[Tensor], axis: int) -> Tensor:
        ret_tensor = tf.concat([t.data for t in tensors], axis=axis)
        return Tensor(ret_tensor)

    @classmethod
    def ones(cls, shape: Union[int, List[int]], device: tf.device) -> Tensor:
        with tf.device(device):
            return Tensor(tf.ones(shape))

    @classmethod
    def assert_allclose(cls, tensors: List[Tensor]) -> None:
        for input_mask in tensors[1:]:
            tf.debugging.assert_near(tensors[0].data, input_mask.data)

    @classmethod
    def repeat(cls, tensor: Tensor, repeats: int) -> Tensor:
        ret_tensor = tf.repeat(tensor.data, repeats=repeats)
        return Tensor(ret_tensor)

    @classmethod
    def elementwise_mask_propagation(cls, input_masks: List[Tensor]) -> Tensor:
        cls.assert_allclose(input_masks)
        return input_masks[0]

    @classmethod
    def split(cls, tensor: Tensor, output_shapes: List[int]) -> List[Tensor]:
        chunks = len(output_shapes)
        ret_tensors = tf.split(tensor.data, chunks)
        return [Tensor(ret_tensor) for ret_tensor in ret_tensors]
