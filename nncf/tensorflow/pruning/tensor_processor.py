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

import tensorflow as tf

from nncf.common.pruning.tensor_processor import NNCFPruningBaseTensorProcessor
from nncf.common.tensor import NNCFTensor
from nncf.tensorflow.tensor import TFNNCFTensor


class TFNNCFPruningTensorProcessor(NNCFPruningBaseTensorProcessor):
    """
    A realization of the processing methods set for TFNNCFTensors.
    """

    @classmethod
    def concatenate(cls, tensors: List[NNCFTensor], axis: int) -> NNCFTensor:
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
        ret_tensor = tf.repeat(tensor.tensor, repeats=repeats)
        return TFNNCFTensor(ret_tensor)

    @classmethod
    def elementwise_mask_propagation(cls, input_masks: List[NNCFTensor]) -> NNCFTensor:
        cls.assert_allclose(input_masks)
        return input_masks[0]

    @classmethod
    def split(cls, tensor: NNCFTensor, output_shapes: List[int]) -> List[NNCFTensor]:
        chunks = len(output_shapes)
        ret_tensors = tf.split(tensor.tensor, chunks)
        return [TFNNCFTensor(ret_tensor) for ret_tensor in ret_tensors]
