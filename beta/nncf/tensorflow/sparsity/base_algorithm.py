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

from nncf.common.sparsity.controller import SparsityController
from beta.nncf.tensorflow.api.compression import TFCompressionAlgorithmController
from beta.nncf.tensorflow.sparsity.utils import strip_model_from_masks

SPARSITY_LAYERS = {
    'Conv1D': {'weight_attr_name': 'kernel'},
    'Conv2D': {'weight_attr_name': 'kernel'},
    'DepthwiseConv2D': {'weight_attr_name': 'depthwise_kernel'},
    'Conv3D': {'weight_attr_name': 'kernel'},
    'Conv2DTranspose': {'weight_attr_name': 'kernel'},
    'Conv3DTranspose': {'weight_attr_name': 'kernel'},
    'Dense': {'weight_attr_name': 'kernel'},
    'SeparableConv1D': {'weight_attr_name': 'pointwise_kernel'},
    'SeparableConv2D': {'weight_attr_name': 'pointwise_kernel'},
    'Embedding': {'weight_attr_name': 'embeddings'},
    'LocallyConnected1D': {'weight_attr_name': 'kernel'},
    'LocallyConnected2D': {'weight_attr_name': 'kernel'}
}


SPARSITY_TF_OPS = [
    'Conv2D',
    'Conv3D',
    'DepthwiseConv2dNative',
    'QuantizedConv2D'
]


class BaseSparsityController(TFCompressionAlgorithmController, SparsityController):
    """
    Serves as a handle to the additional modules, parameters and hooks inserted
    into the original uncompressed model to enable sparsity-specific compression.
    Hosts entities that are to be used during the training process, such as
    compression scheduler and compression loss.
    """

    def __init__(self, target_model, op_names):
        super().__init__(target_model)
        self._op_names = op_names

    def strip_model(self, model):
        return strip_model_from_masks(model, self._op_names)
