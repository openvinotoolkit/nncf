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

from beta.nncf.tensorflow.api.compression import TFCompressionAlgorithmController


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


class BaseSparsityController(TFCompressionAlgorithmController):
    @staticmethod
    def _apply_mask(wrapped_layer, weight_attr, op_name):
        layer_weight = wrapped_layer.layer_weights[weight_attr]
        op = wrapped_layer.weights_attr_ops[weight_attr][op_name]
        layer_weight.assign(
            op(layer_weight,
               wrapped_layer.ops_weights[op_name],
               False)
        )
        wrapped_layer.set_layer_weight(weight_attr, layer_weight)
