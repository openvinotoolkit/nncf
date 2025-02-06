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
import tensorflow as tf

from nncf.common.compression import BaseCompressionAlgorithmController
from nncf.common.sparsity.controller import SparsityController
from nncf.tensorflow.graph.metatypes import keras_layers as layer_metatypes
from nncf.tensorflow.sparsity.utils import strip_model_from_masks

SPARSITY_LAYER_METATYPES = [
    layer_metatypes.TFConv1DLayerMetatype,
    layer_metatypes.TFConv2DLayerMetatype,
    layer_metatypes.TFConv3DLayerMetatype,
    layer_metatypes.TFDepthwiseConv1DSubLayerMetatype,
    layer_metatypes.TFDepthwiseConv2DSubLayerMetatype,
    layer_metatypes.TFDepthwiseConv3DSubLayerMetatype,
    layer_metatypes.TFDepthwiseConv2DLayerMetatype,
    layer_metatypes.TFConv1DTransposeLayerMetatype,
    layer_metatypes.TFConv2DTransposeLayerMetatype,
    layer_metatypes.TFConv3DTransposeLayerMetatype,
    layer_metatypes.TFSeparableConv1DLayerMetatype,
    layer_metatypes.TFSeparableConv2DLayerMetatype,
    layer_metatypes.TFEmbeddingLayerMetatype,
    layer_metatypes.TFLocallyConnected1DLayerMetatype,
    layer_metatypes.TFLocallyConnected2DLayerMetatype,
    layer_metatypes.TFDenseLayerMetatype,
]


class BaseSparsityController(BaseCompressionAlgorithmController, SparsityController):
    """
    Serves as a handle to the additional modules, parameters and hooks inserted
    into the original uncompressed model to enable sparsity-specific compression.
    Hosts entities that are to be used during the training process, such as
    compression scheduler and compression loss.
    """

    def __init__(self, target_model, op_names):
        super().__init__(target_model)
        self._op_names = op_names

    def strip_model(self, model: tf.keras.Model, do_copy: bool = False) -> tf.keras.Model:
        # Transform model for sparsity creates copy of the model.
        return strip_model_from_masks(model, self._op_names)
