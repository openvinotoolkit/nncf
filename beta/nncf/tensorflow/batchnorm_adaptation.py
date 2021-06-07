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

from itertools import islice

import tensorflow as tf

from nncf.common.batchnorm_adaptation import BatchnormAdaptationAlgorithmImpl
from beta.nncf.tensorflow.graph.metatypes.keras_layers import TFBatchNormalizationLayerMetatype
from beta.nncf.tensorflow.graph.metatypes.matcher import get_keras_layer_metatype
from nncf.common.utils.progress_bar import ProgressBar


class BNTrainingStateSwitcher:
    def __init__(self, model):
        self._model = model
        self._original_training_state = {}

    def __enter__(self):
        for layer in self._model.layers:
            if get_keras_layer_metatype(layer) == TFBatchNormalizationLayerMetatype:
                self._original_training_state[layer] = layer.trainable
                layer.trainable = True

    def __exit__(self, type, value, traceback):
        for layer in self._model.layers:
            if get_keras_layer_metatype(layer) == TFBatchNormalizationLayerMetatype:
                layer.trainable = self._original_training_state[layer]


class BNMomentumSwitcher:
    def __init__(self, model):
        self._model = model
        self._original_momenta_values = {}

    def __enter__(self):
        for layer in self._model.layers:
            if get_keras_layer_metatype(layer) == TFBatchNormalizationLayerMetatype:
                self._original_momenta_values[layer] = layer.momentum
                layer.momentum = 0.1

    def __exit__(self, type, value, traceback):
        for layer in self._model.layers:
            if get_keras_layer_metatype(layer) == TFBatchNormalizationLayerMetatype:
                layer.momentum = self._original_momenta_values[layer]


class TFBatchnormAdaptationAlgorithmImpl(BatchnormAdaptationAlgorithmImpl):
    """
    Implementation of the batch-norm statistics adaptation algorithm for the TensorFlow backend.
    """
    def _infer_batch(self, x) -> None:
        """
        Run the forward pass of the model in train mode.
        BatchNormalization moving statistics are getting updated.
        """
        self._model(x, training=True)

    def run(self, model: tf.keras.Model) -> None:
        """
        Runs the batch-norm statistics adaptation algorithm.

        :param model: A model for which the algorithm will be applied.
        """
        self._model = model
        with BNTrainingStateSwitcher(model):
            with BNMomentumSwitcher(model):
                for (x, _) in ProgressBar(
                        islice(self._data_loader, self._num_bn_forget_steps),
                        total=self._num_bn_forget_steps,
                        desc='BatchNorm statistics forget'
                ):
                    self._infer_batch(x)
            for (x, _) in ProgressBar(
                    islice(self._data_loader, self._num_bn_adaptation_steps),
                    total=self._num_bn_adaptation_steps,
                    desc='BatchNorm statistics adaptation'
            ):
                self._infer_batch(x)
