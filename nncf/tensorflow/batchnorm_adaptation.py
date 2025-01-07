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

from itertools import islice

import tensorflow as tf

from nncf.common.initialization.batchnorm_adaptation import BatchnormAdaptationAlgorithmImpl
from nncf.common.logging.progress_bar import ProgressBar
from nncf.tensorflow.graph.metatypes.keras_layers import TFBatchNormalizationLayerMetatype
from nncf.tensorflow.graph.metatypes.matcher import get_keras_layer_metatype


class BNTrainingStateSwitcher:
    """
    Context manager for switching BatchNormalization layer train mode.
    At the enter, set BatchNormalization layers to the train mode.
    At the exit, restore original BatchNormalization layer mode.
    """

    def __init__(self, model: tf.keras.Model):
        self._model = model
        self._original_training_state = {}

    def __enter__(self):
        for layer in self._model.layers:
            if get_keras_layer_metatype(layer) == TFBatchNormalizationLayerMetatype:
                self._original_training_state[layer] = layer.trainable
                layer.trainable = True

    def __exit__(self, *exc):
        for layer in self._model.layers:
            if get_keras_layer_metatype(layer) == TFBatchNormalizationLayerMetatype:
                layer.trainable = self._original_training_state[layer]


class TFBatchnormAdaptationAlgorithmImpl(BatchnormAdaptationAlgorithmImpl):
    """
    Implementation of the batch-norm statistics adaptation algorithm for the TensorFlow backend.
    """

    def run(self, model: tf.keras.Model) -> None:
        """
        Runs the batch-norm statistics adaptation algorithm.

        :param model: A model for which the algorithm will be applied.
        """
        if self._device is not None:
            raise ValueError(
                "TF implementation of batchnorm adaptation algorithm "
                "does not support switch of devices. Model initial device "
                "is used by default for batchnorm adaptation."
            )
        with BNTrainingStateSwitcher(model):
            for x, _ in ProgressBar(
                islice(self._data_loader, self._num_bn_adaptation_steps),
                total=self._num_bn_adaptation_steps,
                desc="BatchNorm statistics adaptation",
            ):
                model(x, training=True)
