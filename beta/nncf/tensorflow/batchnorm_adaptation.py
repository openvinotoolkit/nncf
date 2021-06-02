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

from functools import partial
from contextlib import contextmanager
from typing import Dict

import tensorflow as tf

from nncf.common.batchnorm_adaptation import BatchnormAdaptationAlgorithmImpl
from beta.nncf.tensorflow.graph.metatypes.keras_layers import TFBatchNormalizationLayerMetatype
from beta.nncf.tensorflow.graph.metatypes.matcher import get_keras_layer_metatype
from nncf.common.utils.progress_bar import ProgressBar


class TFBatchnormAdaptationAlgorithmImpl(BatchnormAdaptationAlgorithmImpl):
    """
    Implementation of the batch-norm statistics adaptation algorithm for the TensorFlow backend.
    """

    @staticmethod
    def _apply_to_batchnorms(func):
        def func_apply_to_bns(layer):
            if get_keras_layer_metatype(layer) == TFBatchNormalizationLayerMetatype:
                func(layer)

        return func_apply_to_bns

    def _apply_to_model(self, func):
        for layer in self._model.layers:
            func(layer)

    @contextmanager
    def _bn_training_state_switcher(self) -> None:
        def save_original_bn_training_state(module):
            self.original_training_state[module] = module.trainable

        def set_bn_training_state(module, state: Dict[str, bool]):
            module.trainable = state

        def restore_original_bn_training_state(module):
            module.trainable = self.original_training_state[module]

        self._apply_to_model(self._apply_to_batchnorms(save_original_bn_training_state))
        self._apply_to_model(self._apply_to_batchnorms(partial(set_bn_training_state, state=True)))

        try:
            yield
        finally:
            self._apply_to_model(self._apply_to_batchnorms(restore_original_bn_training_state))

    @contextmanager
    def _bn_momentum_switcher(self) -> None:
        def set_bn_momentum(module, momentum_value):
            module.momentum = momentum_value

        def save_original_bn_momentum(module):
            self.original_momenta_values[module] = module.momentum

        def restore_original_bn_momentum(module):
            module.momentum = self.original_momenta_values[module]

        self._apply_to_model(self._apply_to_batchnorms(save_original_bn_momentum))
        self._apply_to_model(self._apply_to_batchnorms(partial(set_bn_momentum,
                                                               momentum_value=0.1)))
        try:
            yield
        finally:
            self._apply_to_model(self._apply_to_batchnorms(restore_original_bn_momentum))

    def _infer_batch(self, x) -> None:
        """
        Run the forward pass of the model in train mode.
        BatchNormalization moving statistics are getting updated.
        """
        self._model(x, training=True)

    def _run_model_inference(self):
        self.original_momenta_values = {}
        self.original_training_state = {}
        num_bn_adaptation_steps = self._num_bn_adaptation_steps
        num_bn_forget_steps = self._num_bn_forget_steps
        with self._bn_training_state_switcher():
            if num_bn_forget_steps is not None and num_bn_forget_steps > 0:
                with self._bn_momentum_switcher():
                    for i, (x, _) in ProgressBar(
                            enumerate(self._data_loader),
                            total=num_bn_forget_steps,
                            desc='BatchNorm statistics forget'
                    ):
                        if i >= num_bn_forget_steps:
                            break
                        self._infer_batch(x)
            for i, (x, _) in ProgressBar(
                    enumerate(self._data_loader),
                    total=num_bn_adaptation_steps,
                    desc='BatchNorm statistics adaptation'
            ):
                if num_bn_adaptation_steps is not None and i >= num_bn_adaptation_steps:
                    break
                self._infer_batch(x)

    def run(self, model: tf.keras.Model) -> None:
        """
        Runs the batch-norm statistics adaptation algorithm.

        :param model: A model for which the algorithm will be applied.
        """
        self._model = model
        self._run_model_inference()
