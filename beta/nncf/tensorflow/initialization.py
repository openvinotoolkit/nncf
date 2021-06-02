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
from typing import Dict, Union

import tensorflow as tf

from nncf.common.utils.progress_bar import ProgressBar
from nncf.common.initialization import NNCFDataLoader
from nncf.config.structure import BNAdaptationInitArgs
from nncf.config.config import NNCFConfig
from beta.nncf.tensorflow.graph.metatypes.keras_layers import TFBatchNormalizationLayerMetatype
from beta.nncf.tensorflow.graph.metatypes.matcher import get_keras_layer_metatype


class InitializingDataLoader(NNCFDataLoader):
    """
    This class wraps the tf.data.Dataset class.
    This is required for proper initialization of certain compression algorithms.
    """

    def __init__(self, data_loader: tf.data.Dataset):
        self._data_loader = data_loader

    @property
    def batch_size(self) -> int:
        for (x, _) in self._data_loader:
            batch_size = x.shape[0]
            return batch_size

    def __iter__(self):
        return iter(self._data_loader)


class DataLoaderBNAdaptationRunner:
    def __init__(self, model: tf.keras.Model, init_device: str, num_bn_forget_steps: float):
        self.model = model
        self.init_device = init_device
        self.progressbar_adaptation_description = 'BatchNorm statistics adaptation'
        self.progressbar_forget_description = 'BatchNorm statistics forget'
        self.num_bn_forget_steps = num_bn_forget_steps
        self.momentum_bn_forget = 0.1
        self.original_momenta_values = {}
        self.original_training_state = {}

    @staticmethod
    def _apply_to_batchnorms(func):
        def func_apply_to_bns(layer):
            if get_keras_layer_metatype(layer) == TFBatchNormalizationLayerMetatype:
                func(layer)

        return func_apply_to_bns

    def _apply_to_model(self, func):
        for layer in self.model.layers:
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
                                                               momentum_value=self.momentum_bn_forget)))
        try:
            yield
        finally:
            self._apply_to_model(self._apply_to_batchnorms(restore_original_bn_momentum))

    def _infer_batch(self, x: tf.Tensor) -> None:
        """
        Run the forward pass of the model in train mode.
        BatchNormalization moving statistics are getting updated.
        """
        self.model(x, training=True)

    def _run_model_inference(self, data_loader: InitializingDataLoader, num_init_steps: float):
        num_bn_forget_steps = self.num_bn_forget_steps
        with self._bn_training_state_switcher():
            if num_bn_forget_steps is not None and num_bn_forget_steps > 0:
                with self._bn_momentum_switcher():
                    for i, (x, _) in ProgressBar(
                            enumerate(data_loader),
                            total=num_bn_forget_steps,
                            desc=self.progressbar_forget_description
                    ):
                        if i >= num_bn_forget_steps:
                            break
                        self._infer_batch(x)
            for i, (x, _) in ProgressBar(
                    enumerate(data_loader),
                    total=num_init_steps,
                    desc=self.progressbar_adaptation_description
            ):
                if num_init_steps is not None and i >= num_init_steps:
                    break
                self._infer_batch(x)

    def run(self, data_loader: InitializingDataLoader, num_init_steps: float):
        self._run_model_inference(data_loader, num_init_steps)


def register_default_init_args(nncf_config: NNCFConfig,
                               data_loader: Union[tf.data.Dataset, InitializingDataLoader],
                               device: str = None) -> NNCFConfig:
    """
    Register extra structures for the NNCFConfig.
    Initialization of some compression algorithms requires certain extra structures.
    :param nncf_config: NNCF config without extra structures.
    :param data_loader: Dataset used for initialization.
    :param device: Device to perform initialization. If `device` is `None` then the device
            of the model parameters will be used.
    :return: NNCF config with extra structures.
    """
    nncf_config.register_extra_structs([
        BNAdaptationInitArgs(data_loader=InitializingDataLoader(data_loader), device=device)
    ])
    return nncf_config
