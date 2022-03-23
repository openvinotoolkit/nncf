"""
 Copyright (c) 2022 Intel Corporation
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

import gc
import weakref

import tensorflow as tf

from nncf.config.utils import is_experimental_quantization


class TFOriginalModelManager:
    def __init__(self, model_fn, *args, **kwargs):
        self._model_fn = model_fn
        self._kwargs = kwargs
        self._args = args
        self._model = None

    def __enter__(self):
        self._model = self._model_fn(*self._args, **self._kwargs)
        tf.keras.backend.clear_session()
        return weakref.proxy(self._model)

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self._model
        gc.collect()


class TFWithoutModelManager(TFOriginalModelManager):
    def __enter__(self):
        return self._model_fn(*self._args, **self._kwargs)

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class TFModelManager:
    """
    Controls the process of model creation.
    """

    def __init__(self, model_fn, nncf_config, *args, **kwargs):
        """
        Initializes the `TFModelManager`.

        :param model_fn: Function for model creation.
        :param nncf_config: NNCF config.
        """
        self._manager = TFModelManager._create_model_manager(model_fn, nncf_config, *args, **kwargs)

    def __enter__(self):
        return self._manager.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._manager.__exit__(exc_type, exc_val, exc_tb)

    @staticmethod
    def _create_model_manager(model_fn, nncf_config, *args, **kwargs):
        """
        Creates model manager depending on the algorithm name.

        :param model_fn: Function for model creation.
        :param nncf_config: NNCF config.
        :return: Model manager.
        """
        if is_experimental_quantization(nncf_config):
            return TFWithoutModelManager(model_fn, *args, **kwargs)
        return TFOriginalModelManager(model_fn, *args, **kwargs)
