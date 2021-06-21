"""
 Copyright (c) 2020 Intel Corporation
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
