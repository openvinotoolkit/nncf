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

from nncf.common.utils.registry import Registry


class ModelTransformersHandler(Registry):

    def __init__(self, name):
        super().__init__(name)
        self._model_transformer = None

    def register(self):
        def init(obj):
            if self._model_transformer is not None:
                raise ValueError('The model transformer was already registred')
            self._model_transformer = obj(None)
            return obj
        return init

    def get(self):
        if self._model_transformer is None:
            raise ValueError('Need to initialize model transformer before calling')
        return self._model_transformer

PTQ_MODEL_TRANSFORMERS = ModelTransformersHandler('ModelTransformersHandler')
