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

from abc import ABC
from abc import abstractmethod

from typing import Dict
from typing import TypeVar

ModelType = TypeVar('ModelType')
TensorType = TypeVar('TensorType')


class Engine(ABC):
    """
    The basic class aims to provide the interface to infer the model.
    """

    def __init__(self):
        self.model = None

    def set_model(self, model: ModelType) -> None:
        self.model = model

    def is_model_set(self) -> bool:
        return self.model is not None

    def infer(self, _input: TensorType) -> Dict[str, TensorType]:
        if not self.is_model_set():
            raise RuntimeError('The {} tried to infer the model, while the model was not set.'.format(self.__class__))
        return self._infer(_input)

    @abstractmethod
    def _infer(self, _input: TensorType) -> Dict[str, TensorType]:
        """
        Infer the model on the provided input.
        Returns the model outputs and corresponding node names in the model.
        """
