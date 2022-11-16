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
from typing import TypeVar

TModel = TypeVar('TModel')
ModelInput = TypeVar('ModelInput')
ModelOutput = TypeVar('ModelOutput')


class Engine(ABC):
    """
    The basic class aims to provide the interface to infer the model.
    """

    def __init__(self):
        self._model = None

    @abstractmethod
    def infer(self, input_data: ModelInput) -> ModelOutput:
        """
        Runs model on the provided input_data.
        Returns the dictionary of model outputs by node names.

        :param input_data: inputs for the model transformed with the inputs_transforms
        :return output_data: models output after outputs_transforms
        """
