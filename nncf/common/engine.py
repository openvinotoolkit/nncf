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

from abc import ABC
from abc import abstractmethod
from typing import Any


class Engine(ABC):
    """
    The basic class aims to provide the interface to infer the model.
    """

    @abstractmethod
    def infer(self, input_data: Any) -> Any:
        """
        Runs model on the provided input data.
        Returns the raw model outputs.

        :param input_data: inputs for the model
        :return: raw model outputs
        """
