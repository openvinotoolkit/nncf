# Copyright (c) 2026 Intel Corporation
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
from typing import Any, TypeVar

TObj = TypeVar("TObj", bound="StatefulModuleInterface")


class StatefulModuleInterface(ABC):
    """
    Interface that should be implemented for every registered compression module to make it possible
    to save an compression modules state and create an compression module from the saved state.
    Config of the module should be json serializable, no python objects except
    standard (str, list and etc.) should be present in a compression module config.
    Values for attributes with type torch.nn.Parameter
    is recovered from the model `state_dict`, so there is no need to keep them in the module config.
    Modules should avoid implementation of `__call__` method and use `forward` method instead,
    as torch functions called inside the `__call__` method could not be unambiguously
    separated from the wrapped parent nncf module functions calls, thus nncf is unable to
    identify target point for that call during transformations recovery process.
    """

    @abstractmethod
    def get_config(self) -> dict[str, Any]:
        """
        Returns the compression module config.
        """

    @classmethod
    @abstractmethod
    def from_config(cls: type[TObj], state: dict[str, Any]) -> TObj:
        """
        Creates a compression module instance from the given config.
        """
