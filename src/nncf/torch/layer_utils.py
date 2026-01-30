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
from typing import Any

import torch
from torch import nn

from nncf.common.utils.registry import Registry

COMPRESSION_MODULES = Registry("compression modules")


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
    def from_config(cls, state: dict[str, Any]) -> object:
        """
        Creates a compression module instance from the given config.
        """


class CompressionParameter(nn.Parameter):
    """
    The class that should be used in all compression algorithms instead of torch.nn.Parameter.

    This class utilize `compression_lr_multiplier` parameter
    to increase/decrease gradients for compression algorithms' parameters.
    """

    def __new__(cls, data: torch.Tensor = None, requires_grad: bool = True, compression_lr_multiplier: float = None):
        return super().__new__(cls, data, requires_grad=requires_grad)

    def __init__(self, data: torch.Tensor = None, requires_grad: bool = True, compression_lr_multiplier: float = None):
        """
        :param data: Parameter tensor
        :param requires_grad: If the parameter requires gradient
        :param compression_lr_multiplier: Multiplier for gradient values
        """
        super().__init__()

        if compression_lr_multiplier is not None and self.dtype.is_floating_point:
            self.requires_grad = True
            self.register_hook(lambda grad: compression_lr_multiplier * grad)
            self.requires_grad = requires_grad
