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
from abc import abstractclassmethod
from abc import abstractmethod
from typing import Any, Dict

import torch
from torch import nn

from nncf.common.hook_handle import HookHandle
from nncf.common.hook_handle import add_op_to_registry
from nncf.common.utils.registry import Registry

COMPRESSION_MODULES = Registry("compression modules")


class StatefullModuleInterface(ABC):
    """
    Interface that should be implemented for every registered compression module to make it possible
    to save an compression modules state and create an compression module from the saved state.
    Config of the module should be json serializable, no python objects except
    standart (str, list and etc.) should be present in a compression module config.
    Values for attributes with type torch.nn.Parameter
    is recovered from the model `state_dict`, so there is no need to keep them in the module config.
    Modules should avoid implementation of `__call__` method and use `forward` method instead,
    as torch functions called inside the `__call__` method could not be unambiguously
    separated from the wrapped parent nncf module functions calls, thus nncf is unable to
    identify target point for that call during transformations recovery process.
    """

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Returns the compression module config.
        """

    @abstractclassmethod
    def from_config(cls, state: Dict[str, Any]) -> object:
        """
        Creates a compression module instance from the given config.
        """


class ProxyModule:
    def __init__(self, module):
        self._module = module

    def __getattr__(self, name):
        return getattr(self._module, name)

    @property
    def __class__(self):
        return type(self._module)


class _NNCFModuleMixin:
    """
    Default class for modules that will be optimized by NNCF.

        Attributes:
            op_func_name    Name of corresponding torch function.
            target_weight_dim_for_compression   Target dimension of weights that will be compressed in some algorithms.
            ignored_algorithms   List of algorithms that will skip the module.
            _custom_forward_fn  wrapper of the custom forward function that is called with `self` argument equals to the
                ProxyModule
    """

    op_func_name = ""
    target_weight_dim_for_compression = 0
    _custom_forward_fn = None
    ignored_algorithms = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _NNCFModuleMixin.add_mixin_fields(self)

    @staticmethod
    def add_mixin_fields(obj):
        obj.pre_ops = nn.ModuleDict()
        obj.post_ops = nn.ModuleDict()

    def get_pre_op(self, key):
        return self.pre_ops[key]

    def get_post_op(self, key):
        return self.post_ops[key]

    def register_pre_forward_operation(self, op) -> HookHandle:
        return add_op_to_registry(self.pre_ops, op)

    def remove_pre_forward_operation(self, key):
        return self.pre_ops.pop(key)

    def register_post_forward_operation(self, op) -> HookHandle:
        return add_op_to_registry(self.post_ops, op)

    def remove_post_forward_operation(self, key):
        return self.post_ops.pop(key)

    def reset(self):
        self.pre_ops.clear()
        self.post_ops.clear()

    def forward(self, *args):
        proxy_module = ProxyModule(self)
        for op in self.pre_ops.values():
            op_args = op(proxy_module, args)
            if op_args is not None:
                if not isinstance(op_args, tuple):
                    op_args = tuple([op_args])
                args = op_args
        forward_fn = self._custom_forward_fn.__func__ if self._custom_forward_fn else super().forward.__func__
        results = forward_fn(proxy_module, *args)
        for op in self.post_ops.values():
            op_results = op(proxy_module, results)
            if op_results is not None:
                results = op_results
        return results


class CompressionParameter(nn.Parameter):
    """
    The class that should be used in all compression algorithms instead of torch.nn.Parameter.

    This class utilize `compression_lr_multiplier` parameter from :class:`nncf.NNCFConfig`
    to increase/decrease gradients for compression algorithms' parameters.
    """

    def __new__(cls, data: torch.Tensor = None, requires_grad: bool = True, compression_lr_multiplier: float = None):
        return super().__new__(cls, data, requires_grad=requires_grad)

    def __init__(self, data: torch.Tensor = None, requires_grad: bool = True, compression_lr_multiplier: float = None):
        """

        Args:
            data: Parameter tensor
            requires_grad: If the parameter requires gradient
            compression_lr_multiplier: Multiplier for gradient values
        """
        super().__init__()

        if compression_lr_multiplier is not None and self.dtype.is_floating_point:
            self.requires_grad = True
            self.register_hook(lambda grad: compression_lr_multiplier * grad)
            self.requires_grad = requires_grad
