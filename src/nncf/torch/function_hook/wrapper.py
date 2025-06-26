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

from __future__ import annotations

import inspect
import types
from types import MethodType
from typing import Any, Callable, Optional, TypeVar, cast

from torch import nn

import nncf
from nncf.torch.function_hook.hook_executor_mode import FunctionHookMode
from nncf.torch.function_hook.hook_storage import HookStorage
from nncf.torch.function_hook.hook_storage import RemovableHookHandle

ATR_HOOK_STORAGE = "__nncf_hooks"

TModel = TypeVar("TModel", bound=nn.Module)


class ForwardWithHooks:
    """Class to wrap forward function of nn.Module, to forward function of the model with enabled FunctionHookMode"""

    __slots__ = "_orig_forward", "_model", "__dict__", "__weakref__"
    _orig_forward: Callable[..., Any]
    _model: nn.Module

    def __new__(cls, model: nn.Module, orig_forward: Optional[Callable[..., Any]] = None) -> ForwardWithHooks:
        if isinstance(model.forward, ForwardWithHooks):
            msg = "Func already wrapped"
            raise TypeError(msg)

        self = super().__new__(cls)

        self._orig_forward = model.forward if orig_forward is None else orig_forward
        self._model = model
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        with FunctionHookMode(model=self.model, hook_storage=get_hook_storage(self.model)) as ctx:
            args, kwargs = ctx.process_model_inputs(args, kwargs)
            outputs = self.orig_forward(*args, **kwargs)
            outputs = ctx.process_model_outputs(outputs)
            return outputs

    def __repr__(self) -> str:
        return f"ForwardWithHooks.{repr(self.orig_forward)}"

    def __reduce__(self) -> Any:
        return type(self), (self.model, self.orig_forward), (self.model, self.orig_forward, self.__dict__ or None)

    def __setstate__(self, state: Any) -> None:
        if not isinstance(state, tuple):
            msg = "argument to __setstate__ must be a tuple"
            raise TypeError(msg)
        if len(state) != 3:
            msg = f"expected 3 items in state, got {len(state)}"
            raise TypeError(msg)
        model, orig_forward, namespace = state
        if not callable(orig_forward) or (namespace is not None and not isinstance(namespace, dict)):
            msg = "invalid partial state"
            raise TypeError(msg)

        if namespace is None:
            namespace = {}

        self._model = model
        self._orig_forward = orig_forward
        self.__dict__ = namespace

    @property
    def __code__(self) -> types.CodeType:
        return self.__call__.__code__

    @property
    def __globals__(self) -> dict[str, Any]:
        return self.orig_forward.__globals__

    @property
    def __name__(self) -> str:
        return self.orig_forward.__name__

    @property
    def __signature__(self) -> inspect.Signature:
        return inspect.signature(self.orig_forward)

    @property
    def orig_forward(self) -> Callable[..., Any]:
        return self._orig_forward

    @property
    def model(self) -> nn.Module:
        return self._model


class ReplicateForDataParallel:
    """
    Class to wrap _replicate_for_data_parallel function of nn.Module,
    to correctly wrap forward with enabled FunctionHookMode.
    """

    __slots__ = "_func", "__dict__", "__weakref__"
    _func: Callable[..., Any]

    def __new__(cls, func: Callable[..., Any]) -> ReplicateForDataParallel:
        if not callable(func):
            msg = "the first argument must be callable"
            raise TypeError(msg)

        if isinstance(func, ReplicateForDataParallel):
            msg = "Func already wrapped"
            raise TypeError(msg)

        self = super().__new__(cls)

        self._func = func
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> nn.Module:
        module = cast(nn.Module, self._func.__self__)  # type: ignore[attr-defined]
        saved_forward_with_hooks = module.forward

        # Ensure that the forwarding method is not overridden. If it is, calling forward()
        # will raise a RuntimeError due to mismatched device assignments, e.g.:
        # "Expected all tensors to be on the same device, but found at least two devices, cuda:1 and cuda:0!"
        # This happens because __self__ still references the original model for all replicas
        # in an overridden forward method.

        if not isinstance(saved_forward_with_hooks, ForwardWithHooks):
            msg = "Not supported overridden forward method, expected ForwardWithHooks"
            raise nncf.InternalError(msg)

        if not (
            isinstance(saved_forward_with_hooks.orig_forward, types.MethodType)
            and saved_forward_with_hooks.orig_forward.__func__ is module.__class__.forward
        ):
            msg = "Not supported overridden forward method of original module"
            raise nncf.InternalError(msg)

        module.__dict__.pop("forward")

        replica: nn.Module = self.func(*args, **kwargs)
        replica.forward = ForwardWithHooks(replica)
        module.forward = saved_forward_with_hooks

        return replica

    def __repr__(self) -> str:
        return f"ReplicateForDataParallel.{repr(self._func)}"

    def __reduce__(self) -> tuple[Callable[..., Any], tuple[Any, ...], tuple[Any, ...]]:
        return type(self), (self._func,), (self._func, self.__dict__ or None)

    def __setstate__(self, state: dict[str, Any]) -> None:
        if not isinstance(state, tuple):
            msg = "argument to __setstate__ must be a tuple"
            raise TypeError(msg)
        if len(state) != 2:
            msg = f"expected 2 items in state, got {len(state)}"
            raise TypeError(msg)
        func, namespace = state
        if not callable(func) or (namespace is not None and not isinstance(namespace, dict)):
            msg = "invalid partial state"
            raise TypeError(msg)

        if namespace is None:
            namespace = {}

        self.__dict__ = namespace
        self._func = func

    @property
    def func(self) -> MethodType:
        return cast(MethodType, self._func)


def wrap_model(model: TModel) -> TModel:
    """
    Wraps a nn.Module to inject custom behavior into the forward pass and replication process.

    This function modifies the given model by:
    1. Replacing the model's `forward` method with a wrapped version (`ForwardWithHooks`) that allows
       additional hooks to be executed during the forward pass by using FunctionHookMode.
    2. Wrapping the model's `_replicate_for_data_parallel` method with `ReplicateForDataParallel`,
       which allows custom behavior when the model is replicated across multiple devices (e.g., for
       data parallelism).
    3. Adding a new module, `HookStorage`, to the model under the attribute `ATR_HOOK_STORAGE`.

    :param model: The nn.Module to be wrapped.
    :return: The modified model with the custom behavior injected.
    """
    model.forward = ForwardWithHooks(model)
    model._replicate_for_data_parallel = ReplicateForDataParallel(model._replicate_for_data_parallel)  # type: ignore
    model.add_module(ATR_HOOK_STORAGE, HookStorage())
    return model


def is_wrapped(model: nn.Module) -> bool:
    """
    Checks if a given model has been wrapped by the `wrap_model` function.

    :param model: The nn.Module to check.
    :return: `True` if the model's `forward` method is an instance of `ForwardWithHooks`,
        indicating that the model has been wrapped; `False` otherwise.
    """
    return isinstance(model.forward, ForwardWithHooks)


def get_hook_storage(model: nn.Module) -> HookStorage:
    """
    Retrieves the `HookStorage` module from the given model.

    This function accesses the model's attribute defined by `ATR_HOOK_STORAGE`
    and returns the `HookStorage` module associated with it.


    :param model: The PyTorch model from which to retrieve the `HookStorage`.
    :return: The `HookStorage` module associated with the model.
    """
    storage = getattr(model, ATR_HOOK_STORAGE)
    if storage is None:
        msg = "Hook storage is not exist in the model"
        raise nncf.InstallationError(msg)
    return cast(HookStorage, getattr(model, ATR_HOOK_STORAGE))


def register_pre_function_hook(model: nn.Module, op_name: str, port_id: int, hook: nn.Module) -> RemovableHookHandle:
    """
    Registers a pre-function hook for a specific operation in the model.

    :param model: The model to register the hook to.
    :param op_name: The name of the operation associated with the hook.
    :param port_id: The port ID associated with the hook.
    :param hook: The pre-function hook module to be executed.
    :return: A handle that can be used to remove the hook later.
    """
    hook_storage = get_hook_storage(model)
    return hook_storage.register_pre_function_hook(op_name, port_id, hook)


def register_post_function_hook(model: nn.Module, op_name: str, port_id: int, hook: nn.Module) -> RemovableHookHandle:
    """
    Registers a post-function hook for a specific operation in the model.

    :param model: The model to register the hook to.
    :param op_name: The name of the operation associated with the hook.
    :param port_id: The port ID associated with the hook.
    :param hook: The pre-function hook module to be executed.
    :return: A handle that can be used to remove the hook later.
    """
    hook_storage = get_hook_storage(model)
    return hook_storage.register_post_function_hook(op_name, port_id, hook)
