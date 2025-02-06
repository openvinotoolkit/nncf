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
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field
from itertools import chain
from types import MethodType
from types import TracebackType
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Type
from weakref import ReferenceType
from weakref import ref

import torch
from torch import Tensor
from torch import nn
from torch.overrides import TorchFunctionMode

from nncf.common.logging import nncf_logger as logger
from nncf.experimental.torch2.function_hook.handle_inner_functions import get_handle_inner_function
from nncf.experimental.torch2.function_hook.hook_storage import HookStorage
from nncf.experimental.torch2.function_hook.weak_map import WeakUnhashableKeyMap

IGNORED_FN_NAMES = [
    "__repr__",
    "_assert",
    "dim",
    "size",
    "is_floating_point",
    "_set_grad_enabled",
]


@dataclass
class OpMeta:
    """
    Metadata for an operation to be executed, including its name, the callable function,
    and any additional information.


    :param op_name: The name of the operation.
    :param func: The function to be executed for the operation.
    :param extra_info: A dictionary for storing any additional information about the operation.
    """

    op_name: str
    func: Callable[..., Any]
    extra_info: Dict[str, Any] = field(default_factory=lambda: dict())


def _get_full_fn_name(fn: Callable[..., Any]) -> str:
    """
    Get the full name of a function, including its module if applicable.

    :param fn: The function for which to get the full name.
    :return: The full name of the function.
    """
    if inspect.ismethoddescriptor(fn) or inspect.ismethod(fn):
        return fn.__qualname__
    if inspect.isbuiltin(fn) or inspect.isfunction(fn):
        return f"{fn.__module__}.{fn.__name__}"
    return f"{fn.__name__}"


def generate_normalized_op_name(module_name: str, fn_name: str, call_id: Optional[int] = None) -> str:
    """
    Returns a normalized name of operation.

    Returns: The string representation in the format "module_name/fn_name" or "module_name/fn_name/call_id".
    """
    if call_id is None:
        return f"{module_name}/{fn_name}"
    return f"{module_name}/{fn_name}/{call_id}"


class FunctionHookMode(TorchFunctionMode):
    """
    Executes pre- and post-hooks for PyTorch functions within a model's execution.

    This mode wraps the function calls in the model to allow custom hooks to be executed before
    and after the actual function calls.


    :param model: The PyTorch model to which the hooks will be applied.
    :param hook_storage: Storage for hooks to be executed.
    :param module_call_stack: A stack tracking the modules being called.
    :param nested_enter_count: A counter to track nested context manager entries.
    :param op_calls: A dictionary to track operation calls.
    """

    def __init__(self, model: nn.Module, hook_storage: HookStorage) -> None:
        """
        Initialize the FunctionHookMode.

        :param model: The PyTorch model to which the hooks will be applied.
        :param hook_storage: Storage for hooks to be executed.
        """
        super().__init__()
        self.hook_storage: HookStorage = hook_storage
        self.model: nn.Module = model
        self.module_call_stack: List[nn.Module] = []
        self.nested_enter_count = 0
        self.op_calls: Dict[str, int] = defaultdict(int)
        self.enabled: bool = True

        # Variables for hooks after constant nodes
        self.const_name_map: WeakUnhashableKeyMap[torch.Tensor, str] = WeakUnhashableKeyMap()
        for name, parameter in chain(self.model.named_parameters(), self.model.named_buffers()):
            self.const_name_map[parameter] = name
        self.in_process_const = False

        # Hook names
        self.hooks_module_to_group_name: Dict[ReferenceType[nn.Module], str] = {}
        self._get_named_hooks(self.hook_storage.pre_hooks, "pre_hook")
        self._get_named_hooks(self.hook_storage.post_hooks, "post_hook")

    def _get_named_hooks(self, storage: nn.ModuleDict, prefix: str) -> None:
        """
        Associates named hooks from the given module storage with a group name, updating
        the `hooks_module_to_group_name` dictionary for each hook.

        The group name is formatted as '{prefix}__{hook_key}[{hook_id}]', and any forward
        slashes in the hook key are replaced with dashes to avoid conflicts with separator
        characters used by the module hierarchy.

        :param storage: A module that stores hook modules.
        :param prefix: A string prefix used to group hooks by their context or origin.
        """
        for hook_key, hook_module_dict in storage.named_children():
            for hook_id, hook_module in hook_module_dict.named_children():
                # Replace / to avoid collision with module separator
                hook_name = hook_key.replace("/", "-")
                self.hooks_module_to_group_name[ref(hook_module)] = f"{prefix}__{hook_name}[{hook_id}]"

    def _get_wrapped_call(self, fn_call: MethodType) -> Callable[..., Any]:
        """
        Wrap a function call to include pushing to and popping from the module call stack.

        :param fn_call: The original function call to wrap.
        :return: The wrapped function call.
        """

        def wrapped_call(self_: nn.Module, *args: Tuple[Any, ...], **kwargs: Dict[str, Any]) -> Any:
            self.push_module_call_stack(self_)
            retval = fn_call.__func__(self_, *args, **kwargs)
            self.pop_module_call_stack()
            return retval

        return wrapped_call

    def __enter__(self) -> FunctionHookMode:
        """
        Enter the context manager.
        Wrapping the _call_impl function of each module on first nested enter.

        :return: The instance of FunctionHookMode.
        """
        super().__enter__()  # type: ignore
        if self.nested_enter_count == 0:
            # Wrap _call_impl function of instance each module.
            # Note: __call__ can`t not be overrided for instance, the function can be override only in class namespace.
            logger.debug("FunctionHookMode.__enter__: wrap _call_impl function")
            for _, module in self.model.named_modules():
                module._call_impl = types.MethodType(self._get_wrapped_call(module._call_impl), module)
            self.push_module_call_stack(self.model)
        self.nested_enter_count += 1
        return self

    def __exit__(
        self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]
    ) -> None:
        """
        Exit the context manager, unwrapping the _call_impl function of each module.

        :param exc_type: Exception type.
        :param exc_val: Exception value.
        :param exc_tb: Traceback.
        """
        self.nested_enter_count -= 1
        if self.nested_enter_count == 0:
            # Unwrap _call_impl functions
            logger.debug("FunctionHookMode.__exit__: unwrap _call_impl function")
            for _, module in self.model.named_modules():
                module.__dict__.pop("_call_impl")
            self.pop_module_call_stack()
        super().__exit__(exc_type, exc_val, exc_tb)  # type: ignore

    def __torch_function__(
        self,
        func: Callable[..., Any],
        types: List[Type[Any]],
        args: Tuple[Any, ...] = (),
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Override the __torch_function__ method to add pre- and post-hook execution.

        :param func: The function being called.
        :param types: List of types.
        :param args: The arguments to the function.
        :param kwargs: The keyword arguments to the function.
        :return: The output of the function call after hooks have been executed.
        """
        kwargs = kwargs or {}

        fn_name = func.__name__

        # WA: to catch nested calls for some functions
        # https://github.com/pytorch/pytorch/issues/55093
        fn_for_nested_call = get_handle_inner_function(func)
        if fn_for_nested_call is not None:
            with self:
                return fn_for_nested_call(*args, **kwargs)

        if not self.enabled or fn_name in IGNORED_FN_NAMES:
            return func(*args, **kwargs)

        op_name = self.get_current_executed_op_name(fn_name)
        full_fn_name = _get_full_fn_name(func)
        logger.debug(f"FunctionHookMode.__torch_function__: {full_fn_name=} {op_name=}")
        self.register_op(fn_name)
        op_meta = OpMeta(op_name=op_name, func=func)
        args, kwargs = self.execute_pre_hooks(args, kwargs, op_meta)
        output = func(*args, **kwargs)
        output = self.execute_post_hooks(output, op_meta)
        return output

    def push_module_call_stack(self, module: nn.Module) -> None:
        """
        Push a module to the call stack and initialize its operation calls.

        :param module: The module to push onto the call stack.
        """
        assert isinstance(module, nn.Module)
        self.module_call_stack.append(module)
        module_name = self.get_current_relative_name()
        logger.debug(f"FunctionHookMode.push_module_call_stack: {module_name=}")

    def pop_module_call_stack(self) -> None:
        """
        Pop a module from the call stack and remove its operation calls.
        """
        module_name = self.get_current_relative_name()
        self.module_call_stack.pop()
        logger.debug(f"FunctionHookMode.pop_module_call_stack: {module_name=}")

    def get_current_relative_name(self) -> str:
        """
        Get the name of the current module being executed.

        :return: The name of the current module.
        """
        relative_module_names = []
        prev_module = self.module_call_stack[0]

        for module in self.module_call_stack[1:]:
            hook_name = self.hooks_module_to_group_name.get(ref(module))
            if hook_name is not None:
                relative_module_names.append(hook_name)
            else:
                for n, m in prev_module.named_children():
                    if m is module:
                        relative_module_names.append(n)
                        break
            prev_module = module

        return "/".join(relative_module_names)

    def get_current_executed_op_name(self, fn_name: str) -> str:
        """
        Get the name of the current operation being executed.

        :param fn_name: The function name of the operation.
        :return: The name of the operation.
        """
        module_name = self.get_current_relative_name()
        op_name = generate_normalized_op_name(module_name, fn_name)
        call_id = self.op_calls[op_name]
        return generate_normalized_op_name(module_name, fn_name, call_id)

    def register_op(self, fn_name: str) -> None:
        """
        Register an operation call for the current module and increment call counter.

        :param fn_name: The function name of the operation.
        """
        module_name = self.get_current_relative_name()
        op_name = generate_normalized_op_name(module_name, fn_name)
        self.op_calls[op_name] += 1

    def execute_hooks_for_parameter(self, value: torch.Tensor) -> torch.Tensor:
        """
        Executes post-hooks for a model parameter if a hook is defined for it.
        If the input is not a `torch.nn.Parameter`, or if no hook is defined, the original tensor is returned unchanged.

        :param value: The tensor to which the post-hook will be applied..
        :return: The processed tensor with the applied post-hook, if applicable.
        """
        if not isinstance(value, torch.nn.Parameter):
            return value

        name_in_model = self.const_name_map.get(value, None)
        if name_in_model is not None and not self.in_process_const:
            self.in_process_const = True
            value = self.hook_storage.execute_post_function_hooks(name_in_model.replace(".", ":"), 0, value)
            self.in_process_const = False
        return value

    def process_parameters(self, args: List[Any], kwargs: Dict[str, Any]) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Execute post-hooks for parameters.

        :param args: The arguments to the function.
        :param kwargs: The keyword arguments to the function.
        :return: The modified arguments and keyword arguments after pre-hooks.
        """
        for idx, value in enumerate(args):
            args[idx] = self.execute_hooks_for_parameter(value)
        for kw_name, value in kwargs.items():
            kwargs[kw_name] = self.execute_hooks_for_parameter(value)
        return args, kwargs

    def execute_pre_hooks(
        self, args: Tuple[Any, ...], kwargs: Dict[str, Any], op_meta: OpMeta
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """
        Execute pre-hooks for the operation.

        :param args: The arguments to the function.
        :param kwargs: The keyword arguments to the function.
        :param op_meta: Metadata for the operation.
        :return: The modified arguments and keyword arguments after pre-hooks.
        """
        _args: List[Any] = list(args)
        with self:
            _args, kwargs = self.process_parameters(_args, kwargs)

            for idx, value in enumerate(_args):
                _args[idx] = self.hook_storage.execute_pre_function_hooks(op_meta.op_name, idx, value)

            for port_id, kw_name in enumerate(kwargs, start=len(_args)):
                kwargs[kw_name] = self.hook_storage.execute_pre_function_hooks(
                    op_meta.op_name, port_id, kwargs[kw_name]
                )
        return tuple(_args), kwargs

    def process_post_function_hooks_for_value(self, value: Any, op_meta: OpMeta, port_id: int) -> Any:
        """
        Executes post-hooks on a value after an operation.

        :param value: The output value.
        :param op_meta: Metadata about the operation.
        :param port_id: The id of output port.
        :return: The modified output after post-hooks.
        """
        return self.hook_storage.execute_post_function_hooks(op_meta.op_name, port_id, value)

    def execute_post_hooks(self, output: Any, op_meta: OpMeta) -> Any:
        """
        Execute post-hooks for the operation.

        :param output: The output of the function.
        :param op_meta: Metadata for the operation.
        :return: The modified output after post-hooks.
        """
        with self:
            cls_tuple = None
            if isinstance(output, tuple):
                # Need to return named tuples like torch.return_types.max
                cls_tuple = type(output)
                output = list(output)

            if isinstance(output, list):
                for idx, value in enumerate(output):
                    output[idx] = self.process_post_function_hooks_for_value(value, op_meta, idx)
                if cls_tuple is not None:
                    output = cls_tuple(output)
            else:
                output = self.process_post_function_hooks_for_value(output, op_meta, 0)
        return output

    def process_model_inputs(self, args: Tuple[Any], kwargs: Dict[str, Any]) -> Tuple[Tuple[Any], Dict[str, Any]]:
        """
        Processes the input arguments for the model's forward function.

        :param args: Positional arguments passed to the model's forward method.
        :param kwargs: Keyword arguments passed to the model's forward method.
        :return: The processed arguments, with hooks applied to any tensor inputs.
        """
        forward_signature = inspect.signature(self.model.forward)
        bound_arguments = forward_signature.bind(*args, **kwargs)

        # Hooks available only for named arguments
        for name, value in bound_arguments.arguments.items():
            if isinstance(value, Tensor):
                bound_arguments.arguments[name] = self.execute_hooks_for_model_input(name, value)

        return bound_arguments.args, bound_arguments.kwargs

    def execute_hooks_for_model_input(self, name: str, value: Any) -> Any:
        """
        Executes the post-hooks for an input tensor.

        :param name: The name of the input argument.
        :param value: The value of the input argument.
        :return: The processed value after the hook is executed.
        """
        return self.hook_storage.execute_post_function_hooks(name, 0, value)

    def process_model_outputs(self, outputs: Any) -> Any:
        """
        Processes the outputs from the model, applying pre-hooks to any tensors found in the output.

        :param outputs: The outputs returned by the model's forward method.
        :return: The processed outputs with hooks applied.
        """
        if isinstance(outputs, Tensor):
            return self.execute_hooks_for_model_output("output", outputs)

        cls_tuple = None
        if isinstance(outputs, tuple):
            cls_tuple = type(outputs)
            outputs = list(outputs)
        if isinstance(outputs, list):
            outputs = list(outputs)
            for idx, val in enumerate(outputs):
                if isinstance(val, Tensor):
                    outputs[idx] = self.execute_hooks_for_model_output(f"output_{idx}", val)
        if cls_tuple is not None:
            outputs = cls_tuple(outputs)
        return outputs

    def execute_hooks_for_model_output(self, name: str, value: Any) -> Any:
        """
        Executes the pre-hooks for an input tensor.

        :param name: The name of the input argument.
        :param value: The value of the input argument.
        :return: The processed value after the hook is executed.
        """
        return self.hook_storage.execute_pre_function_hooks(name, 0, value)

    @contextmanager
    def disable(self) -> Iterator[None]:
        """
        Temporarily disables the function tracing and execution hooks within a context.

        :return: A context manager which temporary disables the function tracing and execution hooks within a context.
        """
        ret = self.enabled
        self.enabled = False
        yield
        self.enabled = ret
