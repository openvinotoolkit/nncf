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
import inspect
import types
from collections import deque
from collections.abc import Sequence as SequenceABC
from functools import wraps
from inspect import getfullargspec
from typing import (
    Any,
    Callable,
    MutableMapping,
    Protocol,
    Sequence,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

try:
    from typing_extensions import ParamSpec
except ImportError:
    from typing import ParamSpec  # type: ignore[assignment]

from nncf.tensor import Tensor
from nncf.tensor.definitions import TensorBackend

F = TypeVar("F", bound=Callable[..., Any])
R = TypeVar("R", covariant=True)
P = ParamSpec("P")


class DispatchCallable(Protocol[P, R]):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...

    def register(self, r_func: F) -> F: ...

    registry: MutableMapping[type, Callable[..., Any]]


def tensor_dispatcher(func: Callable[P, R]) -> DispatchCallable[P, R]:
    """
    Single-dispatch generic function decorator.

    Dispatcher to select implementation of function according the type of the first argument.

    :param func: Function to decorate.
    :return: Decorated function.
    """
    registry: dict[type, Callable[..., Any]] = {}
    signature = inspect.signature(func)
    pos_to_unwrap, name_to_unwrap = _find_arguments_to_unwrap(signature.parameters)

    def dispatch(type_: type) -> Callable[..., Any]:
        """
        Try to find type in registry if not found try to find a parent class in registry.

        :param type_: Type to find in registry.
        :return: Function from registry.
        """
        ret_fn = registry.get(type_)
        if ret_fn is not None:
            return registry[type_]
        # If input type is subclass of expected type, save it in registry for future use
        for key in registry:
            if issubclass(type_, key):
                registry[type_] = registry[key]
                return registry[key]
        msg = f"Function `{func.__name__}` is not implemented for {type_}"
        raise NotImplementedError(msg)

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """
        Wrapper function for dispatching the appropriate function based on the argument type.

        :param *args: Variable length argument list.
        :param **kwargs: Arbitrary keyword arguments.
        :return: The result of the dispatched function.
        """
        func_type = _get_arg_type(args[0])
        args = tuple(_unwrap_tensors(x) if i in pos_to_unwrap else x for i, x in enumerate(args))
        kwargs = {n: _unwrap_tensors(x) if n in name_to_unwrap else x for n, x in kwargs.items()}
        dispatched_func = dispatch(func_type)
        ret = dispatched_func(*args, **kwargs)
        return _wrap_output(ret, signature.return_annotation)

    def register(r_func: F) -> F:
        """
        Register a function to the dispatcher registry.

        :param r_func: The function to be registered.
        :return: The registered function.
        """
        _check_signature(func, r_func)
        spec = getfullargspec(func)
        type_alias = get_type_hints(r_func).get(spec.args[0])
        types = _get_register_types(type_alias)
        for t in types:
            registry[t] = r_func
        return r_func

    wrapper.register = register  # type: ignore[attr-defined]
    wrapper.registry = registry  # type: ignore[attr-defined]
    return cast(DispatchCallable[P, R], wrapper)


def _find_arguments_to_unwrap(
    parameters: types.MappingProxyType[str, inspect.Parameter],
) -> tuple[list[int], list[str]]:
    """
    Get the arguments to unwrap from a given function.

    :param parameters: The parameters of the function.
    :return: A tuple containing two lists - the indexes of the arguments to unwrap and their names.
    """
    indexes: list[int] = []
    names: list[str] = []
    for idx, (name, param) in enumerate(parameters.items()):
        if inspect.Parameter.empty is param.annotation:
            msg = f"Argument {name} has no annotation"
            raise RuntimeError(msg)
        if Tensor in _get_register_types(param.annotation):
            indexes.append(idx)
            names.append(name)
    return indexes, names


def _check_signature(func: Callable[..., Any], r_func: Callable[..., Any]) -> None:
    """
    Check the signature of the dispatched function compared with expected signature.

    :param func: The original function.
    :param r_func: The dispatched function.
    """
    sign_func = getfullargspec(func)
    sign_r_func = getfullargspec(r_func)

    def _raise_error(text: str, expected: Any, actual: Any) -> None:
        """
        Raises a RuntimeError with detailed information about the error.
        """
        file_line = f"{inspect.getsourcefile(r_func)}:{inspect.getsourcelines(r_func)[1]}"
        msg = (
            f"Differ {text} for dispatched a function {func.__name__} in {r_func.__module__}.\n"
            f"Path: {file_line}\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}"
        )
        raise RuntimeError(msg)

    # Check names of argument
    if sign_func.args != sign_r_func.args or sign_func.kwonlyargs != sign_r_func.kwonlyargs:
        _raise_error(
            "names of arguments",
            f"{sign_func.args} {sign_func.kwonlyargs}",
            f"{sign_r_func.args} {sign_r_func.kwonlyargs}",
        )

    # Check that default values is same
    if sign_func.defaults != sign_r_func.defaults:
        _raise_error("default values", sign_func.defaults, sign_r_func.defaults)

    # Check number of annotated arguments
    if len(sign_func.annotations.items()) != len(sign_r_func.annotations.items()):
        _raise_error("count of annotated arguments", sign_func.annotations, sign_r_func.annotations)

    # Check annotation for arguments that not expect Tensor as input
    for (name, ann), rann in zip(sign_func.annotations.items(), sign_r_func.annotations.values()):
        if Tensor not in _get_register_types(ann) and ann != rann:
            _raise_error(f"annotations for argument '{name}'", ann, rann)


def _get_arg_type(arg: Any) -> type:
    """
    Get the type of the argument.

    :param arg: The argument to get the type of.
    """
    if isinstance(arg, Tensor):
        return type(arg.data)
    if isinstance(arg, Sequence):
        return _get_arg_type(arg[0])
    if isinstance(arg, dict) and arg:
        return _get_arg_type(next(iter(arg.values())))
    return type(arg)


def _get_register_types(type_alias: Any) -> list[type]:
    """
    Retrieves the register types from the given type alias.

    :param type_alias: The type alias to retrieve register types from.
    :return: A list of register types.
    """

    def _unpack_types(t: Any) -> list[type]:
        """
        Recursively find types.
        """
        origin = get_origin(t)
        if origin is Union or origin is list or origin is tuple or origin is SequenceABC:
            ret = []
            for tt in get_args(t):
                ret.extend(_unpack_types(tt))
            return ret
        if origin is dict:
            # unpack only values for dict
            return _unpack_types(get_args(t)[1])
        if not isinstance(t, types.GenericAlias):
            return [t]
        return [origin]

    return _unpack_types(type_alias)


def _unwrap_tensors(data: Any) -> Any:
    """
    Unwraps tensors in the given data structure.

    :param data: The input data structure.
    :return: The unwrapped data structure.
    """
    if isinstance(data, Tensor):
        return data.data
    if isinstance(data, (list, tuple, deque)):
        # Deque converted to list to keep time on conversion, it's not change behavior for existed functions
        is_tuple = isinstance(data, tuple)
        data = list(data)
        for i, arg in enumerate(data):
            data[i] = _unwrap_tensors(arg)
        if is_tuple:
            data = tuple(data)
        return data
    if isinstance(data, dict):
        return {k: _unwrap_tensors(v) for k, v in data.items()}
    return data


def _wrap_output(ret_val: Any, ret_ann: Any) -> Any:
    """
    Wrap outputs of function according return annotation.

    :param ret_val: Return value of function.
    :param ret_ann: Return annotation of function.
    :return: Modified outputs.
    """
    if ret_ann is Tensor:
        return Tensor(ret_val)
    if ret_ann == list[Tensor]:
        return [Tensor(x) for x in ret_val]
    if ret_ann == tuple[Tensor, ...]:
        return tuple(Tensor(x) for x in ret_val)
    if get_origin(ret_ann) is tuple:
        return tuple(Tensor(x) if a is Tensor else x for x, a in zip(ret_val, get_args(ret_ann)))
    if ret_ann == dict[str, Tensor]:
        return {k: Tensor(v) for k, v in ret_val.items()}
    return ret_val


def get_numeric_backend_fn(fn_name: str, backend: TensorBackend) -> Callable[..., Any]:
    """
    Returns a numeric function based on the provided function name and backend type.

    :param fn_name: The name of the numeric function.
    :param backend: The backend type for which the function is required.
    :return: The backend-specific numeric function.
    """
    if backend == TensorBackend.numpy:
        from nncf.tensor.functions import numpy_numeric

        return getattr(numpy_numeric, fn_name)
    if backend == TensorBackend.torch:
        from nncf.tensor.functions import torch_numeric

        return getattr(torch_numeric, fn_name)
    if backend == TensorBackend.tf:
        from nncf.tensor.functions import tf_numeric

        return getattr(tf_numeric, fn_name)
    if backend == TensorBackend.ov:
        from nncf.tensor.functions import openvino_numeric

        return getattr(openvino_numeric, fn_name)
    msg = f"Unsupported backend type: {backend}"
    raise ValueError(msg)


def get_io_backend_fn(fn_name: str, backend: TensorBackend) -> Callable[..., Any]:
    """
    Returns a io function based on the provided function name and backend type.

    :param fn_name: The name of the numeric function.
    :param backend: The backend type for which the function is required.
    :return: The backend-specific io function.
    """
    if backend == TensorBackend.numpy:
        from nncf.tensor.functions import numpy_io

        return getattr(numpy_io, fn_name)
    if backend == TensorBackend.tf:
        from nncf.tensor.functions import tf_io

        return getattr(tf_io, fn_name)
    if backend == TensorBackend.torch:
        from nncf.tensor.functions import torch_io

        return getattr(torch_io, fn_name)
    msg = f"Unsupported backend type: {backend}"
    raise ValueError(msg)
