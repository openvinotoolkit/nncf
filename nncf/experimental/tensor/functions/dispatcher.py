# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from functools import _find_impl
from inspect import getfullargspec
from inspect import isclass
from inspect import isfunction
from types import MappingProxyType
from typing import List, _GenericAlias, _UnionGenericAlias, get_type_hints

from nncf.experimental.tensor import Tensor


def _get_target_types(type_alias):
    if isclass(type_alias):
        return [type_alias]
    if isinstance(type_alias, (_UnionGenericAlias, _GenericAlias)):
        ret = []
        for t in type_alias.__args__:
            ret.extend(_get_target_types(t))
        return ret


def tensor_dispatch(func):
    """
    This decorator creates a registry of functions for different types and provides a wrapper
    that calls the appropriate function based on the type of the first argument.
    It's particularly designed to handle Tensor inputs and outputs effectively.

    :param func: The function to be decorated.
    :return: The decorated function with type-based dispatching functionality.
    """

    registry = {}

    def dispatch(cls):
        """
        Retrieves the registered function for a given type.

        :param cls: The type to retrieve the function for.
        :return: The registered function for the given type, or a function that raises a NotImplementedError
            if no function is registered for type.
        """
        try:
            return registry[cls]
        except KeyError:
            return _find_impl(cls, registry)

    def register(rfunc):
        """Registers a function for a specific type or types.

        :param rfunc: The function to register.
        :return: The registered function.
        """
        assert isfunction(rfunc), "Register object should be a function."
        assert getfullargspec(func)[0] == getfullargspec(rfunc)[0], "Differ names of arguments of function"

        target_type_hint = get_type_hints(rfunc).get(getfullargspec(rfunc)[0][0])
        assert target_type_hint is not None, "No type hint for first argument of function"

        types_to_registry = set(_get_target_types(target_type_hint))

        for t in types_to_registry:
            assert t not in registry, f"{t} already registered for function"
            registry[t] = rfunc
        return rfunc

    def wrapper_tensor_to_tensor(*args, **kw):
        """
        Wrapper for functions that take and return a Tensor.
        This wrapper unwraps Tensor arguments and wraps the returned value in a Tensor if necessary.
        """
        is_wrapped = any(isinstance(x, Tensor) for x in args)
        args = tuple(x.data if isinstance(x, Tensor) else x for x in args)
        ret = dispatch(args[0].__class__)(*args, **kw)
        return Tensor(ret) if is_wrapped else ret

    def wrapper_tensor_to_any(*args, **kw):
        """
        Wrapper for functions that take a Tensor and return any type.
        This wrapper unwraps Tensor arguments but doesn't specifically wrap the returned value.
        """
        args = tuple(x.data if isinstance(x, Tensor) else x for x in args)
        return dispatch(args[0].__class__)(*args, **kw)

    def wrapper_tensor_to_list(*args, **kw):
        """
        Wrapper for functions that take a Tensor and return a list.
        This wrapper unwraps Tensor arguments and wraps the list elements as Tensors if necessary.
        """
        is_wrapped = any(isinstance(x, Tensor) for x in args)
        args = tuple(x.data if isinstance(x, Tensor) else x for x in args)
        ret = dispatch(args[0].__class__)(*args, **kw)
        if is_wrapped:
            return [Tensor(x) for x in ret]
        return ret

    def wrapper_list_to_tensor(list_of_tensors: List[Tensor], *args, **kw):
        """
        Wrapper for functions that take a list of Tensors and return a Tensor.
        This wrapper handles lists containing Tensors appropriately.
        """
        if any(isinstance(x, Tensor) for x in list_of_tensors):
            list_of_tensors = [x.data if isinstance(x, Tensor) else x for x in list_of_tensors]
            return Tensor(dispatch(list_of_tensors[0].__class__)(list_of_tensors, *args, **kw))
        return dispatch(list_of_tensors[0].__class__)(list_of_tensors, *args, **kw)

    def raise_not_implemented(*args, **kw):
        """
        Raises a NotImplementedError for types that are not registered.
        """
        if isinstance(args[0], list):
            arg_type = type(args[0][0].data) if isinstance(args[0][0], Tensor) else type(args[0][0])
        else:
            arg_type = type(args[0].data) if isinstance(args[0], Tensor) else type(args[0])

        raise NotImplementedError(f"Function `{func.__name__}` is not implemented for {arg_type}")

    # Select wrapper by signature of function
    type_hints = get_type_hints(func)
    first_type_hint = type_hints.get(getfullargspec(func)[0][0])
    return_type_hint = type_hints.get("return")
    wrapper = None
    if first_type_hint is Tensor:
        if return_type_hint is Tensor:
            wrapper = wrapper_tensor_to_tensor
        elif isinstance(return_type_hint, _GenericAlias) and not isinstance(return_type_hint, _UnionGenericAlias):
            wrapper = wrapper_tensor_to_list
        else:
            wrapper = wrapper_tensor_to_any
    elif isinstance(first_type_hint, _GenericAlias) and return_type_hint is Tensor:
        wrapper = wrapper_list_to_tensor

    assert wrapper is not None, (
        "Not supported signature of dispatch function, supported:\n"
        "   def foo(a: Tensor, ...) -> Tensor\n"
        "   def foo(a: Tensor, ...) -> Any\n"
        "   def foo(a: Tensor, ...) -> List[Tensor]\n"
        "   def foo(a: List[Tensor], ...) -> Tensor\n"
    )

    registry[object] = raise_not_implemented
    wrapper.register = register
    wrapper.dispatch = dispatch
    wrapper.registry = MappingProxyType(registry)

    return wrapper
