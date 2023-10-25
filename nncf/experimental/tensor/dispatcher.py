# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import types
import weakref
from abc import get_cache_token
from enum import Enum
from enum import auto
from functools import _find_impl
from functools import update_wrapper
from typing import Callable, List, Optional, Type, Union

from nncf.experimental.tensor import Tensor


class WrapperType(Enum):
    TensorToTensor = auto()
    TensorToAny = auto()
    TensorToList = auto()
    ListToTensor = auto()
    OnlyDispatch = auto()


def tensor_dispatch(wrapper_type: WrapperType = WrapperType.TensorToTensor) -> Callable:
    """Custom implementation of functools.singledispatch function decorator.

    Transforms a function into a generic function, which can have different
    behaviours depending upon the type of its first argument. The decorated
    function acts as the default implementation, and additional
    implementations can be registered using the register() attribute of the
    generic function.

    To control work with Tensors, different types of wrapper functions can be selected:
        TensorToTensor - expects Tensor as first argument, result will be wrapped to Tensor.
        TensorToAny - expects Tensor as first argument, result will not be wrapped to Tensor.
        TensorToList - expects Tensor as first argument, each element in result list will be wrapped to Tensor.
        ListToTensor - expects List of Tensors as first argument, result will be wrapped to Tensor.

    For not registered types will be raised NotImplementedError.

    In case of the first argument is not wrapped to Tensor will call backend specific function directory.

    :param wrapper_type: Type of wrapper function, defaults TensorToTensor.
    """

    def decorator(func: Callable) -> Callable:
        registry = {}
        dispatch_cache = weakref.WeakKeyDictionary()
        cache_token = None

        def dispatch(cls: Type) -> Callable:
            """generic_func.dispatch(cls) -> <function implementation>

            Runs the dispatch algorithm to return the best available implementation
            for the given *cls* registered on *generic_func*.
            """
            nonlocal cache_token
            if cache_token is not None:
                current_token = get_cache_token()
                if cache_token != current_token:
                    dispatch_cache.clear()
                    cache_token = current_token
            try:
                impl = dispatch_cache[cls]
            except KeyError:
                try:
                    impl = registry[cls]
                except KeyError:
                    impl = _find_impl(cls, registry)
                dispatch_cache[cls] = impl
            return impl

        def register(cls: Type, func: Optional[Callable] = None):
            """generic_func.register(cls, func) -> func

            Registers a new implementation for the given *cls* on a *generic_func*.

            """
            nonlocal cache_token
            if func is None:
                if isinstance(cls, type):
                    return lambda f: register(cls, f)
                ann = getattr(cls, "__annotations__", {})
                if not ann:
                    raise TypeError(
                        f"Invalid first argument to `register()`: {cls!r}. "
                        f"Use either `@register(some_class)` or plain `@register` "
                        f"on an annotated function."
                    )
                func = cls

                # only import typing if annotation parsing is necessary
                from typing import get_type_hints

                argname, cls = next(iter(get_type_hints(func).items()))
                if not isinstance(cls, type):
                    raise TypeError(f"Invalid annotation for {argname!r}. " f"{cls!r} is not a class.")
            registry[cls] = func
            if cache_token is None and hasattr(cls, "__abstractmethods__"):
                cache_token = get_cache_token()
            dispatch_cache.clear()
            return func

        def wrapper_tensor_to_tensor(tensor: Tensor, *args, **kw):
            args = tuple(x.data if isinstance(x, Tensor) else x for x in args)
            return Tensor(dispatch(tensor.data.__class__)(tensor.data, *args, **kw))

        def wrapper_tensor_to_any(tensor: Tensor, *args, **kw):
            args = tuple(x.data if isinstance(x, Tensor) else x for x in args)
            return dispatch(tensor.data.__class__)(tensor.data, *args, **kw)

        def wrapper_tensor_to_list(tensor: Tensor, *args, **kw):
            args = tuple(x.data if isinstance(x, Tensor) else x for x in args)
            return [Tensor(x) for x in dispatch(tensor.data.__class__)(tensor.data, *args, **kw)]

        def wrapper_list_to_tensor(list_of_tensors: List[Tensor], *args, **kw):
            list_of_tensors = [x.data for x in list_of_tensors]
            return Tensor(dispatch(list_of_tensors[0].__class__)(list_of_tensors, *args, **kw))

        wrappers_map = {
            WrapperType.TensorToTensor: wrapper_tensor_to_tensor,
            WrapperType.TensorToAny: wrapper_tensor_to_any,
            WrapperType.TensorToList: wrapper_tensor_to_list,
            WrapperType.ListToTensor: wrapper_list_to_tensor,
        }

        def raise_not_implemented(data: Union[Tensor, List[Tensor]], *args, **kw):
            """
            Raising NotImplementedError for not registered type.
            """
            if wrapper_type == WrapperType.ListToTensor:
                arg_type = type(data[0].data) if isinstance(data[0], Tensor) else type(data[0])
            else:
                arg_type = type(data.data) if isinstance(data, Tensor) else type(data)

            raise NotImplementedError(f"Function `{func.__name__}` is not implemented for {arg_type}")

        registry[object] = raise_not_implemented
        wrapper = wrappers_map[wrapper_type]
        wrapper.register = register
        wrapper.dispatch = dispatch
        wrapper.registry = types.MappingProxyType(registry)
        wrapper._clear_cache = dispatch_cache.clear
        update_wrapper(wrapper, func)
        return wrapper

    return decorator
