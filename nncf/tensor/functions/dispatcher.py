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
import functools
from typing import Callable, Dict, List

import numpy as np

import nncf
from nncf.tensor import Tensor
from nncf.tensor.definitions import TensorBackend


def tensor_guard(func: callable):
    """
    A decorator that ensures that the first argument to the decorated function is a Tensor.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if isinstance(args[0], Tensor):
            return func(*args, **kwargs)
        raise NotImplementedError(f"Function `{func.__name__}` is not implemented for {type(args[0])}")

    return wrapper


def dispatch_list(fn: "functools._SingleDispatchCallable", tensor_list: List[Tensor], *args, **kwargs):
    """
    Dispatches the function to the type of the wrapped data of the first element in tensor_list.

    :param fn: A function wrapped by `functools.singledispatch`.
    :param tensor_list: List of Tensors.
    :return: The result value of the function call.
    """
    unwrapped_list = [i.data for i in tensor_list]
    return fn.dispatch(type(unwrapped_list[0]))(unwrapped_list, *args, **kwargs)


def dispatch_dict(fn: "functools._SingleDispatchCallable", tensor_dict: Dict[str, Tensor], *args, **kwargs):
    """
    Dispatches the function to the type of the wrapped data of the any element in tensor_dict.

    :param fn: A function wrapped by `functools.singledispatch`.
    :param tensor_dict: Dict of Tensors.
    :return: The result value of the function call.
    """
    unwrapped_dict = {}
    tensor_backend = None
    for key, tensor in tensor_dict.items():
        if tensor_backend is None:
            tensor_backend = type(tensor.data)
        else:
            if tensor_backend is not type(tensor.data):
                raise nncf.InternalError("All tensors in the dictionary should have the same backend")
        unwrapped_dict[key] = tensor.data

    return fn.dispatch(tensor_backend)(unwrapped_dict, *args, **kwargs)


def register_numpy_types(singledispatch_fn):
    """
    Decorator to register function to singledispatch for numpy classes.

    :param singledispatch_fn: singledispatch function.
    """

    def inner(func):
        singledispatch_fn.register(np.ndarray)(func)
        singledispatch_fn.register(np.generic)(func)
        singledispatch_fn.register(float)(func)
        return func

    return inner


def get_numeric_backend_fn(fn_name: str, backend: TensorBackend) -> Callable:
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


def get_io_backend_fn(fn_name: str, backend: TensorBackend) -> Callable:
    """
    Returns a io function based on the provided function name and backend type.

    :param fn_name: The name of the numeric function.
    :param backend: The backend type for which the function is required.
    :return: The backend-specific io function.
    """
    if backend == TensorBackend.numpy:
        from nncf.tensor.functions import numpy_io

        return getattr(numpy_io, fn_name)
    if backend == TensorBackend.torch:
        from nncf.tensor.functions import torch_io

        return getattr(torch_io, fn_name)
