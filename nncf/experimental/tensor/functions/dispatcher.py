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
import functools
from typing import List

import numpy as np

from nncf.experimental.tensor import Tensor


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


def register_numpy_types(singledispatch_fn):
    """
    Decorator to register function to singledispatch for numpy classes.

    :param singledispatch_fn: singledispatch function.
    """

    def inner(func):
        singledispatch_fn.register(np.ndarray)(func)
        singledispatch_fn.register(np.generic)(func)
        return func

    return inner
