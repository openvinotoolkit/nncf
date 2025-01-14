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
from importlib import import_module
from typing import Any, Callable, Dict, List

from nncf.common.logging import nncf_logger

IMPORTED_DEPENDENCIES: Dict[str, bool] = {}


def skip_if_dependency_unavailable(dependencies: List[str]) -> Callable[[Callable[..., None]], Callable[..., None]]:
    """
    Decorator factory to skip a noreturn function if dependencies are not met.

    :param dependencies: A list of dependencies
    :return: A decorator
    """

    def wrap(func: Callable[..., None]) -> Callable[..., None]:
        def wrapped_f(*args: Any, **kwargs: Any):  # type: ignore
            for libname in dependencies:
                if libname in IMPORTED_DEPENDENCIES:
                    if IMPORTED_DEPENDENCIES[libname]:
                        continue
                    break
                try:
                    _ = import_module(libname)
                    IMPORTED_DEPENDENCIES[libname] = True
                except ImportError as ex:
                    nncf_logger.warning(
                        f"{ex.msg} Please install NNCF package with plots "
                        "extra. Use one of the following commands "
                        '"pip install .[plots]" running from the repository '
                        'root directory or "pip install nncf[plots]"'
                    )
                    IMPORTED_DEPENDENCIES[libname] = False
                    break
            else:
                return func(*args, **kwargs)
            return None

        return wrapped_f

    return wrap


class ResultsCacheContainer:
    """
    A container for results decorated with @cache_results decorator.
    """

    def __init__(self) -> None:
        # Stores the results of the decorated function
        self._cache: Dict[Any, Any] = {}
        # Stores the number of times the cached result was accessed
        self._access_count: Dict[Any, int] = {}

    def clear(self) -> None:
        self._cache.clear()
        self._access_count.clear()

    def is_empty(self) -> bool:
        return len(self._cache) == 0

    def __getitem__(self, item: Any) -> Any:
        self._access_count[item] += 1
        return self._cache[item]

    def __setitem__(self, key: Any, value: Any) -> None:
        self._access_count[key] = 0
        self._cache[key] = value

    def __contains__(self, item: Any) -> bool:
        return item in self._cache


def cache_results(cache: ResultsCacheContainer) -> Callable:  # type: ignore
    """
    Decorator to cache the results of a function.

    Decorated function additionally accepts a `disable_caching` argument do disable caching if needed. If it is True,
    the result will not be stored saved to a cache. Also, if there is a corresponding result in the cache, it will be
    recomputed.
    :param cache: A cache container where results will be stored.
    """

    def decorator(func: Callable) -> Callable:  # type: ignore
        def wrapper(*args, disable_caching: bool = False, **kwargs) -> Any:  # type: ignore
            if disable_caching:
                return func(*args, **kwargs)
            sig = inspect.signature(func)
            new_kwargs = {name: arg for name, arg in zip(sig.parameters, args)}
            new_kwargs.update(kwargs)
            cache_key = (func.__name__, frozenset(new_kwargs.items()))
            if cache_key in cache:
                return cache[cache_key]
            result = func(*args, **kwargs)
            cache[cache_key] = result
            return result

        return wrapper

    return decorator
