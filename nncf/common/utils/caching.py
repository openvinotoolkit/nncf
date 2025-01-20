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
from functools import wraps
from typing import Any, Callable, Dict


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

    def __getitem__(self, key: Any) -> Any:
        self._access_count[key] += 1
        return self._cache[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        self._access_count[key] = 0
        self._cache[key] = value

    def __contains__(self, key: Any) -> bool:
        return key in self._cache


def cache_results(cache: ResultsCacheContainer) -> Callable:  # type: ignore
    """
    Decorator to cache results of a function. When decorated function is called with the same set of arguments, it
    will return the cached result instead of recomputing it. If it was the first call with such set of arguments, the
    result will be computed and stored in the cache. The cache is stored in the `cache` object. Function arguments
    must be hashable.

    Decorated function additionally accepts a `disable_caching` argument do disable caching if needed. If it is True,
    the result will not be stored saved to a cache. Also, if there is a corresponding result in the cache, it will be
    recomputed.
    :param cache: A cache container where results will be stored.
    """

    def decorator(func: Callable) -> Callable:  # type: ignore
        @wraps(func)
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
