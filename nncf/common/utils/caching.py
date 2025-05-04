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
import copy
import inspect
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Iterator, TypeVar, cast


class ResultsCache:
    """
    A container for results decorated with @cache_results decorator.
    """

    def __init__(self) -> None:
        self._enabled = True
        # Stores the results of the decorated function
        self._cache: dict[Any, Any] = {}
        # Stores the number of times the cached result was accessed
        self._access_count: dict[Any, int] = {}

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def enabled(self) -> bool:
        return self._enabled

    def access_count(self) -> dict[Any, int]:
        return copy.deepcopy(self._access_count)

    def clear(self) -> None:
        self._cache.clear()
        self._access_count.clear()

    def __getitem__(self, key: Any) -> Any:
        self._access_count[key] += 1
        return self._cache[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        self._access_count[key] = 0
        self._cache[key] = value

    def __contains__(self, key: Any) -> bool:
        return key in self._cache


TFunc = TypeVar("TFunc", bound=Callable[..., Any])


def cache_results(cache: ResultsCache) -> Callable[[TFunc], TFunc]:
    """
    Decorator to cache results of a function. When decorated function is called with the same set of arguments, it
    will return the cached result instead of recomputing it. If it was the first call with such set of arguments, the
    result will be computed and stored in the cache. The cache is stored in the `cache` object. Function arguments
    must be hashable.

    :param cache: A cache container where results will be stored.
    """

    def decorator(func: TFunc) -> TFunc:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not cache.enabled():
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

        return cast(TFunc, wrapper)

    return decorator


@contextmanager
def disable_results_caching(cache: ResultsCache) -> Iterator[None]:
    """
    Context manager to disable caching of results for a block of code.

    :param cache: A cache container where results are stored.
    """
    should_reenable = cache.enabled()
    if should_reenable:
        cache.disable()
    try:
        yield
    finally:
        if should_reenable:
            cache.enable()
