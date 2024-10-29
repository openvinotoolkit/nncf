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

import inspect
from collections import defaultdict


class ResultsCacheContainer:
    def __init__(self):
        self._cache = {}
        self._access_count = {}

    def clear(self):
        self._cache.clear()
        self._access_count.clear()

    def is_empty(self):
        return len(self._cache) == 0

    def __getitem__(self, item):
        self._access_count[item] += 1
        return self._cache[item]

    def __setitem__(self, key, value):
        self._access_count[key] = 0
        self._cache[key] = value

    def __contains__(self, item):
        return item in self._cache


def cache_results(cache: ResultsCacheContainer):
    def decorator(func):
        def wrapper(*args, disable_caching=False, **kwargs):
            sig = inspect.signature(func)
            new_kwargs = {name: arg for name, arg in zip(sig.parameters, args)}
            new_kwargs.update(kwargs)
            cache_key = (func.__name__, frozenset(new_kwargs.items()))
            if cache_key in cache:
                return cache[cache_key]
            result = func(*args, **kwargs)
            if not disable_caching:
                cache[cache_key] = result
            return result

        return wrapper

    return decorator
