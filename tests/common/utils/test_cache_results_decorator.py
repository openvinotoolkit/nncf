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
import pytest

from nncf.common.utils.caching import ResultsCache
from nncf.common.utils.caching import cache_results
from nncf.common.utils.caching import disable_results_caching

TEST_CACHE_CONTAINER = ResultsCache()


@cache_results(TEST_CACHE_CONTAINER)
def cached_addition(a, b):
    return a + b


CALL_SEQUENCE = [
    (
        (1, 2),
        False,
        3,
        False,
        1,
        {("cached_addition", frozenset({("a", 1), ("b", 2)})): 3},
        {("cached_addition", frozenset({("a", 1), ("b", 2)})): 0},
    ),
    (
        (1, 2),
        False,
        3,
        False,
        1,
        {("cached_addition", frozenset({("a", 1), ("b", 2)})): 3},
        {("cached_addition", frozenset({("a", 1), ("b", 2)})): 1},
    ),
    (
        (2, 3),
        True,
        5,
        False,
        1,
        {("cached_addition", frozenset({("a", 1), ("b", 2)})): 3},
        {("cached_addition", frozenset({("a", 1), ("b", 2)})): 1},
    ),
    (
        (3, 4),
        False,
        7,
        False,
        2,
        {
            ("cached_addition", frozenset({("a", 1), ("b", 2)})): 3,
            ("cached_addition", frozenset({("a", 3), ("b", 4)})): 7,
        },
        {
            ("cached_addition", frozenset({("a", 1), ("b", 2)})): 1,
            ("cached_addition", frozenset({("a", 3), ("b", 4)})): 0,
        },
    ),
    (
        (1, 2),
        False,
        3,
        False,
        2,
        {
            ("cached_addition", frozenset({("a", 1), ("b", 2)})): 3,
            ("cached_addition", frozenset({("a", 3), ("b", 4)})): 7,
        },
        {
            ("cached_addition", frozenset({("a", 1), ("b", 2)})): 2,
            ("cached_addition", frozenset({("a", 3), ("b", 4)})): 0,
        },
    ),
    (
        (3, 4),
        False,
        7,
        False,
        2,
        {
            ("cached_addition", frozenset({("a", 1), ("b", 2)})): 3,
            ("cached_addition", frozenset({("a", 3), ("b", 4)})): 7,
        },
        {
            ("cached_addition", frozenset({("a", 1), ("b", 2)})): 2,
            ("cached_addition", frozenset({("a", 3), ("b", 4)})): 1,
        },
    ),
    (
        (3, 4),
        True,
        7,
        False,
        2,
        {
            ("cached_addition", frozenset({("a", 1), ("b", 2)})): 3,
            ("cached_addition", frozenset({("a", 3), ("b", 4)})): 7,
        },
        {
            ("cached_addition", frozenset({("a", 1), ("b", 2)})): 2,
            ("cached_addition", frozenset({("a", 3), ("b", 4)})): 1,
        },
    ),
    ((3, 4), True, 7, True, 0, {}, {}),
    (
        (1, 2),
        False,
        3,
        False,
        1,
        {("cached_addition", frozenset({("a", 1), ("b", 2)})): 3},
        {("cached_addition", frozenset({("a", 1), ("b", 2)})): 0},
    ),
]


def test_caching_results():
    def check_fn():
        assert cached_addition(*inputs) == output
        assert len(TEST_CACHE_CONTAINER._cache) == cache_size
        assert TEST_CACHE_CONTAINER._cache == ref_cache
        assert TEST_CACHE_CONTAINER.access_count() == ref_access_count

    for inputs, disable_caching, output, clear_cache, cache_size, ref_cache, ref_access_count in CALL_SEQUENCE:
        if clear_cache:
            TEST_CACHE_CONTAINER.clear()

        if disable_caching:
            with disable_results_caching(TEST_CACHE_CONTAINER):
                check_fn()
        else:
            check_fn()


def test_disable_caching_with_exception():
    assert TEST_CACHE_CONTAINER.enabled()

    with pytest.raises(RuntimeError):
        with disable_results_caching(TEST_CACHE_CONTAINER):
            assert not TEST_CACHE_CONTAINER.enabled()
            raise RuntimeError

    assert TEST_CACHE_CONTAINER.enabled()
