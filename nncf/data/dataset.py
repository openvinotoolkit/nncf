"""
 Copyright (c) 2022 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from typing import Iterable
from typing import Any
from typing import Callable
from typing import Optional
from typing import List


class Dataset:

    def __init__(self,
                 data_source: Iterable[Any],
                 transform_func: Optional[Callable[[Any], Any]] = None):
        self._data_source = data_source
        self._transform_func = transform_func

    def get_data(self, indices: Optional[List[int]] = None) -> Iterable[Any]:
        return DataProvider(self._data_source, None, indices)

    def get_inference_data(self, indices: Optional[List[int]] = None) -> Iterable[Any]:
        return DataProvider(self._data_source, self._transform_func, indices)


class DataProvider:
    def __init__(self,
                 data_source: Iterable[Any],
                 transform_func: Callable[[Any], Any],
                 indices: Optional[Iterable[int]] = None):
        self._data_source = data_source
        if transform_func is None:
            transform_func = lambda x: x
        self._transform_func = transform_func
        self._indices = indices

    def __iter__(self):
        if self._indices is None:
            return map(self._transform_func, self._data_source)

        if hasattr(self._data_source, '__getitem__'):
            return DataProvider._get_iterator_for_map_style(self._data_source, self._transform_func, self._indices)

        return DataProvider._get_iterator_for_iter(self._data_source, self._transform_func, self._indices)

    @staticmethod
    def _get_iterator_for_map_style(data_source, transform_func, indices):
        for index in indices:
            yield transform_func(data_source[index])

    @staticmethod
    def _get_iterator_for_iter(data_source, transform_func, indices):
        pos = 0
        num_indices = len(indices)
        for idx, data_item in enumerate(data_source):
            if pos == num_indices:
                # All specified data items were selected.
                break
            if idx == indices[pos]:
                pos = pos + 1
                yield transform_func(data_item)
