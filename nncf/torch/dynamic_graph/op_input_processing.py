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
from collections import OrderedDict
from functools import partial
from itertools import islice
from typing import Dict
from typing import List

from nncf.torch.nested_objects_traversal import NestedObjectIndex
from nncf.torch.nested_objects_traversal import InputIndexEntry


def nth(iterable, n, default=None):
    return next(islice(iterable, n, None), default)


class OperatorInput:
    def __init__(self, op_args: List, op_kwargs: Dict):
        self.op_args = op_args
        self.op_kwargs = op_kwargs
        self._index = OrderedDict()  # type: Dict[int, InputIndexEntry]

        op_args_index_entries = NestedObjectIndex(self.op_args, previous_level_setter=partial(setattr, self, "op_args"))
        op_kwargs_index_entries = NestedObjectIndex(self.op_kwargs)

        # pylint:disable=unnecessary-comprehension
        self._index = {idx: entry for idx, entry in
                       enumerate(op_args_index_entries.get_flat_nested_obj_indexing() +
                                 op_kwargs_index_entries.get_flat_nested_obj_indexing())}

    def __iter__(self):
        return OperatorInputIterator(self)

    def __getitem__(self, n):
        return self._index[n].getter()

    def __setitem__(self, n, value):
        self._index[n].setter(value)

    def __len__(self):
        return len(self._index)


class OperatorInputIterator:
    def __init__(self, op_input: OperatorInput):
        self._op_input = op_input
        self._current_element_ordinal = 0

    def __next__(self):
        if self._current_element_ordinal >= len(self._op_input):
            raise StopIteration
        retval = self._op_input[self._current_element_ordinal]
        self._current_element_ordinal += 1
        return retval
