"""
 Copyright (c) 2021 Intel Corporation
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
from typing import Callable
from typing import Tuple
from typing import Union

from nncf.torch.utils import maybe_get_iterator


def nth(iterable, n, default=None):
    return next(islice(iterable, n, None), default)


class InputIndexEntry:
    def __init__(self, path: Tuple[Union[int, str], ...], getter: Callable, setter: Callable):
        self.path = path
        self.getter = getter
        self.setter = setter


class TupleRebuildingSetter:
    def __init__(self, idx_to_set, current_tuple, previous_level_setter_for_current_tuple):
        self._previous_level_setter = previous_level_setter_for_current_tuple
        self._current_tuple = current_tuple
        self._idx_to_set = idx_to_set

    def __call__(self, value):
        tmp_list = list(self._current_tuple)
        tmp_list[self._idx_to_set] = value
        new_tuple = tuple(tmp_list)
        self._current_tuple = new_tuple
        self._previous_level_setter(new_tuple)


class OperatorInput:
    def __init__(self, op_args, op_kwargs):
        self.op_args = op_args
        self.op_kwargs = op_kwargs
        self._index = OrderedDict()  # type: Dict[int, InputIndexEntry]

        op_args_index_entries = []
        self._nested_object_paths_generator(self.op_args, op_args_index_entries,
                                            previous_level_setter=partial(setattr, self, "op_args"))
        op_kwargs_index_entries = []
        self._nested_object_paths_generator(self.op_kwargs, op_kwargs_index_entries)

        # pylint:disable=unnecessary-comprehension
        self._index = {idx: entry for idx, entry in
                       enumerate(op_args_index_entries + op_kwargs_index_entries)}

    @staticmethod
    def _nested_object_paths_generator(obj, out_entries_list, path=(), memo=None, previous_level_setter=None):
        if memo is None:
            memo = set()
        iterator = maybe_get_iterator(obj)
        if iterator is not None:
            if id(obj) not in memo:
                memo.add(id(obj))
                current_level_getters = []
                current_level_setters = []
                for idx, iterval in enumerate(iterator(obj)):
                    path_component, value = iterval
                    current_level_getters.append(partial(obj.__getitem__, path_component))
                    if not isinstance(obj, tuple):
                        # `range` objects, for instance, have no __setitem__ and should be disregarded
                        if hasattr(obj, '__setitem__'):
                            current_level_setters.append(partial(obj.__setitem__, path_component))
                        else:
                            current_level_setters.append(None)
                    else:
                        current_level_setters.append(TupleRebuildingSetter(idx, obj, previous_level_setter))

                for idx, iterval in enumerate(iterator(obj)):
                    path_component, value = iterval
                    retval = OperatorInput._nested_object_paths_generator(value, out_entries_list,
                                                                          path + (path_component,), memo,
                                                                          current_level_setters[idx])
                    was_leaf = retval[1]
                    if was_leaf:
                        leaf_entry_path = retval
                        # getter = partial(obj.__getitem__, path_component)
                        getter = current_level_getters[idx]
                        setter = current_level_setters[idx]
                        if setter is not None:  # see note above about non-settable objects
                            out_entries_list.append(InputIndexEntry(leaf_entry_path,
                                                                    getter,
                                                                    setter))

                memo.remove(id(obj))
            is_leaf = False
            return path, is_leaf

        is_leaf = True
        return path, is_leaf

    def __iter__(self):
        return iter(self._index.values())

    def __getitem__(self, n):
        return self._index[n].getter()

    def __setitem__(self, n, value):
        self._index[n].setter(value)

    def __len__(self):
        return len(self._index)
