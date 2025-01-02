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

from functools import partial
from typing import Any, Callable, Dict, List, Mapping, Sequence, Set, Tuple, Type, Union

string_types = (str, bytes)
iteritems = lambda mapping: getattr(mapping, "iteritems", mapping.items)()


def to_tuple(lst: List, named_tuple_class: Type = None, named_tuple_fields: List[str] = None) -> Tuple:
    # Able to produce namedtuples if a corresponding parameter is given
    if named_tuple_fields is None:
        return tuple(lst)
    return named_tuple_class(*lst)


def is_tuple(obj) -> bool:
    return isinstance(obj, tuple)


def is_named_tuple(obj) -> bool:
    return is_tuple(obj) and (obj.__class__ is not tuple)


def maybe_get_iterator(obj):
    it = None

    if isinstance(obj, Mapping):
        it = iteritems

    elif isinstance(obj, (Sequence, Set)) and not isinstance(obj, string_types):
        it = enumerate
    return it


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


class NestedObjectIndex:
    def __init__(self, obj, path=(), memo=None, previous_level_setter=None):
        self._flat_nested_obj_indexing: List[InputIndexEntry] = []
        self._nested_object_paths_generator(obj, self._flat_nested_obj_indexing, path, memo, previous_level_setter)

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
                        if hasattr(obj, "__setitem__"):
                            current_level_setters.append(partial(obj.__setitem__, path_component))
                        else:
                            current_level_setters.append(None)
                    else:
                        current_level_setters.append(TupleRebuildingSetter(idx, obj, previous_level_setter))

                for idx, iterval in enumerate(iterator(obj)):
                    path_component, value = iterval
                    retval = NestedObjectIndex._nested_object_paths_generator(
                        value, out_entries_list, path + (path_component,), memo, current_level_setters[idx]
                    )
                    was_leaf = retval[1]
                    if was_leaf:
                        leaf_entry_path = retval
                        # getter = partial(obj.__getitem__, path_component)
                        getter = current_level_getters[idx]
                        setter = current_level_setters[idx]
                        if setter is not None:  # see note above about non-settable objects
                            out_entries_list.append(InputIndexEntry(leaf_entry_path, getter, setter))

                memo.remove(id(obj))
            is_leaf = False
            return path, is_leaf

        is_leaf = True
        return path, is_leaf

    def get_flat_nested_obj_indexing(self) -> List[InputIndexEntry]:
        return self._flat_nested_obj_indexing


def objwalk(obj, unary_predicate: Callable[[Any], bool], apply_fn: Callable, memo=None):
    """
    Walks through the indexable container hierarchy of obj and replaces all sub-objects matching a criterion
    with the result of a given function application.
    """

    if memo is None:
        memo = set()

    named_tuple_class = None
    named_tuple_fields = None
    if is_named_tuple(obj):
        named_tuple_class = obj.__class__

        named_tuple_fields = obj._fields

    was_tuple = is_tuple(obj)
    if was_tuple:
        obj = list(obj)

    iterator = maybe_get_iterator(obj)

    if iterator is not None:
        if id(obj) not in memo:
            memo.add(id(obj))
            indices_to_apply_fn_to = set()
            indices_vs_named_tuple_data: Dict[Any, Tuple[list, Type, List[str]]] = {}
            for idx, value in iterator(obj):
                next_level_it = maybe_get_iterator(value)
                if next_level_it is None:
                    if unary_predicate(value):
                        indices_to_apply_fn_to.add(idx)
                else:
                    if is_tuple(value):
                        processed_tuple = objwalk(value, unary_predicate, apply_fn, memo)
                        if is_named_tuple(value):
                            indices_vs_named_tuple_data[idx] = processed_tuple, value.__class__, value._fields
                        else:
                            indices_vs_named_tuple_data[idx] = processed_tuple, None, None
                    else:
                        objwalk(value, unary_predicate, apply_fn)
            for idx in indices_to_apply_fn_to:
                obj[idx] = apply_fn(obj[idx])
            for idx, tpl_data in indices_vs_named_tuple_data.items():
                tpl, n_tpl_class, n_tpl_fields = tpl_data
                obj[idx] = to_tuple(tpl, n_tpl_class, n_tpl_fields)

            memo.remove(id(obj))
    else:
        if unary_predicate(obj):
            return apply_fn(obj)

    if was_tuple:
        return to_tuple(obj, named_tuple_class, named_tuple_fields)

    return obj
