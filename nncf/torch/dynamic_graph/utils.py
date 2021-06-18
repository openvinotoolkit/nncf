"""
 Copyright (c) 2019 Intel Corporation
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
from functools import partial
from typing import Tuple, Union, Callable

from graphviz import Digraph

from nncf.utils import maybe_get_iterator

graph_theme = {
    "background_color": "#FFFFFF",
    "fill_color": "#E8E8E8",
    "outline_color": "#000000",
    "font_color": "#000000",
    "font_name": "Times",
    "font_size": "10",
    "margin": "0,0",
    "padding": "1.0,1.0",
}


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


class InputIndexEntry:
    def __init__(self, path: Tuple[Union[int, str], ...], getter: Callable, setter: Callable):
        self.path = path
        self.getter = getter
        self.setter = setter


def nested_object_paths_generator(obj, out_entries_list, path=(), memo=None, previous_level_setter=None):
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
                    current_level_setters.append(partial(obj.__setitem__, path_component))
                else:
                    current_level_setters.append(TupleRebuildingSetter(idx, obj, previous_level_setter))

            for idx, iterval in enumerate(iterator(obj)):
                path_component, value = iterval
                retval = nested_object_paths_generator(value, out_entries_list,
                                                                     path + (path_component,), memo,
                                                                     current_level_setters[idx])
                was_leaf = retval[1]
                if was_leaf:
                    leaf_entry_path = retval
                    # getter = partial(obj.__getitem__, path_component)
                    getter = current_level_getters[idx]
                    setter = current_level_setters[idx]

                    out_entries_list.append(InputIndexEntry(leaf_entry_path,
                                                            getter,
                                                            setter))

            memo.remove(id(obj))
        is_leaf = False
        return path, is_leaf

    is_leaf = True
    return path, is_leaf


def draw_dot(context):
    graph = context.graph
    dot = Digraph()

    dot.attr("graph",
             bgcolor=graph_theme["background_color"],
             color=graph_theme["outline_color"],
             fontsize=graph_theme["font_size"],
             fontcolor=graph_theme["font_color"],
             fontname=graph_theme["font_name"],
             margin=graph_theme["margin"],
             # rankdir="LR",
             pad=graph_theme["padding"])
    dot.attr("node", shape="box",
             style="filled", margin="0,0",
             fillcolor=graph_theme["fill_color"],
             color=graph_theme["outline_color"],
             fontsize=graph_theme["font_size"],
             fontcolor=graph_theme["font_color"],
             fontname=graph_theme["font_name"])
    dot.attr("edge", style="solid",
             color=graph_theme["outline_color"],
             fontsize=graph_theme["font_size"],
             fontcolor=graph_theme["font_color"],
             fontname=graph_theme["font_name"])

    for node in graph.nodes:
        dot.node(graph.nodes[node]['name'])
        for child in graph.successors(node):
            dot.edge(node, child)
    return dot
