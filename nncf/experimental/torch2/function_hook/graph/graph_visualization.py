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

import hashlib
from enum import Enum
from typing import Any, Dict, Tuple

import networkx as nx  # type: ignore[import-untyped]
import pydot  # type: ignore[import-untyped]

from nncf.experimental.torch2.function_hook.graph.graph_utils import ConstMeta
from nncf.experimental.torch2.function_hook.graph.graph_utils import EdgeMeta
from nncf.experimental.torch2.function_hook.graph.graph_utils import FunctionMeta
from nncf.experimental.torch2.function_hook.graph.graph_utils import InOutMeta


class PydotStyleTemplate(Enum):
    """
    Enum to define different styles for Pydot graph representation.
     - disable: labels contain only names, used for tests  (not recommend to convert to svg)
     - short: labels contain names, add edge info
     - full: labels contain all full information about nodes
    """

    disable = "disable"
    short = "short"
    full = "full"

    def __str__(self) -> str:
        return self.value


def fix_dot_label(label: str) -> str:
    """
    Escapes curly braces in a DOT label to avoid syntax errors.

    :param label: The label string to be fixed.
    :return: The label with escaped curly braces.
    """
    return label.replace("{", r"\{").replace("}", r"\}")


def args_to_label(args: Tuple[Any, ...]) -> str:
    """
    Converts function arguments to a formatted string label.

    :param args: Function arguments.
    :return: Formatted string label of arguments.
    """
    if not args:
        return "[]"
    ret = "["
    for arg in args:
        ret += f"\n{arg},"
    return ret + "\n]"


def kwargs_to_label(kwargs: Dict[str, Any]) -> str:
    """
    Converts function keyword arguments to a formatted string label.

    :param kwargs: Function keyword arguments.
    :return: Formatted string label of keyword arguments.
    """
    if not kwargs:
        return "{}"
    ret = "{"
    for key, val in kwargs.items():
        ret += f"\n{key} : {str(val)[:50]}"
    return ret + "\n}"


def get_label_from_node_data(node_data: Dict[str, Any], style: PydotStyleTemplate) -> str:
    """
    Generates a label for a graph node based on its metadata and the desired style.

    :param node_data: Metadata of the node.
    :param style: Style template to determine the label format.
    :return: Formatted label for the node.
    """
    meta = node_data["meta"]
    node_type = node_data["type"]
    if style == PydotStyleTemplate.full:
        rows = []
        if isinstance(meta, InOutMeta):
            rows = [
                f"type: {node_type}",
                f"name: {meta.name}",
                f"dtype: {meta.dtype}",
                f"shape: {meta.shape}",
            ]
        elif isinstance(meta, ConstMeta):
            rows = [
                f"type: {node_type}",
                f"name: {meta.name_in_model}",
                f"dtype: {meta.dtype}",
                f"shape: {meta.shape}",
            ]
        if isinstance(meta, FunctionMeta):
            rows = [
                f"type: {node_type}",
                f"op_name: {meta.op_name}",
                f"fn_name: {meta.fn_name}",
                f"args: {args_to_label(meta.args)}",
                f"kwargs: {kwargs_to_label(meta.kwargs)}",
            ]
        return "{" + fix_dot_label("|".join(rows)) + "}"
    else:
        if isinstance(meta, InOutMeta):
            return f"{meta.name}"
        if isinstance(meta, ConstMeta):
            return f"{meta.name_in_model}"
        if isinstance(meta, FunctionMeta):
            return f"{meta.op_name}"
    raise ValueError(f"Unknown meta node {type(meta)}")


def get_label_from_edge_data(node_data: Dict[str, Any], style: PydotStyleTemplate) -> str:
    """
    Generates a label for a graph edge based on its metadata and the desired style.

    :param edge_data: Metadata of the edge.
    :param style: Style template to determine the label format.
    :return: Formatted label for the edge.
    """
    meta = node_data["meta"]
    assert isinstance(meta, EdgeMeta)

    if style == PydotStyleTemplate.disable:
        return f"{meta.output_port} → {meta.input_port}"
    else:
        return f"{meta.shape}\n{meta.output_port} → {meta.input_port}"


_RAINBOW_COLORS = [
    "#ffadad",
    "#ffc2a9",
    "#ffd6a5",
    "#fdffb6",
    "#caffbf",
    "#b3fbdf",
    "#aae0ef",
    "#a0c4ff",
    "#bdb2ff",
    "#ffc6ff",
]


def color_picker(data: str) -> str:
    """
    Picks a color from a predefined set of colors based on the input string.

    :param data: Input string to determine the color.
    :return: Hex code of the selected color.
    """
    data = "".join(d for d in data if d.isalpha())
    hash_object = hashlib.sha256(data.encode())
    hex_int = int(hash_object.hexdigest()[:6], 16)
    return _RAINBOW_COLORS[hex_int % len(_RAINBOW_COLORS)]


def get_style(node: Dict[str, Any], style: PydotStyleTemplate) -> Dict[str, str]:
    """
    Generates a style dictionary for a graph node based on its metadata and the desired style.

    :param node: Metadata of the node.
    :param style: Style template to determine the node style.
    :return: Dictionary containing style attributes for the node.
    """
    if style == PydotStyleTemplate.disable:
        return {}
    meta = node["meta"]

    if isinstance(meta, InOutMeta):
        return {
            "fillcolor": "#adadad",
            "fontcolor": "#000000",
            "shape": "record",
            "style": '"filled,rounded"',
        }
    if isinstance(meta, ConstMeta):
        return {
            "fillcolor": "#ffffff",
            "fontcolor": "#000000",
            "shape": "record",
            "style": '"filled,rounded"',
        }
    if isinstance(meta, FunctionMeta):
        return {
            "fillcolor": color_picker(meta.fn_name),
            "fontcolor": "#000000",
            "shape": "record",
            "style": '"filled,rounded"',
        }

    raise ValueError(f"Unknown meta node {type(meta)}")


def to_pydot(nx_graph: nx.MultiDiGraph, style_template: PydotStyleTemplate = PydotStyleTemplate.full) -> pydot.Graph:
    """
    Converts a NetworkX directed graph to a Pydot graph with specified styling.

    :param nx_graph: Input NetworkX directed graph.
    :param style_template: Style template to determine node and edge styles.
    :return: Pydot graph representation of the input NetworkX graph.
    """
    dot_graph = pydot.Dot("", rankdir="TB")

    for key, data in nx_graph.nodes(data=True):
        style = get_style(data, style_template)
        dot_node = pydot.Node(key, label=get_label_from_node_data(data, style_template), **style)
        dot_graph.add_node(dot_node)

    for key_from, key_to, data in nx_graph.edges(data=True):
        dot_edge = pydot.Edge(key_from, key_to, label=get_label_from_edge_data(data, style_template))
        dot_graph.add_edge(dot_edge)

    return dot_graph
