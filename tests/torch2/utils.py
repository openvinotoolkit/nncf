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

from pathlib import Path

import networkx as nx

from nncf.common.graph.graph import NNCFGraph


def compare_with_reference_file(text_data: str, ref_path: Path, regen_ref_data: bool):
    """
    Compares the given text data with the contents of a reference file.
    Optionally, the reference file can be regenerated with the provided text data.

    :param text_data: The actual data to compare against the reference file.
    :param ref_path: The path to the reference file that contains the expected data.
    :param regen_ref_data: If True, the reference file will be overwritten with the current text_data.
    """
    if regen_ref_data:
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        ref_path.write_text(text_data)
        return

    assert ref_path.exists(), f"Reference file does not exist: {ref_path}"
    ref_text_data = ref_path.read_text(encoding="utf-8")

    # Split lines for comparison without keeping newlines
    act = text_data.splitlines(keepends=False)
    ref = ref_text_data.splitlines(keepends=False)

    assert act == ref, (
        f"Data mismatch between actual data and reference file: {ref_path}\n"
        f"Actual data and reference data differ. Please review the file contents."
    )


def _quote_str(s: str) -> str:
    """
    Add quotes to a string if it contains a colon.
    """
    if ":" in s:
        return f'"{s}"'
    return s


def to_comparable_nx_graph(graph: NNCFGraph) -> nx.DiGraph:
    """
    Convert NNCFGraph to nx.DiGraph for comparison with references.

    Attributes:
        - NODE:
            - id
            - type
            - metatype class name

        - EDGE:
            - dtype
            - shape
            - out_port_id
            - in_port_id
            - parallel_input_port_ids (if exists)

    :param graph: NNCFGraph to convert.
    :return: Graph in nx.DiGraph.
    """
    out_graph = nx.DiGraph()
    for node in sorted(graph.get_all_nodes(), key=lambda x: x.node_id):
        attrs_node = {
            "id": node.node_id,
            "type": node.node_type,
            "metatype": node.metatype.__name__,
        }
        out_graph.add_node(_quote_str(node.node_name), **attrs_node)

    for edge in graph.get_all_edges():
        attrs_edge = {
            "dtype": edge.dtype.value,
            "shape": edge.tensor_shape,
            "out_port_id": edge.output_port_id,
            "in_port_id": edge.input_port_id,
        }
        if edge.parallel_input_port_ids:
            attrs_edge["parallel_input_port_ids"] = edge.parallel_input_port_ids

        out_graph.add_edge(_quote_str(edge.from_node.node_name), _quote_str(edge.to_node.node_name), **attrs_edge)
    return out_graph
