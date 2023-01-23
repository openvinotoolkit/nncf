"""
 Copyright (c) 2023 Intel Corporation
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
import os
import json
from copy import deepcopy
from pathlib import Path

import networkx as nx
import numpy as np
import openvino.runtime as ov

from nncf import Dataset
from nncf.common.utils.dot_file_rw import read_dot_graph
from nncf.common.utils.dot_file_rw import write_dot_graph
from nncf.experimental.openvino_native.graph.nncf_graph_builder import GraphConverter
from tests.common.graph.nx_graph import sort_dot


def compare_nncf_graphs(model: ov.Model, path_ref_graph: str) -> None:
    nncf_graph = GraphConverter.create_nncf_graph(model)
    nx_graph = nncf_graph.get_graph_for_structure_analysis(extended=True)

    compare_ov_nx_graph_with_reference(nx_graph, path_ref_graph, check_edge_attrs=True)


def compare_ov_nx_graph_with_reference(nx_graph: nx.DiGraph, path_to_dot: str,
                                       sort_dot_graph=True, check_edge_attrs: bool = False) -> None:
    """
    Checks whether the two nx.DiGraph are identical. The first one is 'nx_graph' argument
    and the second graph is read from the absolute path - 'path_to_dot'.
    Also, could dump the graph, based in the global variable NNCF_TEST_REGEN_DOT.
    If 'sort_dot_graph' is True sorts the second graph before dumping.
    If 'check_edge_attrs' is True checks edge attributes of the graphs.
    :param nx_graph: The first nx.DiGraph.
    :param path_to_dot: The absolute path to the second nx.DiGraph.
    :param sort_dot_graph: whether to call sort_dot() function on the second graph.
    :param check_edge_attrs: whether to check edge attributes of the graphs.
    :return: None
    """
    dot_dir = Path(path_to_dot).parent
    # validate .dot file manually!
    if os.getenv("NNCF_TEST_REGEN_DOT") is not None:
        if not os.path.exists(dot_dir):
            os.makedirs(dot_dir)
        write_dot_graph(nx_graph, path_to_dot)
        if sort_dot_graph:
            sort_dot(path_to_dot)

    expected_graph = nx.DiGraph(read_dot_graph(path_to_dot))
    check_openvino_nx_graph(nx_graph, expected_graph, check_edge_attrs)


def check_openvino_nx_graph(nx_graph: nx.DiGraph, expected_graph: nx.DiGraph, check_edge_attrs: bool = False) -> None:
    attrs = {}
    expected_attrs = {}
    for node_attrs in nx_graph.nodes.values():
        node_id = int(node_attrs['id'])
        attrs[node_id] = {k: str(v) for k, v in node_attrs.items()}

    for node_attrs in expected_graph.nodes.values():
        node_id = int(node_attrs['id'])
        expected_attrs[node_id] = {k: str(v).strip('"') for k, v in node_attrs.items()}

    for attr_name, expected_attr in expected_attrs.items():
        assert attr_name in attrs, f'Not found {attr_name} in attributes.'
        assert expected_attr == attrs[attr_name], \
            f'Incorrect attribute value for {attr_name}.' \
            f' expected {expected_attr}, but actual {attrs[attr_name]}.'

    edges = {}
    for edge in nx_graph.edges:
        nx_edge_attrs = nx_graph.edges[edge]
        if isinstance(nx_edge_attrs, dict):
            nx_edge_attrs['label'] = str(nx_edge_attrs['label'])

        src, dst = edge
        src = src.split(" ")[0]
        dst = dst.split(" ")[0]
        simplified_edge = f'{src} -> {dst}'
        edges[simplified_edge] = nx_edge_attrs

    expected_edges = {}
    for edge in expected_graph.edges:
        expected_graph_edge_attrs = expected_graph.edges[edge]
        if not isinstance(expected_graph_edge_attrs['label'], list):
            expected_graph_edge_attrs['label'] = expected_graph_edge_attrs['label'].replace('"', '')
        else:
            expected_graph_edge_attrs['label'] = str(expected_graph_edge_attrs['label'])

        src, dst = edge
        src = src.split(" ")[0]
        dst = dst.split(" ")[0]
        simplified_edge = f'{src} -> {dst}'
        expected_edges[simplified_edge] = expected_graph_edge_attrs

    if check_edge_attrs:
        for edge_name, expected_edge in expected_edges.items():
            assert edge_name in edges, f'{edge_name} not found in edges.'
            assert expected_edge == edges[edge_name], \
                f'Incorrect edge attributes for {edge_name}.' \
                f' expected {expected_edge}, but actual {edges[edge_name]}.'
    else:
        assert edges.keys() == expected_edges.keys()


def get_dataset_for_test(model):
    rng = np.random.default_rng(seed=0)
    input_data = {}
    for param in model.get_parameters():
        input_shape = param.get_output_shape(0)
        input_data[param.get_friendly_name()] = rng.uniform(0, 1, input_shape)

    dataset = Dataset([input_data])
    return dataset


def load_json(stats_path):
    with open(stats_path, 'r', encoding='utf8') as json_file:
        return json.load(json_file)


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    # pylint: disable=W0221, E0202

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return json.JSONEncoder.default(self, o)


def dump_to_json(local_path, data):
    with open(local_path, 'w', encoding='utf8') as file:
        json.dump(deepcopy(data), file, indent=4, cls=NumpyEncoder)
