# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
from copy import deepcopy

import numpy as np
import openvino.runtime as ov

from nncf import Dataset
from nncf.openvino.graph.nncf_graph_builder import GraphConverter
from tests.shared.nx_graph import compare_nx_graph_with_reference


def compare_nncf_graphs(model: ov.Model, path_ref_graph: str) -> None:
    nncf_graph = GraphConverter.create_nncf_graph(model)
    nx_graph = nncf_graph.get_graph_for_structure_analysis(extended=True)

    compare_nx_graph_with_reference(nx_graph, path_ref_graph, check_edge_attrs=True, unstable_node_names=True)


def get_dataset_for_test(model):
    rng = np.random.default_rng(seed=0)
    input_data = {}
    for param in model.get_parameters():
        input_shape = param.partial_shape.get_max_shape()
        input_data[param.get_output_tensor(0).get_any_name()] = rng.uniform(0, 1, input_shape)

    dataset = Dataset([input_data])
    return dataset


def load_json(stats_path):
    with open(stats_path, "r", encoding="utf8") as json_file:
        return json.load(json_file)


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

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
    with open(local_path, "w", encoding="utf8") as file:
        json.dump(deepcopy(data), file, indent=4, cls=NumpyEncoder)
