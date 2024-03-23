# Copyright (c) 2024 Intel Corporation
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
from pathlib import Path
from typing import Tuple

import numpy as np
import openvino.runtime as ov
from packaging import version

import nncf
from nncf import Dataset
from nncf.openvino.graph.nncf_graph_builder import GraphConverter
from tests.openvino.conftest import OPENVINO_NATIVE_TEST_ROOT
from tests.shared.nx_graph import compare_nx_graph_with_reference


def compare_nncf_graphs(model: ov.Model, path_ref_graph: str) -> None:
    nncf_graph = GraphConverter.create_nncf_graph(model)
    nx_graph = nncf_graph.get_graph_for_structure_analysis(extended=True)

    compare_nx_graph_with_reference(nx_graph, path_ref_graph, check_edge_attrs=True, unstable_node_names=True)


def dump_model(model: ov.Model, xml_path: str, bin_path: str):
    ov.serialize(model, xml_path, bin_path)


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


def get_openvino_major_minor_version() -> Tuple[int]:
    ov_version = ov.__version__
    pos = ov_version.find("-")
    if pos != -1:
        ov_version = ov_version[:pos]

    ov_version = version.parse(ov_version).base_version
    return tuple(map(int, ov_version.split(".")[:2]))


def get_openvino_version() -> str:
    major_verison, minor_version = get_openvino_major_minor_version()

    return f"{major_verison}.{minor_version}"


def get_actual_reference_for_current_openvino(rel_path: Path) -> Path:
    """
    Get path to actual reference file.

    :param rel_path: Relative path to reference file.

    :return: Path to reference file or raise RuntimeError.
    """
    root_dir = OPENVINO_NATIVE_TEST_ROOT / "data"
    current_ov_version = version.parse(get_openvino_version())

    def is_valid_version(dir_path: Path) -> bool:
        try:
            version.parse(dir_path.name)
        except version.InvalidVersion:
            return False
        return True

    ref_versions = filter(is_valid_version, root_dir.iterdir())
    ref_versions = sorted(ref_versions, key=lambda x: version.parse(x.name), reverse=True)
    ref_versions = filter(lambda x: version.parse(x.name) <= current_ov_version, ref_versions)

    for root_version in ref_versions:
        file_name = root_version / rel_path
        if file_name.is_file():
            return file_name

    raise nncf.InternalError(f"Not found file {root_dir}/{current_ov_version}/{rel_path}")
