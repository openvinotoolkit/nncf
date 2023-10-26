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

import openvino.runtime as ov
import pytest

from nncf.common.graph.graph import NNCFGraphEdge
from nncf.common.graph.layer_attributes import Dtype
from nncf.openvino.graph.nncf_graph_builder import GraphConverter
from tests.openvino.conftest import OPENVINO_NATIVE_TEST_ROOT
from tests.openvino.native.common import compare_nncf_graphs
from tests.openvino.native.common import get_openvino_version
from tests.openvino.native.models import SYNTHETIC_MODELS
from tests.openvino.native.models import ParallelEdgesModel
from tests.openvino.omz_helpers import convert_model
from tests.openvino.omz_helpers import download_model

OV_VERSION = get_openvino_version()
REFERENCE_GRAPHS_DIR = OPENVINO_NATIVE_TEST_ROOT / "data" / OV_VERSION / "reference_graphs" / "original_nncf_graph"


@pytest.mark.parametrize("model_cls_to_test", SYNTHETIC_MODELS.values())
def test_compare_nncf_graph_synthetic_models(model_cls_to_test):
    model_to_test = model_cls_to_test()
    path_to_dot = REFERENCE_GRAPHS_DIR / model_to_test.ref_graph_name
    compare_nncf_graphs(model_to_test.ov_model, path_to_dot)


OMZ_MODELS = [
    "mobilenet-v2-pytorch",
    "mobilenet-v3-small-1.0-224-tf",
    "resnet-18-pytorch",
    "googlenet-v3-pytorch",
    "ssd_mobilenet_v1_coco",
    "yolo-v4-tiny-tf",
]


@pytest.mark.parametrize("model_name", OMZ_MODELS)
def test_compare_nncf_graph_omz_models(tmp_path, omz_cache_dir, model_name):
    download_model(model_name, tmp_path, omz_cache_dir)
    convert_model(model_name, tmp_path)
    model_path = tmp_path / "public" / model_name / "FP32" / f"{model_name}.xml"
    model = ov.Core().read_model(model_path)

    path_to_dot = REFERENCE_GRAPHS_DIR / f"{model_name}.dot"
    compare_nncf_graphs(model, path_to_dot)


def test_parallel_edges():
    def _get_default_nncf_graph_edge(from_node, to_node, input_port_id, output_port_id):
        return NNCFGraphEdge(
            from_node,
            to_node,
            input_port_id=input_port_id,
            output_port_id=output_port_id,
            tensor_shape=[1, 3, 3],
            dtype=Dtype.FLOAT,
            parallel_input_port_ids=[],
        )

    model = ParallelEdgesModel().ov_model
    nncf_graph = GraphConverter.create_nncf_graph(model)
    input_node = nncf_graph.get_node_by_name("Input")
    mm_node = nncf_graph.get_node_by_name("Mm")
    add_node = nncf_graph.get_node_by_name("Add")
    ref_input_edges = {
        _get_default_nncf_graph_edge(
            input_node,
            mm_node,
            input_port_id=0,
            output_port_id=0,
        ),
        _get_default_nncf_graph_edge(
            input_node,
            mm_node,
            input_port_id=1,
            output_port_id=0,
        ),
    }
    ref_output_edges = ref_input_edges.copy()
    ref_output_edges.add(
        _get_default_nncf_graph_edge(
            input_node,
            add_node,
            input_port_id=0,
            output_port_id=0,
        )
    )
    assert set(nncf_graph.get_input_edges(mm_node)) == ref_input_edges
    assert set(nncf_graph.get_output_edges(input_node)) == ref_output_edges
