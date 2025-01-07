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

import openvino as ov
import pytest

from nncf.common.graph.graph import NNCFGraphEdge
from nncf.common.graph.layer_attributes import Dtype
from nncf.openvino.graph.nncf_graph_builder import GraphConverter
from tests.openvino.native.common import compare_nncf_graphs
from tests.openvino.native.common import convert_torch_model
from tests.openvino.native.common import get_actual_reference_for_current_openvino
from tests.openvino.native.models import SYNTHETIC_MODELS
from tests.openvino.native.models import ParallelEdgesModel
from tests.openvino.native.models import get_torch_model_info

REFERENCE_GRAPHS_DIR = Path("reference_graphs") / "original_nncf_graph"


@pytest.mark.parametrize("model_cls_to_test", SYNTHETIC_MODELS.values())
def test_compare_nncf_graph_synthetic_models(model_cls_to_test):
    model_to_test = model_cls_to_test()
    path_to_dot = get_actual_reference_for_current_openvino(REFERENCE_GRAPHS_DIR / model_to_test.ref_graph_name)
    compare_nncf_graphs(model_to_test.ov_model, path_to_dot)


@pytest.mark.parametrize(
    "model_name",
    (
        "mobilenet-v2",
        "mobilenet-v3-small",
        "resnet-18",
        "inception-v3",
        "ssd-mobilenet",
    ),
)
def test_compare_nncf_graph_real_models(tmp_path, model_name):
    model_cls, input_shape = get_torch_model_info(model_name)
    ov_model = convert_torch_model(model_cls(), input_shape, tmp_path)
    path_to_dot = get_actual_reference_for_current_openvino(REFERENCE_GRAPHS_DIR / f"{model_name}.dot")
    compare_nncf_graphs(ov_model, path_to_dot)


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


@pytest.mark.parametrize(
    "ov_type,expected_nncf_dtype",
    [
        (ov.Type.f16, Dtype.FLOAT),
        (ov.Type.f32, Dtype.FLOAT),
        (ov.Type.f64, Dtype.FLOAT),
        (ov.Type.i4, Dtype.INTEGER),
        (ov.Type.i8, Dtype.INTEGER),
        (ov.Type.i16, Dtype.INTEGER),
        (ov.Type.i32, Dtype.INTEGER),
        (ov.Type.i64, Dtype.INTEGER),
        (ov.Type.u1, Dtype.INTEGER),
        (ov.Type.u4, Dtype.INTEGER),
        (ov.Type.u8, Dtype.INTEGER),
        (ov.Type.u16, Dtype.INTEGER),
        (ov.Type.u32, Dtype.INTEGER),
        (ov.Type.u64, Dtype.INTEGER),
        (ov.Type.boolean, Dtype.INTEGER),
        (ov.Type.string, Dtype.INTEGER),
    ],
)
def test_convert_to_nncf_dtype_supported_types(ov_type: ov.Type, expected_nncf_dtype: Dtype):
    actual_nncf_dtype = GraphConverter.convert_to_nncf_dtype(ov_type)
    assert actual_nncf_dtype == expected_nncf_dtype


@pytest.mark.parametrize(
    "ov_type",
    [
        ov.Type.nf4,
        ov.Type.undefined,
        ov.Type.f8e4m3,
        ov.Type.f8e5m2,
    ],
)
def test_convert_to_nncf_dtype_unsupported_types(ov_type: ov.Type):
    with pytest.raises(NotImplementedError):
        _ = GraphConverter.convert_to_nncf_dtype(ov_type)
