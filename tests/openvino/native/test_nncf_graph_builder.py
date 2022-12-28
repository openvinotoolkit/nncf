"""
 Copyright (c) 2022 Intel Corporation
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

import pytest
import openvino.runtime as ov

from nncf.experimental.openvino_native.graph.nncf_graph_builder import GraphConverter

from tests.common.graph.nx_graph import compare_nx_graph_with_reference
from tests.openvino.conftest import OPENVINO_TEST_ROOT
from tests.openvino.omz_helpers import convert_model
from tests.openvino.omz_helpers import download_model
from tests.openvino.native.models import SYNTHETIC_MODELS

REFERENCE_GRAPHS_DIR = OPENVINO_TEST_ROOT / 'data' / 'reference_graphs' / 'original_nncf_graph'


@pytest.mark.skip(reason="Enable after fixing an issue with operation outputs order")
@pytest.mark.parametrize("model_cls_to_test", SYNTHETIC_MODELS.values())
def test_compare_nncf_graph_synthetic_models(model_cls_to_test):
    model_to_test = model_cls_to_test()
    path_to_dot = REFERENCE_GRAPHS_DIR / 'synthetic' / model_to_test.ref_graph_name

    nncf_graph = GraphConverter.create_nncf_graph(model_to_test.ov_model)
    nx_graph = nncf_graph.get_graph_for_structure_analysis(extended=True)
    compare_nx_graph_with_reference(nx_graph, path_to_dot, check_edge_attrs=True)


OMZ_MODELS = [
    'mobilenet-v2-pytorch',
    'mobilenet-v3-small-1.0-224-tf',
    'resnet-18-pytorch',
    'googlenet-v3-pytorch',

    'ssd_mobilenet_v1_coco',
    'yolo-v4-tiny-tf',
]

@pytest.mark.skip(reason="Enable after fixing an issue with operation outputs order")
@pytest.mark.parametrize('model_name', OMZ_MODELS)
def test_compare_nncf_graph_omz_models(tmp_path, model_name):
    _ = download_model(model_name, tmp_path)
    model_path = convert_model(model_name, tmp_path)
    model = ov.Core().read_model(model_path)

    nncf_graph = GraphConverter.create_nncf_graph(model)
    nx_graph = nncf_graph.get_graph_for_structure_analysis(extended=True)

    path_to_dot = REFERENCE_GRAPHS_DIR / 'omz' / f'{model_name}.dot'
    compare_nx_graph_with_reference(nx_graph, path_to_dot, check_edge_attrs=True)
