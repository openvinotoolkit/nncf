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

from tests.openvino.conftest import OPENVINO_NATIVE_TEST_ROOT
from tests.openvino.native.common import compare_nncf_graphs
from tests.openvino.native.models import SYNTHETIC_MODELS
from tests.openvino.omz_helpers import convert_model
from tests.openvino.omz_helpers import download_model

REFERENCE_GRAPHS_DIR = OPENVINO_NATIVE_TEST_ROOT / "data" / "reference_graphs" / "original_nncf_graph"


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
def test_compare_nncf_graph_omz_models(tmp_path, model_name):
    download_model(model_name, tmp_path)
    convert_model(model_name, tmp_path)
    model_path = tmp_path / "public" / model_name / "FP32" / f"{model_name}.xml"
    model = ov.Core().read_model(model_path)

    path_to_dot = REFERENCE_GRAPHS_DIR / f"{model_name}.dot"
    compare_nncf_graphs(model, path_to_dot)
