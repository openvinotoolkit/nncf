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

import onnx
import pytest

from nncf.scopes import IgnoredScope
from tests.onnx.conftest import ONNX_MODEL_DIR
from tests.onnx.quantization.common import ModelToTest
from tests.onnx.quantization.common import compare_nncf_graph
from tests.onnx.quantization.common import min_max_quantize_model
from tests.onnx.quantization.common import mock_collect_statistics
from tests.onnx.weightless_model import load_model_topology_with_zeros_weights

TEST_DATA = [
    ModelToTest("ssd_mobilenet_v1_12", [1, 300, 300, 3]),
    ModelToTest("ssd-12", [1, 3, 1200, 1200]),
    ModelToTest("yolov2-coco-9", [1, 3, 416, 416]),
    ModelToTest("MaskRCNN-12", [3, 1200, 800]),
    ModelToTest("retinanet-9", [1, 3, 480, 640]),
    ModelToTest("fcn-resnet50-12", [1, 3, 480, 640]),
]


@pytest.mark.parametrize(("model_to_test"), TEST_DATA, ids=[model_to_test.model_name for model_to_test in TEST_DATA])
def test_min_max_quantization_graph(tmp_path, mocker, model_to_test):
    mock_collect_statistics(mocker)

    if model_to_test.model_name == "ssd_mobilenet_v1_12":
        pytest.skip("Ticket 96156")
    convert_opset_version = True

    onnx_model_path = ONNX_MODEL_DIR / (model_to_test.model_name + ".onnx")
    original_model = load_model_topology_with_zeros_weights(onnx_model_path)

    ignored_scopes = IgnoredScope()
    if model_to_test.model_name == "MaskRCNN-12":
        # The problem with convert function - convert_opset_version.
        convert_opset_version = False
        # TODO: need to investigate disallowed_op_types for Mask RCNN
        ignored_scopes = IgnoredScope(types=["Resize", "Div", "RoiAlign", "ScatterElements"])
    if model_to_test.model_name == "ssd_mobilenet_v1_12":
        ignored_scopes = IgnoredScope(
            [
                "copy__21/Preprocessor/map/while/Less",
                "Preprocessor/mul",
                "copy__43/Postprocessor/BatchMultiClassNonMaxSuppression/map/while/Less",
                "add",
            ]
        )

    quantized_model = min_max_quantize_model(
        original_model, convert_model_opset=convert_opset_version, quantization_params={"ignored_scope": ignored_scopes}
    )
    onnx.save_model(quantized_model, tmp_path / (model_to_test.model_name + "_int8.onnx"))
    compare_nncf_graph(quantized_model, model_to_test.path_ref_graph)
