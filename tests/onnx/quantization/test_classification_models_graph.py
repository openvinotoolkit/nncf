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
import torch
from torchvision import models

from nncf.parameters import TargetDevice
from tests.onnx.conftest import ONNX_MODEL_DIR
from tests.onnx.quantization.common import ModelToTest
from tests.onnx.quantization.common import compare_nncf_graph
from tests.onnx.quantization.common import min_max_quantize_model
from tests.onnx.quantization.common import mock_collect_statistics
from tests.onnx.weightless_model import load_model_topology_with_zeros_weights

TORCHVISION_TEST_DATA = [
    (
        ModelToTest("resnet18", [1, 3, 224, 224]),
        models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1),
        {},
    ),
    (
        ModelToTest("resnet50_cpu_spr", [1, 3, 224, 224]),
        models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1),
        {"target_device": TargetDevice.CPU_SPR},
    ),
    (
        ModelToTest("mobilenet_v2", [1, 3, 224, 224]),
        models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1),
        {},
    ),
    (
        ModelToTest("mobilenet_v3_small", [1, 3, 224, 224]),
        models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1),
        {},
    ),
    (
        ModelToTest("inception_v3", [1, 3, 224, 224]),
        models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1),
        {},
    ),
    (
        ModelToTest("googlenet", [1, 3, 224, 224]),
        models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1),
        {},
    ),
    (
        ModelToTest("vgg16", [1, 3, 224, 224]),
        models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1),
        {},
    ),
    (
        ModelToTest("shufflenet_v2_x1_0", [1, 3, 224, 224]),
        models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1),
        {},
    ),
    (
        ModelToTest("squeezenet1_0", [1, 3, 224, 224]),
        models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.IMAGENET1K_V1),
        {},
    ),
    (
        ModelToTest("densenet121", [1, 3, 224, 224]),
        models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1),
        {},
    ),
    (
        ModelToTest("mnasnet0_5", [1, 3, 224, 224]),
        models.mnasnet0_5(weights=models.MNASNet0_5_Weights.IMAGENET1K_V1),
        {},
    ),
]


@pytest.mark.parametrize(
    ("model_to_test", "model", "quantization_parameters"),
    TORCHVISION_TEST_DATA,
    ids=[model_to_test[0].model_name for model_to_test in TORCHVISION_TEST_DATA],
)
def test_min_max_quantization_graph_torchvision_models(tmp_path, mocker, model_to_test, model, quantization_parameters):
    mock_collect_statistics(mocker)
    onnx_model_path = tmp_path / (model_to_test.model_name + ".onnx")
    x = torch.randn(model_to_test.input_shape, requires_grad=False)
    torch.onnx.export(model, x, onnx_model_path, opset_version=13)

    original_model = onnx.load(onnx_model_path)
    quantized_model = min_max_quantize_model(original_model, quantization_params=quantization_parameters)
    compare_nncf_graph(quantized_model, model_to_test.path_ref_graph)


ONNX_TEST_DATA = [ModelToTest("densenet-12", [1, 3, 224, 224])]


@pytest.mark.parametrize(
    ("model_to_test"), ONNX_TEST_DATA, ids=[model_to_test.model_name for model_to_test in ONNX_TEST_DATA]
)
def test_min_max_quantization_graph_onnx_model(tmp_path, mocker, model_to_test):
    mock_collect_statistics(mocker)
    onnx_model_path = ONNX_MODEL_DIR / (model_to_test.model_name + ".onnx")
    original_model = load_model_topology_with_zeros_weights(onnx_model_path)

    quantized_model = min_max_quantize_model(original_model)
    onnx.save_model(quantized_model, tmp_path / (model_to_test.model_name + "_int8.onnx"))
    compare_nncf_graph(quantized_model, model_to_test.path_ref_graph)
