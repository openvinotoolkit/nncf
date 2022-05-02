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

import os

import torch
from torchvision import models
import onnx
# pylint: disable=no-member

from tests.common.helpers import TEST_ROOT

from tests.onnx.quantization.common import min_max_quantize_model
from tests.onnx.quantization.common import ptq_quantize_model
from tests.onnx.quantization.common import compare_nncf_graph
from tests.onnx.quantization.common import infer_model

MODELS = [
    models.resnet18(),
    models.mobilenet_v2(),
    models.inception_v3(),
    models.googlenet(),
    models.vgg16(),
    models.shufflenet_v2_x1_0(),
]

PATH_REF_GRAPHS = [
    'resnet18.dot',
    'mobilenet_v2.dot',
    'inception_v3.dot',
    'googlenet.dot',
    'vgg16.dot',
    'shufflenet_v2_x1_0.dot'
]

INPUT_SHAPES = [
    [1, 3, 224, 224],
    [1, 3, 224, 224],
    [1, 3, 224, 224],
    [1, 3, 224, 224],
    [1, 3, 224, 224],
    [1, 3, 224, 224],
]


@pytest.mark.parametrize(('model', 'path_ref_graph', 'input_shape'),
                         zip(MODELS, PATH_REF_GRAPHS, INPUT_SHAPES))
def test_min_max_quantization_graph(tmp_path, model, path_ref_graph, input_shape):
    model_name = str(model.__class__)
    onnx_model_dir = str(TEST_ROOT.joinpath('onnx', 'data', 'models'))
    onnx_model_path = str(TEST_ROOT.joinpath(onnx_model_dir, model_name))
    if not os.path.isdir(onnx_model_dir):
        os.mkdir(onnx_model_dir)
    x = torch.randn(input_shape, requires_grad=False)
    torch.onnx.export(model, x, onnx_model_path, opset_version=13)

    original_model = onnx.load(onnx_model_path)
    quantized_model = min_max_quantize_model(input_shape, original_model)
    compare_nncf_graph(quantized_model, path_ref_graph)
    infer_model(input_shape, quantized_model)


@pytest.mark.parametrize(('model', 'path_ref_graph', 'input_shape'),
                         zip(MODELS, PATH_REF_GRAPHS, INPUT_SHAPES))
def test_post_training_quantization_graph(tmp_path, model, path_ref_graph, input_shape):
    model_name = str(model.__class__)
    onnx_model_dir = str(TEST_ROOT.joinpath('onnx', 'data', 'models'))
    onnx_model_path = str(TEST_ROOT.joinpath(onnx_model_dir, model_name))
    if not os.path.isdir(onnx_model_dir):
        os.mkdir(onnx_model_dir)
    x = torch.randn(input_shape, requires_grad=False)
    torch.onnx.export(model, x, onnx_model_path, opset_version=13)

    original_model = onnx.load(onnx_model_path)
    quantized_model = ptq_quantize_model(input_shape, original_model)
    compare_nncf_graph(quantized_model, path_ref_graph)
    infer_model(input_shape, quantized_model)
