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

import os
import pytest

import torch
from torchvision import models
import onnx

import networkx as nx

from nncf.experimental.onnx.graph.nncf_graph_builder import GraphConverter
from nncf.experimental.onnx.model_normalizer import ONNXModelNormalizer

from tests.onnx.models import ALL_MODELS
from tests.common.helpers import TEST_ROOT
from tests.onnx.quantization.common import ModelToTest
from tests.onnx.quantization.common import check_nx_graph

PROJECT_ROOT = os.path.dirname(__file__)
REFERENCE_GRAPHS_TEST_ROOT = 'data/reference_graphs/original_nncf_graph'


@pytest.mark.parametrize(["model_creator_func", 'dump_graph'], zip(ALL_MODELS, [False] * len(ALL_MODELS)))
def test_compare_nncf_graph_synthetic_models(model_creator_func, dump_graph):
    model = model_creator_func()
    data_dir = os.path.join(PROJECT_ROOT, REFERENCE_GRAPHS_TEST_ROOT)
    path_to_dot = os.path.abspath(os.path.join(data_dir, 'synthetic', model.path_ref_graph))

    nncf_graph = GraphConverter.create_nncf_graph(model.onnx_model)
    nx_graph = nncf_graph.get_graph_for_structure_analysis(extended=True)

    if dump_graph:
        nncf_graph.dump_graph(path_to_dot, dump_graph)

    expected_graph = nx.drawing.nx_pydot.read_dot(path_to_dot)
    check_nx_graph(nx_graph, expected_graph)


@pytest.mark.parametrize(('model_to_test', 'model'),
                         [(ModelToTest('resnet18', [1, 3, 224, 224]), models.resnet18()),
                          (ModelToTest('mobilenet_v2', [1, 3, 224, 224]), models.mobilenet_v2()),
                          (ModelToTest('mobilenet_v3_small', [1, 3, 224, 224]), models.mobilenet_v3_small()),
                          (ModelToTest('inception_v3', [1, 3, 224, 224]), models.inception_v3()),
                          (ModelToTest('googlenet', [1, 3, 224, 224]), models.googlenet()),
                          (ModelToTest('vgg16', [1, 3, 224, 224]), models.vgg16()),
                          (ModelToTest('shufflenet_v2_x1_0', [1, 3, 224, 224]), models.shufflenet_v2_x1_0()),
                          (ModelToTest('squeezenet1_0', [1, 3, 224, 224]), models.squeezenet1_0()),
                          (ModelToTest('densenet121', [1, 3, 224, 224]), models.densenet121()),
                          (ModelToTest('mnasnet0_5', [1, 3, 224, 224]), models.mnasnet0_5()),
                          ]
                         )
def test_compare_nncf_graph_classification_real_models(tmp_path, model_to_test, model):
    onnx_model_dir = str(TEST_ROOT.joinpath('onnx', 'data', 'models'))
    onnx_model_path = str(TEST_ROOT.joinpath(onnx_model_dir, model_to_test.model_name))
    if not os.path.isdir(onnx_model_dir):
        os.mkdir(onnx_model_dir)
    x = torch.randn(model_to_test.input_shape, requires_grad=False)
    torch.onnx.export(model, x, onnx_model_path, opset_version=13)

    original_model = onnx.load(onnx_model_path)
    original_model = ONNXModelNormalizer.normalize_model(original_model)

    data_dir = os.path.join(PROJECT_ROOT, REFERENCE_GRAPHS_TEST_ROOT)
    path_to_dot = os.path.abspath(os.path.join(data_dir, model_to_test.path_ref_graph))

    nncf_graph = GraphConverter.create_nncf_graph(original_model)
    nx_graph = nncf_graph.get_graph_for_structure_analysis(extended=True)

    dump_graph = False
    if dump_graph:
        nncf_graph.dump_graph(path_to_dot, dump_graph)

    expected_graph = nx.drawing.nx_pydot.read_dot(path_to_dot)
    check_nx_graph(nx_graph, expected_graph)


@pytest.mark.parametrize(('model_to_test'),
                         [ModelToTest('ssd-12', [1, 3, 1200, 1200]),
                          ModelToTest('yolov2-coco-9', [1, 3, 416, 416]),
                          ModelToTest('tiny-yolov2', [1, 3, 416, 416]),
                          ModelToTest('MaskRCNN-12', [3, 30, 30]),
                          ModelToTest('retinanet-9', [1, 3, 480, 640]),
                          ModelToTest('fcn-resnet50-12', [1, 3, 480, 640])
                          ]
                         )
def test_compare_nncf_graph_detection_real_models(tmp_path, model_to_test):
    onnx_model_dir = str(TEST_ROOT.joinpath('onnx', 'data', 'models'))
    onnx_model_path = str(TEST_ROOT.joinpath(onnx_model_dir, model_to_test.model_name + '.onnx'))
    if not os.path.isdir(onnx_model_dir):
        os.mkdir(onnx_model_dir)
    original_model = onnx.load(onnx_model_path)

    convert_opset_version = True
    if model_to_test.model_name == 'MaskRCNN-12':
        convert_opset_version = False
    original_model = ONNXModelNormalizer.normalize_model(original_model, convert_opset_version)

    data_dir = os.path.join(PROJECT_ROOT, REFERENCE_GRAPHS_TEST_ROOT)
    path_to_dot = os.path.abspath(os.path.join(data_dir, model_to_test.path_ref_graph))

    nncf_graph = GraphConverter.create_nncf_graph(original_model)
    nx_graph = nncf_graph.get_graph_for_structure_analysis(extended=True)

    dump_graph = True
    if dump_graph:
        nncf_graph.dump_graph(path_to_dot, dump_graph)

    expected_graph = nx.drawing.nx_pydot.read_dot(path_to_dot)
    check_nx_graph(nx_graph, expected_graph)
