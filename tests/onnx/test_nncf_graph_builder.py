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

import os

import numpy as np
import onnx
import pytest
import torch

from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXMatMulMetatype
from nncf.onnx.graph.model_transformer import ONNXModelTransformer
from nncf.onnx.graph.nncf_graph_builder import GraphConverter
from tests.cross_fw.shared.nx_graph import compare_nx_graph_with_reference
from tests.cross_fw.shared.paths import TEST_ROOT
from tests.onnx.common import ModelBuilder
from tests.onnx.conftest import ONNX_TEST_ROOT
from tests.onnx.models import ALL_SYNTHETIC_MODELS
from tests.onnx.models import OneConvolutionalModel
from tests.onnx.opset_converter import convert_opset_version
from tests.onnx.quantization.common import ModelToTest
from tests.onnx.quantization.test_classification_models_graph import model_builder
from tests.onnx.weightless_model import load_model_topology_with_zeros_weights

REFERENCE_GRAPHS_DIR = ONNX_TEST_ROOT / "data" / "reference_graphs" / "original_nncf_graph"


@pytest.mark.parametrize(
    "model_cls_to_test",
    [pytest.param(mod, marks=mod.get_pytest_marks()) for mod in ALL_SYNTHETIC_MODELS.values()],
)
def test_compare_nncf_graph_synthetic_models(model_cls_to_test):
    model_to_test = model_cls_to_test()
    path_to_dot = REFERENCE_GRAPHS_DIR / "synthetic" / model_to_test.path_ref_graph

    nncf_graph = GraphConverter.create_nncf_graph(model_to_test.onnx_model)
    nx_graph = nncf_graph.get_graph_for_structure_analysis(extended=True)

    compare_nx_graph_with_reference(nx_graph, path_to_dot, check_edge_attrs=True)


CLASSIFICATION_MODEL_DEF_AND_OBJ = [
    ModelToTest("resnet18", [1, 3, 224, 224]),
    ModelToTest("mobilenet_v2", [1, 3, 224, 224]),
    ModelToTest("mobilenet_v3_small", [1, 3, 224, 224]),
    ModelToTest("inception_v3", [1, 3, 224, 224]),
    ModelToTest("googlenet", [1, 3, 224, 224]),
    ModelToTest("vgg16", [1, 3, 224, 224]),
    ModelToTest("shufflenet_v2_x1_0", [1, 3, 224, 224]),
    ModelToTest("squeezenet1_0", [1, 3, 224, 224]),
    ModelToTest("densenet121", [1, 3, 224, 224]),
    ModelToTest("mnasnet0_5", [1, 3, 224, 224]),
]


@pytest.mark.parametrize(
    ("model_to_test"),
    CLASSIFICATION_MODEL_DEF_AND_OBJ,
    ids=[x.model_name for x in CLASSIFICATION_MODEL_DEF_AND_OBJ],
)
def test_compare_nncf_graph_classification_real_models(tmp_path, model_to_test):
    model = model_builder(model_to_test.model_name)
    onnx_model_path = tmp_path / (model_to_test.model_name + ".onnx")
    x = torch.randn(model_to_test.input_shape, requires_grad=False)
    torch.onnx.export(model, x, onnx_model_path, opset_version=13, dynamo=False)

    original_model = onnx.load(onnx_model_path)

    path_to_dot = REFERENCE_GRAPHS_DIR / model_to_test.path_ref_graph

    nncf_graph = GraphConverter.create_nncf_graph(original_model)
    nx_graph = nncf_graph.get_graph_for_structure_analysis(extended=True)

    compare_nx_graph_with_reference(nx_graph, path_to_dot, check_edge_attrs=True)


DETECTION_MODELS = [
    ModelToTest("ssd-12", [1, 3, 1200, 1200]),
    ModelToTest("yolov2-coco-9", [1, 3, 416, 416]),
    ModelToTest("MaskRCNN-12", [3, 30, 30]),
    ModelToTest("retinanet-9", [1, 3, 480, 640]),
    ModelToTest("fcn-resnet50-12", [1, 3, 480, 640]),
]


@pytest.mark.parametrize(("model_to_test"), DETECTION_MODELS, ids=[x.model_name for x in DETECTION_MODELS])
def test_compare_nncf_graph_detection_real_models(model_to_test):
    onnx_model_dir = TEST_ROOT / "onnx" / "data" / "models"
    onnx_model_path = onnx_model_dir / (model_to_test.model_name + ".onnx")
    if not os.path.isdir(onnx_model_dir):
        os.mkdir(onnx_model_dir)
    original_model = load_model_topology_with_zeros_weights(onnx_model_path)

    if model_to_test.model_name != "MaskRCNN-12":
        original_model = convert_opset_version(original_model)

    path_to_dot = REFERENCE_GRAPHS_DIR / model_to_test.path_ref_graph

    nncf_graph = GraphConverter.create_nncf_graph(original_model)
    nx_graph = nncf_graph.get_graph_for_structure_analysis(extended=True)

    compare_nx_graph_with_reference(nx_graph, path_to_dot, check_edge_attrs=True)


def test_add_output_nodes_with_no_parents_node():
    model_to_test = OneConvolutionalModel().onnx_model
    model_outputs = (value_info.name for value_info in model_to_test.graph.output)
    model_with_output = ONNXModelTransformer._insert_outputs(model_to_test, (*model_outputs, "Conv1_W"))
    nncf_graph = GraphConverter.create_nncf_graph(model_with_output)
    nx_graph = nncf_graph.get_graph_for_structure_analysis(extended=True)
    path_to_dot = REFERENCE_GRAPHS_DIR / "synthetic" / "output_with_no_parents_model.dot"
    compare_nx_graph_with_reference(nx_graph, path_to_dot, check_edge_attrs=True)


@pytest.mark.parametrize("opset_version, ref_shape", [[13, ()], [19, (-1, -1, -1)]])
def test_unknown_shape(opset_version: int, ref_shape: tuple[int, ...]):
    mb = ModelBuilder()

    x = mb.add_input("x", ("batch", 3, 4, 5))

    y = mb.add_shape(x)
    y = mb.add_gather(y, mb.add_initializer(np.array(0, dtype=np.int64)))
    y = mb.add_unsqueeze(y, axes=[0])
    y = mb.add_concat([y, mb.add_initializer(np.array([-1, 60], dtype=np.int64))], axis=0)

    x = mb.add_reshape(x, y)
    x = mb.add_matmul(x, (60, 10))

    mb.add_output(x, ("batch", 1, 10))

    model = mb.build(opset_version, ir_version=9)

    graph = GraphConverter.create_nncf_graph(model)
    matmul = graph.get_nodes_by_metatypes([ONNXMatMulMetatype])[0]  # only 1 matmul

    for e in graph.get_input_edges(matmul):
        assert e.tensor_shape == ref_shape
