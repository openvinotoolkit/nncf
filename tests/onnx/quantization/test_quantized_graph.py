import pytest

import os

import torch
from torchvision import models
import numpy as np
import onnx
# pylint: disable=no-member

from tests.common.helpers import TEST_ROOT
from tests.onnx.test_nncf_graph_builder import check_nx_graph

from nncf.experimental.post_training.compression_builder import CompressionBuilder
from nncf.experimental.onnx.algorithms.quantization.min_max_quantization import ONNXMinMaxQuantization
from nncf.experimental.post_training.algorithms.quantization import MinMaxQuantizationParameters
from nncf.experimental.post_training.algorithms.quantization import PostTrainingQuantization
from nncf.experimental.post_training.algorithms.quantization import PostTrainingQuantizationParameters
from nncf.experimental.post_training.api.dataloader import DataLoader

from nncf.experimental.onnx.graph.nncf_graph_builder import GraphConverter

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

REFERENCE_GRAPHS_TEST_ROOT = 'data/reference_graphs/quantization'


class TestDataloader(DataLoader):
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape

    def __getitem__(self, item):
        return np.squeeze(np.random.random(self.input_shape)), 0

    def __len__(self):
        return 10


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

    dataloader = TestDataloader(input_shape)
    builder = CompressionBuilder()
    builder.add_algorithm(ONNXMinMaxQuantization(MinMaxQuantizationParameters(number_samples=1)))
    quantized_model = builder.apply(original_model, dataloader)

    nncf_graph = GraphConverter.create_nncf_graph(quantized_model)
    nx_graph = nncf_graph.get_graph_for_structure_analysis(extended=True)

    data_dir = os.path.join(TEST_ROOT, 'onnx', REFERENCE_GRAPHS_TEST_ROOT)
    path_to_dot = os.path.abspath(os.path.join(data_dir, path_ref_graph))

    check_nx_graph(nx_graph, path_to_dot)


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

    dataloader = TestDataloader(input_shape)
    builder = CompressionBuilder()
    builder.add_algorithm(PostTrainingQuantization(PostTrainingQuantizationParameters(number_samples=1)))
    quantized_model = builder.apply(original_model, dataloader)

    nncf_graph = GraphConverter.create_nncf_graph(quantized_model)
    nx_graph = nncf_graph.get_graph_for_structure_analysis(extended=True)

    data_dir = os.path.join(TEST_ROOT, 'onnx', REFERENCE_GRAPHS_TEST_ROOT)
    path_to_dot = os.path.abspath(os.path.join(data_dir, path_ref_graph))

    check_nx_graph(nx_graph, path_to_dot)
