import pytest

import os

import torch
import torchvision.models as models
import numpy as np
import onnx

from tests.common.helpers import TEST_ROOT
from tests.onnx.test_nncf_graph_builder import check_nx_graph

from nncf.experimental.onnx.engine import ONNXEngine
from nncf.experimental.onnx.algorithms.min_max_quantization import ONNXMinMaxQuantization
from nncf.experimental.onnx.algorithms.min_max_quantization import MinMaxQuantizationParameters
from nncf.experimental.onnx.statistics.statistics_collector import ONNXStatisticsCollector

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

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
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
def test_quantized_graph(tmp_path, model, path_ref_graph, input_shape):
    model_name = str(model.__class__)
    onnx_model_path = str(TEST_ROOT.joinpath('onnx', 'data', 'models', model_name))
    x = torch.randn(input_shape, requires_grad=False)
    torch.onnx.export(model, x, onnx_model_path, opset_version=13)

    original_model = onnx.load(onnx_model_path)

    dataloader = TestDataloader(input_shape)
    engine = ONNXEngine(dataloader)
    statistics_collector = ONNXStatisticsCollector(engine, 1)
    algorithm = ONNXMinMaxQuantization(statistics_collector, MinMaxQuantizationParameters())
    compressed_model = algorithm.apply(original_model, engine)

    nncf_graph = GraphConverter.create_nncf_graph(compressed_model)
    nx_graph = nncf_graph.get_graph_for_structure_analysis(extended=True)

    data_dir = os.path.join(PROJECT_ROOT, REFERENCE_GRAPHS_TEST_ROOT)
    path_to_dot = os.path.abspath(os.path.join(data_dir, path_ref_graph))

    check_nx_graph(nx_graph, path_to_dot)
