import pytest

import os

import torch
from torchvision import models
import onnxruntime as rt
import numpy as np

from examples.experimental.onnx_ptq_classification import run
from tests.common.helpers import TEST_ROOT

MODELS = [
    models.resnet18(),
    models.mobilenet_v2(),
    models.inception_v3(),
    models.googlenet(),
    models.vgg16(),
    models.shufflenet_v2_x1_0(),
]

INPUT_SHAPES = [
    [1, 3, 224, 224],
    [1, 3, 224, 224],
    [1, 3, 224, 224],
    [1, 3, 224, 224],
    [1, 3, 224, 224],
    [1, 3, 224, 224],
]


@pytest.mark.parametrize(("model, input_shape"),
                         zip(MODELS, INPUT_SHAPES))
def test_sanity_quantize_sample(tmp_path, model, input_shape):
    model_name = str(model.__class__)
    onnx_model_dir = str(TEST_ROOT.joinpath('onnx', 'data', 'models'))
    onnx_model_path = str(TEST_ROOT.joinpath(onnx_model_dir, model_name))
    if not os.path.isdir(onnx_model_dir):
        os.mkdir(onnx_model_dir)
    x = torch.randn(input_shape, requires_grad=False)
    torch.onnx.export(model, x, onnx_model_path, opset_version=13)

    dataset_path = '/home/aleksei/datasetsimagenet/'
    onnx_output_model_path = str(tmp_path / model_name)
    batch_size = 1
    shuffle = True
    num_init_samples = 1
    ignored_scopes = None

    run(onnx_model_path, onnx_output_model_path, dataset_path,
        batch_size, shuffle, num_init_samples,
        input_shape, ignored_scopes)

    sess = rt.InferenceSession(onnx_output_model_path, providers=['OpenVINOExecutionProvider'])
    _input = np.random.random(input_shape)
    input_name = sess.get_inputs()[0].name
    _ = sess.run([], {input_name: _input.astype(np.float32)})
