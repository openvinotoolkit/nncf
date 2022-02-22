import pytest

import onnxruntime as rt
import numpy as np

from examples.experimental.onnx_ptq_classification import run
from tests.common.helpers import TEST_ROOT

MODELS = [
    'efficientnet-b0.onnx',
    "googlenet_imagenet.onnx",
    'mobilenet_v2.onnx',
    'shufflenet-v2-10.onnx',
    'densenet121.onnx',
    'efficientnet_b1.onnx',
    'squeezenet1_1.onnx',
    'alexnet.onnx',
    'vgg19.onnx',
    'inception_v3.onnx',
    'shufflenet_v1.onnx',
    'resnet50.onnx',
    'squeezenet1_0.onnx',
    'efficientnet-v2-b0.onnx'
]

INPUT_SHAPE = [
    [1, 3, 224, 224],
    [1, 3, 224, 224],
    [1, 3, 224, 224],
    [1, 3, 224, 224],
    [1, 3, 224, 224],
    [1, 3, 240, 240],
    [1, 3, 224, 224],
    [1, 3, 224, 224],
    [1, 3, 224, 224],
    [1, 3, 299, 299],
    [1, 3, 224, 224],
    [1, 3, 224, 224],
    [1, 3, 224, 224],
    [1, 3, 224, 224],
]


@pytest.mark.parametrize(("model_name, input_shape"),
                         zip(MODELS, INPUT_SHAPE))
def test_quantize_models(tmp_path, model_name, input_shape):
    onnx_model_path = str(TEST_ROOT.joinpath('onnx', 'data', 'models', model_name))
    # onnx_output_model_path = str(tmp_path / model_name)
    onnx_output_model_path = str(tmp_path / model_name)
    dataset_path = '/home/aleksei/datasetsimagenet/'
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
    output_tensor = sess.run([], {input_name: _input.astype(np.float32)})
