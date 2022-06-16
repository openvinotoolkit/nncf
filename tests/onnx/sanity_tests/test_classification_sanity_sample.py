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

from typing import List
from typing import Tuple

import pytest
from unittest.mock import patch

import os

import torch
from torchvision import models
import onnxruntime as rt
import numpy as np

from examples.experimental.onnx.classification.onnx_ptq_classification import run
from nncf.experimental.post_training.api.dataset import Dataset
from tests.common.helpers import TEST_ROOT

MODEL_NAMES = [
    'resnet18',
    'mobilenet_v2',
    'inception_v3',
    'googlenet',
    'vgg16',
    'shufflenet_v2_x1_0'
]

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

TEST_CASES = [
    pytest.param(name, model, shape) if name != "shufflenet_v2_x1_0"
    else pytest.param(name, model, shape, marks=pytest.mark.xfail)
    for name, model, shape in zip(MODEL_NAMES, MODELS, INPUT_SHAPES)
]


class TestDataset(Dataset):
    def __init__(self, samples: List[Tuple[np.ndarray, int]]):
        super().__init__(shuffle=False)
        self.samples = samples

    def __getitem__(self, item):
        return self.samples[item]

    def __len__(self):
        return 1


def mock_dataset_creator(dataset_path, input_shape, batch_size, shuffle):
    return TestDataset([(np.zeros(input_shape[1:]), 0), ])


@pytest.mark.parametrize(("model_name, model, input_shape"), TEST_CASES)
@patch('examples.experimental.onnx.classification.onnx_ptq_classification.create_imagenet_torch_dataset',
       new=mock_dataset_creator)
def test_sanity_quantize_sample(tmp_path, model_name, model, input_shape):
    onnx_model_dir = str(TEST_ROOT.joinpath('onnx', 'data', 'models'))
    onnx_model_path = str(TEST_ROOT.joinpath(onnx_model_dir, model_name))
    if not os.path.isdir(onnx_model_dir):
        os.mkdir(onnx_model_dir)
    x = torch.randn(input_shape, requires_grad=False)
    torch.onnx.export(model, x, onnx_model_path, opset_version=13)
    onnx_output_model_path = str(tmp_path / model_name)
    run(onnx_model_path, onnx_output_model_path, 'none',
        batch_size=1, shuffle=True, num_init_samples=1,
        input_shape=input_shape, ignored_scopes=None)

    sess = rt.InferenceSession(onnx_output_model_path, providers=[
                               'OpenVINOExecutionProvider'])
    _input = np.random.random(input_shape)
    input_name = sess.get_inputs()[0].name
    _ = sess.run([], {input_name: _input.astype(np.float32)})
