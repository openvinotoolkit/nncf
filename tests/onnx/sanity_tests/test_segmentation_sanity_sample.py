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

import onnxruntime as rt
import numpy as np

from examples.experimental.onnx.semantic_segmentation.onnx_ptq_segmentation import run
from nncf.experimental.post_training.api.dataset import Dataset
from tests.common.helpers import TEST_ROOT

MODELS_NAME = [
    'icnet_camvid',
    'unet_camvid'
]

INPUT_SHAPES = [
    [1, 3, 768, 960],
    [1, 3, 368, 480]
]


class TestDataloader(Dataset):
    def __init__(self, samples: List[Tuple[np.ndarray, int]]):
        super().__init__(shuffle=False)
        self.samples = samples

    def __getitem__(self, item):
        return self.samples[item]

    def __len__(self):
        return 1


def mock_dataloader_creator(dataset_name, dataset_path, input_shape):
    return TestDataloader([(np.zeros(input_shape[1:]), 0), ])


@pytest.mark.parametrize(("model_name, input_shape"),
                         zip(MODELS_NAME, INPUT_SHAPES))
@patch(
    'examples.experimental.onnx.semantic_segmentation.onnx_ptq_segmentation.'
    'create_dataset_from_segmentation_torch_dataset',
    new=mock_dataloader_creator)
def test_sanity_quantize_sample(tmp_path, model_name, input_shape):
    onnx_model_dir = str(TEST_ROOT.joinpath('onnx', 'data', 'models'))
    onnx_model_path = str(TEST_ROOT.joinpath(onnx_model_dir, model_name + '.onnx'))
    if not os.path.isdir(onnx_model_dir):
        os.mkdir(onnx_model_dir)

    onnx_output_model_path = str(tmp_path / model_name)

    run(onnx_model_path, onnx_output_model_path, 'CamVid',
        'none', num_init_samples=1, input_shape=input_shape, ignored_scopes=None)

    sess = rt.InferenceSession(onnx_output_model_path, providers=['OpenVINOExecutionProvider'])
    _input = np.random.random(input_shape)
    input_name = sess.get_inputs()[0].name
    _ = sess.run([], {input_name: _input.astype(np.float32)})
