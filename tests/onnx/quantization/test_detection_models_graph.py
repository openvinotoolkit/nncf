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
import pytest

import os

import onnx

from nncf.experimental.onnx.model_normalizer import ONNXModelNormalizer
from nncf.experimental.onnx.algorithms.quantization.utils import find_ignored_scopes

from tests.common.helpers import TEST_ROOT

from tests.onnx.quantization.common import min_max_quantize_model
from tests.onnx.quantization.common import compare_nncf_graph
from tests.onnx.quantization.common import infer_model


class TestCase:
    def __init__(self, model_name: str, input_shape: List[int]):
        self.model_name = model_name
        self.path_ref_graph = self.model_name + '.dot'
        self.input_shape = input_shape


@pytest.mark.parametrize(('test_case'),
                         [TestCase('ssd-12', [1, 3, 1200, 1200]),
                          TestCase('yolov2-coco-9', [1, 3, 416, 416]),
                          TestCase('tiny-yolov2', [1, 3, 416, 416]),
                          TestCase('MaskRCNN-12', [3, 30, 30]),
                          TestCase('retinanet-9', [1, 3, 480, 640]),
                          TestCase('fcn-resnet50-12', [1, 3, 480, 640])
                          ]
                         )
def test_min_max_quantization_graph(tmp_path, test_case):
    convert_opset_version = True
    dataset_has_batch_size = len(test_case.input_shape) > 3

    onnx_model_dir = str(TEST_ROOT.joinpath('onnx', 'data', 'models'))
    onnx_model_path = str(TEST_ROOT.joinpath(onnx_model_dir, test_case.model_name + '.onnx'))
    if not os.path.isdir(onnx_model_dir):
        os.mkdir(onnx_model_dir)
    original_model = onnx.load(onnx_model_path)

    ignored_scopes = []
    if test_case.model_name == 'MaskRCNN-12':
        # The problem with convert function - convert_opset_version.
        convert_opset_version = False
        # TODO: need to investigate disallowed_op_types for Mask RCNN
        ignored_scopes += find_ignored_scopes(
            ["Concat", "Mul", "Add", "Sub", "Sigmoid", "Softmax", "Floor", "RoiAlign", "Resize", 'Div'], original_model)

    quantized_model = min_max_quantize_model(test_case.input_shape, original_model,
                                             convert_opset_version=convert_opset_version,
                                             ignored_scopes=ignored_scopes,
                                             dataset_has_batch_size=dataset_has_batch_size)
    if convert_opset_version:
        quantized_model = ONNXModelNormalizer.convert_opset_version(quantized_model)
        infer_model(test_case.input_shape, quantized_model)
    compare_nncf_graph(quantized_model, test_case.path_ref_graph)
