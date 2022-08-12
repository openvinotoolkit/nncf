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

import onnx

from nncf.experimental.onnx.model_normalizer import ONNXModelNormalizer
from nncf.experimental.onnx.algorithms.quantization.utils import find_ignored_scopes

from tests.common.helpers import TEST_ROOT

from tests.onnx.quantization.common import min_max_quantize_model
from tests.onnx.quantization.common import compare_nncf_graph
from tests.onnx.quantization.common import infer_model

MODELS_NAME = [
    'ssd-12',
    'yolov2-coco-9',
    'tiny-yolov2',
    'MaskRCNN-12',
    'retinanet-9',
    'fcn-resnet50-12'
]

PATH_REF_GRAPHS = [
    'ssd-12.dot',
    'yolov2-coco-9.dot',
    'tiny-yolov2.dot',
    'MaskRCNN-12.dot',
    'retinanet-9.dot',
    'fcn-resnet50-12.dot'
]

INPUT_SHAPES = [
    [1, 3, 1200, 1200],
    [1, 3, 416, 416],
    [1, 3, 416, 416],
    [3, 30, 30],
    [1, 3, 480, 640],
    [1, 3, 480, 640],
]

DISSALOWED_OP_TYPES = [
    [],
    [],
    [],
    ["Concat", "Mul", "Add", "Sub", "Sigmoid", "Softmax", "Floor", "RoiAlign", "Resize"],
    [],
    []
]


@pytest.mark.parametrize(('model_name', 'path_ref_graph', 'input_shape', 'disallowed_op_types'),
                         zip(MODELS_NAME, PATH_REF_GRAPHS, INPUT_SHAPES, 'disallowed_op_types'))
def test_min_max_quantization_graph(tmp_path, model_name, path_ref_graph, input_shape, disallowed_op_types):
    convert_opset_version = True
    dataset_has_batch_size = True
    if model_name == 'MaskRCNN-12':
        convert_opset_version = False
        dataset_has_batch_size = False
    onnx_model_dir = str(TEST_ROOT.joinpath('onnx', 'data', 'models'))
    onnx_model_path = str(TEST_ROOT.joinpath(onnx_model_dir, model_name + '.onnx'))
    if not os.path.isdir(onnx_model_dir):
        os.mkdir(onnx_model_dir)
    original_model = onnx.load(onnx_model_path)

    ignored_scopes = []
    if disallowed_op_types is not None:
        ignored_scopes += find_ignored_scopes(disallowed_op_types, original_model)

    quantized_model = min_max_quantize_model(input_shape, original_model, convert_opset_version=convert_opset_version,
                                             ignored_scopes=ignored_scopes,
                                             dataset_has_batch_size=dataset_has_batch_size)
    if convert_opset_version:
        quantized_model = ONNXModelNormalizer.convert_opset_version(quantized_model)
    ONNXModelNormalizer.add_input_from_initializer(quantized_model)

    compare_nncf_graph(quantized_model, path_ref_graph)
    if model_name == 'MaskRCNN-12':
        # TODO(kshpv): align inferred input shape
        return
    infer_model(input_shape, quantized_model)
