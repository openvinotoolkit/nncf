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
import requests

import onnx

from tests.common.helpers import TEST_ROOT
from tests.onnx.test_nncf_graph_builder import check_nx_graph

from tests.onnx.quantization.test_quantized_graph import TestDataloader

from nncf.experimental.post_training.compression_builder import CompressionBuilder
from nncf.experimental.onnx.algorithms.quantization.min_max_quantization import ONNXMinMaxQuantization
from nncf.experimental.post_training.algorithms.quantization import MinMaxQuantizationParameters

from nncf.experimental.onnx.graph.nncf_graph_builder import GraphConverter

MODELS_NAME = [
    'yolov2-coco-9',
    'tiny-yolov2',

]

MODELS_URL = [
    'https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/yolov2-coco/model/yolov2-coco-9.onnx',
    'https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/tiny-yolov2/model/tinyyolov2-7.onnx',
    # 'https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/ssd-mobilenetv1/model/ssd_mobilenet_v1_10.onnx',
    # "https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/ssd/model/ssd-12.onnx"
]

PATH_REF_GRAPHS = [
    'yolov2-coco-9.dot',
    'tiny-yolov2.dot',
    # 'ssd_mobilenet_v1_10.dot',
    # 'ssd-12.dot'
]

INPUT_SHAPES = [
    [1, 3, 416, 416],
    [1, 3, 416, 416],
]

REFERENCE_GRAPHS_TEST_ROOT = 'data/reference_graphs/quantization'


@pytest.mark.parametrize(('model_name', 'model_url', 'path_ref_graph', 'input_shape'),
                         zip(MODELS_NAME, MODELS_URL, PATH_REF_GRAPHS, INPUT_SHAPES))
def test_min_max_quantization_graph(tmp_path, model_name, model_url, path_ref_graph, input_shape):
    onnx_model_dir = str(TEST_ROOT.joinpath('onnx', 'data', 'models'))
    onnx_model_path = str(TEST_ROOT.joinpath(onnx_model_dir, model_name))
    if not os.path.isdir(onnx_model_dir):
        os.mkdir(onnx_model_dir)

    r = requests.get(model_url)
    f = open(onnx_model_path, 'wb')
    f.write(r.content)

    original_model = onnx.load(onnx_model_path)

    dataloader = TestDataloader(input_shape)
    builder = CompressionBuilder()
    builder.add_algorithm(ONNXMinMaxQuantization(MinMaxQuantizationParameters(number_samples=1)))
    quantized_model = builder.apply(original_model, dataloader)

    nncf_graph = GraphConverter.create_nncf_graph(quantized_model)
    nx_graph = nncf_graph.get_graph_for_structure_analysis(extended=True)

    data_dir = os.path.join(TEST_ROOT, 'onnx', REFERENCE_GRAPHS_TEST_ROOT)
    path_to_dot = os.path.abspath(os.path.join(data_dir, path_ref_graph))

    check_nx_graph(nx_graph, path_to_dot, True)
