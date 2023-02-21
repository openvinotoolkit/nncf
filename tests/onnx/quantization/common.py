"""
 Copyright (c) 2023 Intel Corporation
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

from typing import List, Optional, Tuple

import os

import numpy as np
import onnx
import onnxruntime as rt

from nncf import Dataset
from tests.shared.paths import TEST_ROOT
from tests.shared.nx_graph import compare_nx_graph_with_reference
from tests.shared.nx_graph import check_nx_graph
from tests.onnx.opset_converter import convert_opset_version
from nncf.quantization.algorithms.min_max.algorithm import MinMaxQuantization
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantization
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantizationParameters
from nncf.onnx.graph.nncf_graph_builder import GraphConverter
from nncf.onnx.graph.onnx_graph import ONNXGraph
from nncf.onnx.statistics.statistics import ONNXMinMaxTensorStatistic
from nncf.quantization.fake_quantize import FakeQuantizeParameters

REFERENCE_GRAPHS_TEST_ROOT = 'data/reference_graphs/quantization'


def mock_collect_statistics(mocker):
    _ = mocker.patch(
        'nncf.quantization.fake_quantize.calculate_quantizer_parameters',
        return_value=FakeQuantizeParameters(np.array(0), np.array(0), np.array(0), np.array(0), 256))
    _ = mocker.patch(
        'nncf.common.tensor_statistics.aggregator.StatisticsAggregator.collect_statistics', return_value=None)
    _ = mocker.patch(
        'nncf.common.tensor_statistics.collectors.TensorStatisticCollectorBase.get_statistics',
        return_value=ONNXMinMaxTensorStatistic(min_values=np.array(0), max_values=np.array(0)))


def _get_input_keys(original_model: onnx.ModelProto) -> str:
    input_keys = [node.name for node in original_model.graph.input]
    return input_keys


def get_random_dataset_for_test(model: onnx.ModelProto, has_batch_dim: bool,
                                length: Optional[int] = 10):
    keys = _get_input_keys(model)
    onnx_graph = ONNXGraph(model)

    def transform_fn(i):
        output = {}
        for key in keys:
            input_dtype = onnx_graph.get_edge_dtype(key)
            input_np_dtype = onnx.helper.mapping.TENSOR_TYPE_TO_NP_TYPE[input_dtype]
            shape = onnx_graph.get_edge_shape(key)
            tensor = np.random.random(shape).astype(input_np_dtype)
            if has_batch_dim:
                tensor = np.squeeze(np.random.random(shape).astype(input_np_dtype), axis=0)
            output[key] = tensor
        return output

    return Dataset(list(range(length)), transform_fn)


def get_dataset_for_test(samples: List[Tuple[np.ndarray, int]], input_name: str):
    def transform_fn(data_item):
        inputs = data_item
        return {input_name: [inputs]}

    return Dataset(samples, transform_fn)


class ModelToTest:
    def __init__(self, model_name: str, input_shape: List[int]):
        self.model_name = model_name
        self.path_ref_graph = self.model_name + '.dot'
        self.input_shape = input_shape


def min_max_quantize_model(original_model: onnx.ModelProto, convert_model_opset: bool = True,
                           ignored_scopes: List[str] = None, dataset_has_batch_size: bool = False) -> onnx.ModelProto:
    if convert_model_opset:
        original_model = convert_opset_version(original_model)
    dataset = get_random_dataset_for_test(original_model, dataset_has_batch_size)
    post_training_quantization = PostTrainingQuantization(
        PostTrainingQuantizationParameters(number_samples=1, ignored_scopes=ignored_scopes))
    # Using PTQ, but apply only MinMax
    updated_algorithms = []
    for algo in post_training_quantization.algorithms:
        if isinstance(algo, MinMaxQuantization):
            updated_algorithms.append(algo)
    post_training_quantization.algorithms = updated_algorithms
    quantized_model = post_training_quantization.apply(original_model, dataset=dataset)
    return quantized_model


def ptq_quantize_model(original_model: onnx.ModelProto, convert_model_opset: bool = True,
                       ignored_scopes: List[str] = None, dataset_has_batch_size: bool = False) -> onnx.ModelProto:
    if convert_model_opset:
        original_model = convert_opset_version(original_model)
    dataset = get_random_dataset_for_test(original_model, dataset_has_batch_size)
    post_training_quantization = PostTrainingQuantization(
        PostTrainingQuantizationParameters(number_samples=1, ignored_scopes=ignored_scopes))
    quantized_model = post_training_quantization.apply(original_model, dataset=dataset)
    return quantized_model


def compare_nncf_graph(quantized_model: onnx.ModelProto, path_ref_graph: str) -> None:
    nncf_graph = GraphConverter.create_nncf_graph(quantized_model)
    nx_graph = nncf_graph.get_graph_for_structure_analysis(extended=True)

    data_dir = os.path.join(TEST_ROOT, 'onnx', REFERENCE_GRAPHS_TEST_ROOT)
    path_to_dot = os.path.abspath(os.path.join(data_dir, path_ref_graph))

    compare_nx_graph_with_reference(nx_graph, path_to_dot, check_edge_attrs=True)


def compare_nncf_graph_onnx_models(quantized_model: onnx.ModelProto, _quantized_model: onnx.ModelProto) -> None:
    nncf_graph = GraphConverter.create_nncf_graph(quantized_model)
    nx_graph = nncf_graph.get_graph_for_structure_analysis(extended=True)

    _nncf_graph = GraphConverter.create_nncf_graph(_quantized_model)
    _nx_graph = _nncf_graph.get_graph_for_structure_analysis(extended=True)

    check_nx_graph(nx_graph, _nx_graph, check_edge_attrs=True)


def infer_model(input_shape: List[int], quantized_model: onnx.ModelProto) -> None:
    onnx_graph = ONNXGraph(quantized_model)
    input_dtype = onnx_graph.get_edge_dtype(quantized_model.graph.input[0].name)
    input_np_dtype = onnx.helper.mapping.TENSOR_TYPE_TO_NP_TYPE[input_dtype]
    serialized_model = quantized_model.SerializeToString()
    sess = rt.InferenceSession(serialized_model, providers=['OpenVINOExecutionProvider'])
    _input = np.random.random(input_shape)
    input_name = sess.get_inputs()[0].name
    _ = sess.run([], {input_name: _input.astype(input_np_dtype)})


def find_ignored_scopes(disallowed_op_types: List[str], model: onnx.ModelProto) -> List[str]:
    disallowed_op_types = set(disallowed_op_types)
    return [node.name for node in model.graph.node if node.op_type in disallowed_op_types]
