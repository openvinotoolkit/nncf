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

import os
import warnings

import networkx as nx
import numpy as np
import onnx
import onnxruntime as rt

from tests.common.helpers import TEST_ROOT
from tests.onnx.test_nncf_graph_builder import check_nx_graph

from nncf.experimental.post_training.api.dataset import Dataset
from nncf.experimental.post_training.compression_builder import CompressionBuilder
from nncf.experimental.onnx.algorithms.quantization.min_max_quantization import ONNXMinMaxQuantization
from nncf.experimental.post_training.algorithms.quantization import MinMaxQuantizationParameters
from nncf.experimental.post_training.algorithms.quantization import PostTrainingQuantization
from nncf.experimental.post_training.algorithms.quantization import PostTrainingQuantizationParameters
from nncf.experimental.onnx.graph.nncf_graph_builder import GraphConverter
from nncf.experimental.onnx.tensor import ONNXNNCFTensor

REFERENCE_GRAPHS_TEST_ROOT = 'data/reference_graphs/quantization'


class DatasetForTest(Dataset):
    def __init__(self, input_key: str, input_shape: List[int], has_batch_dim: bool):
        super().__init__()
        self.input_key = input_key
        self.input_shape = input_shape
        self.has_batch_dim = has_batch_dim

    def __getitem__(self, item: int):
        return {
            self.input_key: ONNXNNCFTensor(
                np.squeeze(np.random.random(self.input_shape).astype(np.float32),
                           axis=0)) if self.has_batch_dim else ONNXNNCFTensor(
                np.random.random(self.input_shape).astype(np.float32)),
            "targets": ONNXNNCFTensor(0)
        }

    def __len__(self):
        return 10


class ModelToTest:
    def __init__(self, model_name: str, input_shape: List[int]):
        self.model_name = model_name
        self.path_ref_graph = self.model_name + '.dot'
        self.input_shape = input_shape


def _get_input_key(original_model: onnx.ModelProto) -> str:
    input_keys = [node.name for node in original_model.graph.input]
    if len(input_keys) != 1:
        warnings.warn(
            f"The number of inputs should be 1(!={len(input_keys)}). "
            "Use input_keys[0]={input_keys[0]}.")

    return input_keys[0]


def min_max_quantize_model(
        input_shape: List[int], original_model: onnx.ModelProto, convert_opset_version: bool = True,
        ignored_scopes: List[str] = None, dataset_has_batch_size: bool = True) -> onnx.ModelProto:
    dataset = DatasetForTest(_get_input_key(original_model), input_shape, dataset_has_batch_size)
    builder = CompressionBuilder(convert_opset_version)
    builder.add_algorithm(
        ONNXMinMaxQuantization(MinMaxQuantizationParameters(number_samples=1, ignored_scopes=ignored_scopes)))
    quantized_model = builder.apply(original_model, dataset)
    return quantized_model


def ptq_quantize_model(
        input_shape: List[int], original_model: onnx.ModelProto, convert_opset_version: bool = True,
        ignored_scopes: List[str] = None, dataset_has_batch_size: bool = True) -> onnx.ModelProto:
    dataset = DatasetForTest(_get_input_key(original_model), input_shape, dataset_has_batch_size)
    builder = CompressionBuilder(convert_opset_version)
    builder.add_algorithm(
        PostTrainingQuantization(PostTrainingQuantizationParameters(number_samples=1, ignored_scopes=ignored_scopes)))
    quantized_model = builder.apply(original_model, dataset)
    return quantized_model


def compare_nncf_graph(quantized_model: onnx.ModelProto, path_ref_graph: str,
                       generate_ref_graphs: bool = False) -> None:
    nncf_graph = GraphConverter.create_nncf_graph(quantized_model)
    nx_graph = nncf_graph.get_graph_for_structure_analysis(extended=True)

    data_dir = os.path.join(TEST_ROOT, 'onnx', REFERENCE_GRAPHS_TEST_ROOT)
    path_to_dot = os.path.abspath(os.path.join(data_dir, path_ref_graph))

    if generate_ref_graphs:
        nx.drawing.nx_pydot.write_dot(nx_graph, path_to_dot)

    expected_graph = nx.drawing.nx_pydot.read_dot(path_to_dot)

    check_nx_graph(nx_graph, expected_graph)


def compare_nncf_graph_onnx_models(quantized_model: onnx.ModelProto, _quantized_model: onnx.ModelProto) -> None:
    nncf_graph = GraphConverter.create_nncf_graph(quantized_model)
    nx_graph = nncf_graph.get_graph_for_structure_analysis(extended=True)

    _nncf_graph = GraphConverter.create_nncf_graph(_quantized_model)
    _nx_graph = _nncf_graph.get_graph_for_structure_analysis(extended=True)

    check_nx_graph(nx_graph, _nx_graph)


def infer_model(input_shape: List[int], quantized_model: onnx.ModelProto) -> None:
    serialized_model = quantized_model.SerializeToString()
    sess = rt.InferenceSession(serialized_model, providers=['OpenVINOExecutionProvider'])
    _input = np.random.random(input_shape)
    input_name = sess.get_inputs()[0].name
    _ = sess.run([], {input_name: _input.astype(np.float32)})
