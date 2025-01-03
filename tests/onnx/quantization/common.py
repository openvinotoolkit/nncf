# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx

from nncf import Dataset
from nncf.experimental.common.tensor_statistics.statistics import MinMaxTensorStatistic
from nncf.onnx.graph.nncf_graph_builder import GraphConverter
from nncf.onnx.graph.onnx_helper import get_edge_dtype
from nncf.onnx.graph.onnx_helper import get_edge_info_mapping
from nncf.onnx.graph.onnx_helper import get_edge_shape
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from nncf.quantization.advanced_parameters import AdvancedSmoothQuantParameters
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantization
from nncf.quantization.fake_quantize import FakeQuantizeParameters
from nncf.tensor import Tensor
from tests.cross_fw.shared.nx_graph import check_nx_graph
from tests.cross_fw.shared.nx_graph import compare_nx_graph_with_reference
from tests.cross_fw.shared.paths import TEST_ROOT
from tests.onnx.common import get_random_generator
from tests.onnx.opset_converter import convert_opset_version

REFERENCE_GRAPHS_TEST_ROOT = "data/reference_graphs/quantization"


def mock_collect_statistics(mocker):
    get_statistics_value = MinMaxTensorStatistic(
        min_values=Tensor(np.array(-1, dtype=np.float32)), max_values=Tensor(np.array(1, dtype=np.float32))
    )
    _ = mocker.patch(
        "nncf.quantization.fake_quantize.calculate_quantizer_parameters",
        return_value=FakeQuantizeParameters(
            Tensor(np.array(0, dtype=np.float32)),
            Tensor(np.array(0, dtype=np.float32)),
            Tensor(np.array(0, dtype=np.float32)),
            Tensor(np.array(0, dtype=np.float32)),
            256,
        ),
    )
    _ = mocker.patch(
        "nncf.common.tensor_statistics.aggregator.StatisticsAggregator.collect_statistics", return_value=None
    )
    _ = mocker.patch(
        "nncf.experimental.common.tensor_statistics.collectors.TensorCollector.get_statistics",
        return_value=get_statistics_value,
    )


def _get_input_keys(original_model: onnx.ModelProto) -> str:
    input_keys = [node.name for node in original_model.graph.input]
    return input_keys


def get_random_dataset_for_test(model: onnx.ModelProto, has_batch_dim: bool, length: Optional[int] = 10):
    keys = _get_input_keys(model)
    edge_info_mapping = get_edge_info_mapping(model)

    def transform_fn(i):
        output = {}
        for key in keys:
            edge = edge_info_mapping[key]
            input_dtype = get_edge_dtype(edge)
            input_np_dtype = onnx.helper.tensor_dtype_to_np_dtype(input_dtype)
            shape = get_edge_shape(edge)
            static_shape = [axis if axis > 0 else 1 for axis in shape]
            rng = get_random_generator()
            tensor = rng.uniform(-1, 1, static_shape).astype(input_np_dtype)
            if has_batch_dim:
                tensor = np.squeeze(tensor, axis=0)
            output[key] = tensor
        return output

    return Dataset(list(range(length)), transform_fn)


def get_dataset_for_test(samples: List[Tuple[np.ndarray, int]], input_name: str):
    def transform_fn(data_item):
        inputs = data_item
        return {input_name: [inputs]}

    return Dataset(samples, transform_fn)


class ModelToTest:
    def __init__(self, model_name: str, input_shape: Optional[List[int]] = None):
        self.model_name = model_name
        self.path_ref_graph = self.model_name + ".dot"
        self.input_shape = input_shape


def min_max_quantize_model(
    original_model: onnx.ModelProto,
    convert_model_opset: bool = True,
    dataset_has_batch_size: bool = False,
    quantization_params: Dict[str, Any] = None,
) -> onnx.ModelProto:
    if convert_model_opset:
        original_model = convert_opset_version(original_model)
    graph = GraphConverter.create_nncf_graph(original_model)
    dataset = get_random_dataset_for_test(original_model, dataset_has_batch_size)
    quantization_params = {} if quantization_params is None else quantization_params

    advanced_parameters = quantization_params.get("advanced_parameters", AdvancedQuantizationParameters())

    # Turn off bias correction to test only MinMax
    advanced_parameters.disable_bias_correction = True
    # ONNX backend does not support these algorithms
    advanced_parameters.disable_channel_alignment = True
    advanced_parameters.smooth_quant_alphas = AdvancedSmoothQuantParameters(convolution=-1, matmul=-1)
    quantization_params["advanced_parameters"] = advanced_parameters

    post_training_quantization = PostTrainingQuantization(subset_size=1, **quantization_params)

    quantized_model = post_training_quantization.apply(original_model, graph, dataset=dataset)
    return quantized_model


def ptq_quantize_model(
    original_model: onnx.ModelProto,
    convert_model_opset: bool = True,
    dataset_has_batch_size: bool = False,
    quantization_params: Dict[str, Any] = None,
) -> onnx.ModelProto:
    if convert_model_opset:
        original_model = convert_opset_version(original_model)
    graph = GraphConverter.create_nncf_graph(original_model)
    dataset = get_random_dataset_for_test(original_model, dataset_has_batch_size)
    quantization_params = {} if quantization_params is None else quantization_params
    post_training_quantization = PostTrainingQuantization(subset_size=1, **quantization_params)
    quantized_model = post_training_quantization.apply(original_model, graph, dataset=dataset)
    return quantized_model


def compare_nncf_graph(quantized_model: onnx.ModelProto, path_ref_graph: str) -> None:
    nncf_graph = GraphConverter.create_nncf_graph(quantized_model)
    nx_graph = nncf_graph.get_graph_for_structure_analysis(extended=True)

    data_dir = os.path.join(TEST_ROOT, "onnx", REFERENCE_GRAPHS_TEST_ROOT)
    path_to_dot = os.path.abspath(os.path.join(data_dir, path_ref_graph))

    compare_nx_graph_with_reference(nx_graph, path_to_dot, check_edge_attrs=True)


def compare_nncf_graph_onnx_models(quantized_model: onnx.ModelProto, _quantized_model: onnx.ModelProto) -> None:
    nncf_graph = GraphConverter.create_nncf_graph(quantized_model)
    nx_graph = nncf_graph.get_graph_for_structure_analysis(extended=True)

    _nncf_graph = GraphConverter.create_nncf_graph(_quantized_model)
    _nx_graph = _nncf_graph.get_graph_for_structure_analysis(extended=True)

    check_nx_graph(nx_graph, _nx_graph, check_edge_attrs=True)


def find_ignored_scopes(disallowed_op_types: List[str], model: onnx.ModelProto) -> List[str]:
    disallowed_op_types = set(disallowed_op_types)
    return [node.name for node in model.graph.node if node.op_type in disallowed_op_types]
