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
from typing import List

import numpy as np
import onnx
import torch

from nncf.quantization.algorithms.fast_bias_correction.onnx_backend import ONNXFastBiasCorrectionAlgoBackend
from tests.onnx.quantization.common import get_random_dataset_for_test
from tests.post_training.test_fast_bias_correction import TemplateTestFBCAlgorithm
from tests.torch.ptq.helpers import ConvTestModel


def get_data_from_node(model: onnx.ModelProto, node_name: str):
    data = [t for t in model.graph.initializer if t.name == node_name]
    if data:
        return onnx.numpy_helper.to_array(data[0])
    return None


class TestTorchFBCAlgorithm(TemplateTestFBCAlgorithm):
    @staticmethod
    def list_to_backend_type(data: List) -> np.array:
        return np.array(data)

    @staticmethod
    def get_backend() -> ONNXFastBiasCorrectionAlgoBackend:
        return ONNXFastBiasCorrectionAlgoBackend

    @staticmethod
    def get_model(with_bias, tmp_dir):
        model = ConvTestModel(bias=with_bias)
        onnx_path = f"{tmp_dir}/model.onnx"
        torch.onnx.export(model, torch.rand([1, 1, 4, 4]), onnx_path, opset_version=13)
        onnx_model = onnx.load(onnx_path)
        return onnx_model

    @staticmethod
    def get_dataset(model):
        dataset = get_random_dataset_for_test(model, False)
        return dataset

    @staticmethod
    def check_bias(model, with_bias):
        if with_bias:
            assert np.all(np.isclose(get_data_from_node(model, "conv.bias"), np.array([-2.0424285, -2.0424285])))
        else:
            assert get_data_from_node(model, "conv.bias") is None
