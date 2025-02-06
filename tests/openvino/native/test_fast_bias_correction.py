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

from typing import List

import numpy as np
import openvino as ov
import torch

from nncf.common.factory import NNCFGraphFactory
from nncf.common.utils.os import is_macos
from nncf.openvino.graph.node_utils import get_bias_value
from nncf.openvino.graph.node_utils import is_node_with_bias
from nncf.quantization.algorithms.fast_bias_correction.openvino_backend import OVFastBiasCorrectionAlgoBackend
from tests.cross_fw.test_templates.test_fast_bias_correction import TemplateTestFBCAlgorithm


class TestOVFBCAlgorithm(TemplateTestFBCAlgorithm):
    @staticmethod
    def list_to_backend_type(data: List) -> np.ndarray:
        return np.array(data)

    @staticmethod
    def get_backend() -> OVFastBiasCorrectionAlgoBackend:
        return OVFastBiasCorrectionAlgoBackend

    @staticmethod
    def backend_specific_model(model: bool, tmp_dir: str):
        onnx_path = f"{tmp_dir}/model.onnx"
        torch.onnx.export(model, torch.rand(model.INPUT_SIZE), onnx_path, opset_version=13, input_names=["input.1"])
        ov_model = ov.convert_model(onnx_path, input=model.INPUT_SIZE)
        return ov_model

    @staticmethod
    def fn_to_type(tensor):
        return np.array(tensor)

    @staticmethod
    def get_transform_fn():
        def transform_fn(data_item):
            tensor, _ = data_item
            return {"input.1": tensor}

        return transform_fn

    @staticmethod
    def check_bias(model: ov.Model, ref_bias: list):
        ref_bias = np.array(ref_bias)
        nncf_graph = NNCFGraphFactory.create(model)

        atol = 0.0001 if not is_macos() else 0.01

        for node in nncf_graph.get_all_nodes():
            if not is_node_with_bias(node, nncf_graph):
                continue
            bias_value = get_bias_value(node, nncf_graph, model)
            bias_value = bias_value.reshape(ref_bias.shape)
            assert np.all(np.isclose(bias_value, ref_bias, atol=atol)), f"{bias_value} != {ref_bias}"

            return
        raise ValueError("Not found node with bias")
