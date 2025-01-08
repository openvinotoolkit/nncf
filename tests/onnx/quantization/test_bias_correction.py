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

from typing import Dict, List

import numpy as np
import onnx
import pytest
import torch

from nncf.common.factory import NNCFGraphFactory
from nncf.onnx.graph.model_utils import remove_fq_from_inputs
from nncf.onnx.graph.nncf_graph_builder import GraphConverter
from nncf.onnx.graph.node_utils import get_bias_value
from nncf.quantization.algorithms.bias_correction.onnx_backend import ONNXBiasCorrectionAlgoBackend
from tests.cross_fw.test_templates.helpers import ConvTestModel
from tests.cross_fw.test_templates.helpers import DepthwiseConvTestModel
from tests.cross_fw.test_templates.helpers import MultipleConvTestModel
from tests.cross_fw.test_templates.helpers import SplittedModel
from tests.cross_fw.test_templates.helpers import TransposeConvTestModel
from tests.cross_fw.test_templates.test_bias_correction import TemplateTestBCAlgorithm
from tests.onnx.quantization.common import compare_nncf_graph


def get_data_from_node(model: onnx.ModelProto, node_name: str):
    data = [t for t in model.graph.initializer if t.name == node_name]
    if data:
        return onnx.numpy_helper.to_array(data[0])
    return None


class TestONNXBCAlgorithm(TemplateTestBCAlgorithm):
    @staticmethod
    def list_to_backend_type(data: List) -> np.ndarray:
        return np.array(data)

    @staticmethod
    def get_backend() -> ONNXBiasCorrectionAlgoBackend:
        return ONNXBiasCorrectionAlgoBackend

    @staticmethod
    def backend_specific_model(model: torch.nn.Module, tmp_dir: str) -> onnx.ModelProto:
        onnx_path = f"{tmp_dir}/model.onnx"
        torch.onnx.export(model, torch.rand(model.INPUT_SIZE), onnx_path, opset_version=13, input_names=["input.1"])
        onnx_model = onnx.load(onnx_path)
        return onnx_model

    @staticmethod
    def fn_to_type(tensor) -> np.ndarray:
        return np.array(tensor)

    @staticmethod
    def get_transform_fn() -> callable:
        def transform_fn(data_item):
            tensor, _ = data_item
            return {"input.1": tensor}

        return transform_fn

    @staticmethod
    def remove_fq_from_inputs(model: onnx.ModelProto) -> onnx.ModelProto:
        graph = GraphConverter.create_nncf_graph(model)
        return remove_fq_from_inputs(model, graph)

    @staticmethod
    def compare_nncf_graphs(model: onnx.ModelProto, ref_path: str) -> None:
        return compare_nncf_graph(model, ref_path)

    @staticmethod
    def check_bias(model: onnx.ModelProto, ref_biases: Dict) -> None:
        nncf_graph = NNCFGraphFactory.create(model)
        for ref_name, ref_value in ref_biases.items():
            node = nncf_graph.get_node_by_name(ref_name)
            ref_value = np.array(ref_value)
            curr_value = get_bias_value(node, model)
            # TODO(AlexanderDokuchaev): return atol=0.0001 after fix 109189
            assert np.all(np.isclose(curr_value, ref_value, atol=0.01)), f"{curr_value} != {ref_value}"

    @pytest.mark.parametrize(
        "layer_name, ref_data",
        (
            (
                "/conv_1/Conv",
                {
                    "collected_inputs": {
                        ("/Concat", 1): ("nncf_model_input_0", 0),
                        ("/conv_1/Conv", 0): ("nncf_model_input_0", 0),
                    },
                    "subgraph_data": {
                        "subgraph_input_ids": {("/conv_1/Conv", 0)},
                        "subgraph_output_ids": {("/Split", 0), ("/maxpool_1/MaxPool", 0), ("/Split", 1)},
                    },
                },
            ),
            (
                "/conv_2/Conv",
                {
                    "collected_inputs": {
                        ("/conv_1/Conv", 0): ("nncf_model_input_0", 0),
                        ("/conv_2/Conv", 0): ("/maxpool_1/MaxPool", 0),
                        ("/conv_4/Conv", 0): ("/Split", 0),
                        ("/conv_6/Conv", 0): ("/Split", 1),
                    },
                    "subgraph_data": {
                        "subgraph_input_ids": {("/conv_2/Conv", 0)},
                        "subgraph_output_ids": {("/Relu_1", 0)},
                    },
                },
            ),
            (
                "/conv_3/Conv",
                {
                    "collected_inputs": {
                        ("/conv_1/Conv", 0): ("nncf_model_input_0", 0),
                        ("/conv_2/Conv", 0): ("/maxpool_1/MaxPool", 0),
                        ("/conv_3/Conv", 0): ("/Relu_1", 0),
                        ("/conv_4/Conv", 0): ("/Split", 0),
                        ("/conv_6/Conv", 0): ("/Split", 1),
                    },
                    "subgraph_data": {
                        "subgraph_input_ids": {("/conv_1/Conv", 0), ("/conv_3/Conv", 0)},
                        "subgraph_output_ids": {("/Split", 0), ("/Split", 1)},
                    },
                },
            ),
            (
                "/conv_4/Conv",
                {
                    "collected_inputs": {
                        ("/conv_4/Conv", 0): ("/Split", 0),
                        ("/conv_6/Conv", 0): ("/Split", 1),
                    },
                    "subgraph_data": {
                        "subgraph_input_ids": {("/conv_4/Conv", 0)},
                        "subgraph_output_ids": {("/Relu_2", 0)},
                    },
                },
            ),
            (
                "/conv_6/Conv",
                {
                    "collected_inputs": {
                        ("/conv_5/Conv", 0): ("/Relu_2", 0),
                        ("/conv_6/Conv", 0): ("/Split", 1),
                    },
                    "subgraph_data": {
                        "subgraph_input_ids": {("/conv_5/Conv", 0), ("/conv_6/Conv", 0)},
                        "subgraph_output_ids": {("/Add_3", 0), ("/Concat_1", 0)},
                    },
                },
            ),
            (
                "/conv_10/Conv",
                {
                    "collected_inputs": {
                        ("/conv_8/Conv", 0): ("/conv_7/Conv", 0),
                        ("/conv_9/Conv", 0): ("/Add_3", 0),
                        ("/conv_10/Conv", 0): ("/Concat_1", 0),
                    },
                    "subgraph_data": {
                        "subgraph_input_ids": {
                            ("/conv_8/Conv", 0),
                            ("/conv_9/Conv", 0),
                            ("/conv_10/Conv", 0),
                        },
                        "subgraph_output_ids": {("/Concat_2", 0)},
                    },
                },
            ),
            # Disabled, because ONNX backend doesn't support bias correction for MatMul
            # Ticket - CVS-115696
            # (
            #     "/MatMul",
            #     {
            #         "collected_inputs": {
            #             "/MatMul": ("/Reshape", 0),
            #         },
            #         "subgraph_data": {
            #             "subgraph_input_names": {"/MatMul"},
            #             "subgraph_output_names": {"/Reshape_1", "/Add_4"},
            #             "subgraph_output_ids": {("/Reshape_1", 0), ("/Add_4", 0)},
            #         },
            #     },
            # ),
        ),
    )
    def test__get_subgraph_data_for_node(self, quantized_test_model, layer_name, ref_data):
        return super().test__get_subgraph_data_for_node(quantized_test_model, layer_name, ref_data)

    @pytest.mark.parametrize(
        "model_cls, ref_stat_inputs_map",
        (
            (
                SplittedModel,
                {
                    ("/conv_1/Conv", 0): ("/Concat", 0),
                    ("/Concat", 1): ("nncf_model_input_0", 0),
                },
            ),
            (
                MultipleConvTestModel,
                {
                    ("/conv_1/Conv", 0): ("nncf_model_input_0", 0),
                    ("/conv_3/Conv", 0): ("nncf_model_input_0", 0),
                },
            ),
            (ConvTestModel, {("/conv/Conv", 0): ("nncf_model_input_0", 0)}),
            (DepthwiseConvTestModel, {("/conv/Conv", 0): ("nncf_model_input_0", 0)}),
            (TransposeConvTestModel, {("/conv/ConvTranspose", 0): ("nncf_model_input_0", 0)}),
        ),
    )
    def test_verify_collected_stat_inputs_map(self, model_cls, ref_stat_inputs_map, tmpdir):
        return super().test_verify_collected_stat_inputs_map(model_cls, ref_stat_inputs_map, tmpdir)
