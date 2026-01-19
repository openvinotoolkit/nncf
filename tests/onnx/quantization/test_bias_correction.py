# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import onnx
import pytest
import torch

from nncf.common.factory import build_graph
from nncf.onnx.graph.model_utils import remove_fq_from_inputs
from nncf.onnx.graph.nncf_graph_builder import GraphConverter
from nncf.onnx.graph.node_utils import get_bias_value
from nncf.quantization.algorithms.bias_correction.onnx_backend import ONNXBiasCorrectionAlgoBackend
from tests.cross_fw.test_templates.helpers import ConvTestModel
from tests.cross_fw.test_templates.helpers import DepthwiseConvTestModel
from tests.cross_fw.test_templates.helpers import MultipleConvTestModel
from tests.cross_fw.test_templates.helpers import OneDimMM
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
    def list_to_backend_type(data: list) -> np.ndarray:
        return np.array(data)

    @staticmethod
    def get_backend() -> ONNXBiasCorrectionAlgoBackend:
        return ONNXBiasCorrectionAlgoBackend

    @staticmethod
    def backend_specific_model(model: torch.nn.Module, tmp_dir: str) -> onnx.ModelProto:
        if isinstance(model, OneDimMM):
            pytest.skip("ONNX does not support BC with MM ops")
        onnx_path = f"{tmp_dir}/model.onnx"
        torch.onnx.export(model, torch.rand(model.INPUT_SIZE), onnx_path, opset_version=18, input_names=["input.1"])
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
    def check_bias(model: onnx.ModelProto, ref_biases: dict) -> None:
        nncf_graph = build_graph(model)
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
                "node_conv2d",
                {
                    "collected_inputs": {
                        ("node_concat", 1): ("nncf_model_input_0", 0),
                        ("node_conv2d", 0): ("nncf_model_input_0", 0),
                    },
                    "subgraph_data": {
                        "subgraph_input_ids": {("node_conv2d", 0)},
                        "subgraph_output_ids": {("node_Split_27", 0), ("node_max_pool2d", 0), ("node_Split_27", 1)},
                    },
                },
            ),
            (
                "node_conv2d_1",
                {
                    "collected_inputs": {
                        ("node_conv2d", 0): ("nncf_model_input_0", 0),
                        ("node_conv2d_1", 0): ("node_max_pool2d", 0),
                        ("node_conv2d_3", 0): ("node_Split_27", 0),
                        ("node_conv2d_5", 0): ("node_Split_27", 1),
                    },
                    "subgraph_data": {
                        "subgraph_input_ids": {("node_conv2d_1", 0)},
                        "subgraph_output_ids": {("node_relu_1", 0)},
                    },
                },
            ),
            (
                "node_conv2d_2",
                {
                    "collected_inputs": {
                        ("node_conv2d", 0): ("nncf_model_input_0", 0),
                        ("node_conv2d_1", 0): ("node_max_pool2d", 0),
                        ("node_conv2d_2", 0): ("node_relu_1", 0),
                        ("node_conv2d_3", 0): ("node_Split_27", 0),
                        ("node_conv2d_5", 0): ("node_Split_27", 1),
                    },
                    "subgraph_data": {
                        "subgraph_input_ids": {("node_conv2d", 0), ("node_conv2d_2", 0)},
                        "subgraph_output_ids": {("node_Split_27", 0), ("node_Split_27", 1)},
                    },
                },
            ),
            (
                "node_conv2d_3",
                {
                    "collected_inputs": {
                        ("node_conv2d_3", 0): ("node_Split_27", 0),
                        ("node_conv2d_5", 0): ("node_Split_27", 1),
                    },
                    "subgraph_data": {
                        "subgraph_input_ids": {("node_conv2d_3", 0)},
                        "subgraph_output_ids": {("node_relu_2", 0)},
                    },
                },
            ),
            (
                "node_conv2d_5",
                {
                    "collected_inputs": {
                        ("node_conv2d_4", 0): ("node_relu_2", 0),
                        ("node_conv2d_5", 0): ("node_Split_27", 1),
                    },
                    "subgraph_data": {
                        "subgraph_input_ids": {("node_conv2d_4", 0), ("node_conv2d_5", 0)},
                        "subgraph_output_ids": {("node_add_3", 0), ("node_concat_1", 0)},
                    },
                },
            ),
            (
                "node_conv2d_9",
                {
                    "collected_inputs": {
                        ("node_conv2d_7", 0): ("node_conv2d_6", 0),
                        ("node_conv2d_8", 0): ("node_add_3", 0),
                        ("node_conv2d_9", 0): ("node_concat_1", 0),
                    },
                    "subgraph_data": {
                        "subgraph_input_ids": {
                            ("node_conv2d_7", 0),
                            ("node_conv2d_8", 0),
                            ("node_conv2d_9", 0),
                        },
                        "subgraph_output_ids": {("node_concat_2", 0)},
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
                    ("node_conv2d", 0): ("node_concat", 0),
                    ("node_concat", 1): ("nncf_model_input_0", 0),
                },
            ),
            (
                MultipleConvTestModel,
                {
                    ("node_conv2d", 0): ("nncf_model_input_0", 0),
                    ("node_conv2d_2", 0): ("nncf_model_input_0", 0),
                },
            ),
            (ConvTestModel, {("node_conv2d", 0): ("nncf_model_input_0", 0)}),
            (DepthwiseConvTestModel, {("node_conv2d", 0): ("nncf_model_input_0", 0)}),
            (TransposeConvTestModel, {("node_convolution", 0): ("nncf_model_input_0", 0)}),
        ),
    )
    def test_verify_collected_stat_inputs_map(self, model_cls, ref_stat_inputs_map, tmpdir):
        return super().test_verify_collected_stat_inputs_map(model_cls, ref_stat_inputs_map, tmpdir)
