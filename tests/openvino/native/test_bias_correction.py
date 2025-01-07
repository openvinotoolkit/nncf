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

from typing import Any, Dict, List

import numpy as np
import openvino as ov
import pytest
import torch

from nncf.common.factory import NNCFGraphFactory
from nncf.openvino.graph.model_utils import remove_fq_from_inputs
from nncf.openvino.graph.nncf_graph_builder import GraphConverter
from nncf.openvino.graph.node_utils import get_bias_value
from nncf.quantization.algorithms.bias_correction.openvino_backend import OVBiasCorrectionAlgoBackend
from tests.cross_fw.test_templates.helpers import ConvTestModel
from tests.cross_fw.test_templates.helpers import DepthwiseConvTestModel
from tests.cross_fw.test_templates.helpers import MultipleConvTestModel
from tests.cross_fw.test_templates.helpers import SplittedModel
from tests.cross_fw.test_templates.helpers import TransposeConvTestModel
from tests.cross_fw.test_templates.test_bias_correction import TemplateTestBCAlgorithm
from tests.openvino.native.common import compare_nncf_graphs


class TestOVBCAlgorithm(TemplateTestBCAlgorithm):
    TRANSPOSE_CONV_NAME = "/conv/ConvTranspose/WithoutBiases"

    @staticmethod
    def list_to_backend_type(data: List) -> np.ndarray:
        return np.array(data)

    @staticmethod
    def get_backend() -> OVBiasCorrectionAlgoBackend:
        return OVBiasCorrectionAlgoBackend

    @staticmethod
    def backend_specific_model(model: torch.nn.Module, tmp_dir: str):
        onnx_path = f"{tmp_dir}/model.onnx"
        torch.onnx.export(model, torch.rand(model.INPUT_SIZE), onnx_path, opset_version=13, input_names=["input.1"])
        ov_model = ov.convert_model(onnx_path)
        if isinstance(model, TransposeConvTestModel):
            for node in ov_model.get_ops():
                if node.get_type_name() == "ConvolutionBackpropData":
                    node.set_friendly_name(TestOVBCAlgorithm.TRANSPOSE_CONV_NAME)
        return ov_model

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
    def map_references(ref_biases: Dict, model_cls: Any) -> Dict[str, List]:
        mapping = {f"{name}/WithoutBiases": val for name, val in ref_biases.items()}
        return mapping

    @staticmethod
    def remove_fq_from_inputs(model: ov.Model) -> ov.Model:
        graph = GraphConverter.create_nncf_graph(model)
        return remove_fq_from_inputs(model, graph)

    @staticmethod
    def compare_nncf_graphs(model: ov.Model, ref_path: str) -> None:
        return compare_nncf_graphs(model, ref_path)

    @staticmethod
    def check_bias(model: ov.Model, ref_biases: Dict) -> None:
        nncf_graph = NNCFGraphFactory.create(model)
        for ref_name, ref_value in ref_biases.items():
            node = nncf_graph.get_node_by_name(ref_name)
            ref_value = np.array(ref_value)
            curr_value = get_bias_value(node, nncf_graph, model)
            curr_value = curr_value.reshape(ref_value.shape)
            assert np.all(np.isclose(curr_value, ref_value, atol=0.0001)), f"{curr_value} != {ref_value}"

    @pytest.mark.parametrize(
        "layer_name, ref_data",
        (
            (
                "/conv_1/Conv/WithoutBiases",
                {
                    "collected_inputs": {
                        ("/Concat", 1): ("input.1", 0),
                        ("/conv_1/Conv/WithoutBiases", 0): ("/Concat", 0),
                    },
                    "subgraph_data": {
                        "subgraph_input_ids": {("/conv_1/Conv/WithoutBiases", 0)},
                        "subgraph_output_ids": {("/Split", 0), ("/maxpool_1/MaxPool", 0), ("/Split", 1)},
                    },
                },
            ),
            (
                "/conv_2/Conv/WithoutBiases",
                {
                    "collected_inputs": {
                        ("/conv_1/Conv/WithoutBiases", 0): ("/Concat", 0),
                        ("/conv_2/Conv/WithoutBiases", 0): ("/maxpool_1/MaxPool", 0),
                        ("/conv_4/Conv/WithoutBiases", 0): ("/Split", 0),
                        ("/conv_6/Conv/WithoutBiases", 0): ("/Split", 1),
                    },
                    "subgraph_data": {
                        "subgraph_input_ids": {("/conv_2/Conv/WithoutBiases", 0)},
                        "subgraph_output_ids": {("/Relu_1", 0)},
                    },
                },
            ),
            (
                "/conv_3/Conv/WithoutBiases",
                {
                    "collected_inputs": {
                        ("/conv_1/Conv/WithoutBiases", 0): ("/Concat", 0),
                        ("/conv_2/Conv/WithoutBiases", 0): ("/maxpool_1/MaxPool", 0),
                        ("/conv_3/Conv/WithoutBiases", 0): ("/Relu_1", 0),
                        ("/conv_4/Conv/WithoutBiases", 0): ("/Split", 0),
                        ("/conv_6/Conv/WithoutBiases", 0): ("/Split", 1),
                    },
                    "subgraph_data": {
                        "subgraph_input_ids": {("/conv_1/Conv/WithoutBiases", 0), ("/conv_3/Conv/WithoutBiases", 0)},
                        "subgraph_output_ids": {("/Split", 0), ("/Split", 1)},
                    },
                },
            ),
            (
                "/conv_4/Conv/WithoutBiases",
                {
                    "collected_inputs": {
                        ("/conv_4/Conv/WithoutBiases", 0): ("/Split", 0),
                        ("/conv_6/Conv/WithoutBiases", 0): ("/Split", 1),
                    },
                    "subgraph_data": {
                        "subgraph_input_ids": {("/conv_4/Conv/WithoutBiases", 0)},
                        "subgraph_output_ids": {("/Relu_2", 0)},
                    },
                },
            ),
            (
                "/conv_6/Conv/WithoutBiases",
                {
                    "collected_inputs": {
                        ("/conv_5/Conv/WithoutBiases", 0): ("/Relu_2", 0),
                        ("/conv_6/Conv/WithoutBiases", 0): ("/Split", 1),
                    },
                    "subgraph_data": {
                        "subgraph_input_ids": {("/conv_5/Conv/WithoutBiases", 0), ("/conv_6/Conv/WithoutBiases", 0)},
                        "subgraph_output_ids": {("/Add_3", 0), ("/Concat_1", 0)},
                    },
                },
            ),
            (
                "/conv_10/Conv/WithoutBiases",
                {
                    "collected_inputs": {
                        ("/conv_8/Conv/WithoutBiases", 0): ("/conv_7/Conv", 0),
                        ("/conv_9/Conv/WithoutBiases", 0): ("/Add_3", 0),
                        ("/conv_10/Conv/WithoutBiases", 0): ("/Concat", 0),
                    },
                    "subgraph_data": {
                        "subgraph_input_ids": {
                            ("/conv_8/Conv/WithoutBiases", 0),
                            ("/conv_9/Conv/WithoutBiases", 0),
                            ("/conv_10/Conv/WithoutBiases", 0),
                        },
                        "subgraph_output_ids": {("/Concat_2", 0)},
                    },
                },
            ),
            (
                "/MatMul",
                {
                    "collected_inputs": {
                        ("/MatMul", 0): ("/Reshape", 0),
                    },
                    "subgraph_data": {
                        "subgraph_input_ids": {("/MatMul", 0)},
                        "subgraph_output_ids": {("/Reshape_1", 0), ("/Add_4", 0)},
                    },
                },
            ),
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
                    ("/conv_1/Conv/WithoutBiases", 0): ("/Concat", 0),
                    ("/Concat", 1): ("input.1", 0),
                },
            ),
            (
                MultipleConvTestModel,
                {
                    ("/conv_1/Conv/WithoutBiases", 0): ("input.1", 0),
                    ("/conv_3/Conv/WithoutBiases", 0): ("input.1", 0),
                },
            ),
            (ConvTestModel, {("/conv/Conv/WithoutBiases", 0): ("input.1", 0)}),
            (DepthwiseConvTestModel, {("/conv/Conv/WithoutBiases", 0): ("input.1", 0)}),
            (TransposeConvTestModel, {(TRANSPOSE_CONV_NAME, 0): ("input.1", 0)}),
        ),
    )
    def test_verify_collected_stat_inputs_map(self, model_cls, ref_stat_inputs_map, tmpdir):
        return super().test_verify_collected_stat_inputs_map(model_cls, ref_stat_inputs_map, tmpdir)
