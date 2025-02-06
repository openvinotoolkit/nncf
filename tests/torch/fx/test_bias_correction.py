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

import re
from typing import Any, Dict, List

import numpy as np
import openvino as ov
import pytest
import torch.fx

from nncf.common.factory import NNCFGraphFactory
from nncf.experimental.torch.fx.model_utils import remove_fq_from_inputs
from nncf.experimental.torch.fx.nncf_graph_builder import GraphConverter
from nncf.experimental.torch.fx.node_utils import get_bias_value
from nncf.quantization.algorithms.bias_correction.torch_fx_backend import FXBiasCorrectionAlgoBackend
from tests.cross_fw.test_templates.helpers import ConvTestModel
from tests.cross_fw.test_templates.helpers import DepthwiseConvTestModel
from tests.cross_fw.test_templates.helpers import MultipleConvTestModel
from tests.cross_fw.test_templates.helpers import SplittedModel
from tests.cross_fw.test_templates.helpers import TransposeConvTestModel
from tests.cross_fw.test_templates.test_bias_correction import TemplateTestBCAlgorithm
from tests.torch.fx.helpers import get_torch_fx_model_q_transformed


class TestFXBCAlgorithm(TemplateTestBCAlgorithm):

    @staticmethod
    def list_to_backend_type(data: List) -> torch.Tensor:
        return torch.tensor(data)

    @staticmethod
    def get_backend() -> FXBiasCorrectionAlgoBackend:
        return FXBiasCorrectionAlgoBackend

    @staticmethod
    def backend_specific_model(model: torch.nn.Module, tmp_dir: str) -> torch.fx.GraphModule:
        return get_torch_fx_model_q_transformed(model, torch.ones(model.INPUT_SIZE))

    @staticmethod
    def fn_to_type(tensor) -> np.ndarray:
        return np.array(tensor)

    @staticmethod
    def get_transform_fn() -> callable:
        def transform_fn(data_item):
            return torch.tensor(data_item[0])

        return transform_fn

    @staticmethod
    def map_references(ref_biases: Dict, model_cls: Any) -> Dict[str, List]:
        if model_cls is ConvTestModel or model_cls is DepthwiseConvTestModel:
            return {"conv2d": ref_biases["/conv/Conv"]}
        if model_cls is TransposeConvTestModel:
            return {"conv_transpose2d": ref_biases["/conv/ConvTranspose"]}
        mapping = dict()
        for name, value in ref_biases.items():
            conv_idx = int(name[re.search(r"\d", name).start()])
            conv_idx -= 1
            conv_idx = "_" + str(conv_idx) if conv_idx else ""
            fx_name = "conv2d" + conv_idx
            mapping[fx_name] = value
        return mapping

    @staticmethod
    def remove_fq_from_inputs(model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        graph = GraphConverter.create_nncf_graph(model)
        return remove_fq_from_inputs(model, graph)

    @staticmethod
    def check_bias(model: ov.Model, ref_biases: Dict) -> None:
        nncf_graph = NNCFGraphFactory.create(model)
        for ref_name, ref_value in ref_biases.items():
            node = nncf_graph.get_node_by_name(ref_name)
            ref_value = torch.tensor(ref_value)
            curr_value = get_bias_value(node, nncf_graph, model).data
            curr_value = curr_value.reshape(ref_value.shape)
            assert all(torch.isclose(curr_value, ref_value, atol=0.01)), f"{curr_value} != {ref_value}"

    @pytest.mark.parametrize(
        "layer_name, ref_data",
        (
            (
                "conv2d",
                {
                    "collected_inputs": {
                        ("concat", 1): ("arg0_1", 0),
                        ("conv2d", 0): ("concat", 0),
                    },
                    "subgraph_data": {
                        "subgraph_input_ids": {("conv2d", 0)},
                        "subgraph_output_ids": {("getitem", 0), ("max_pool2d", 0), ("getitem_1", 0)},
                    },
                },
            ),
            (
                "conv2d_1",
                {
                    "collected_inputs": {
                        ("conv2d", 0): ("concat", 0),
                        ("conv2d_1", 0): ("max_pool2d", 0),
                        ("conv2d_3", 0): ("getitem", 0),
                        ("conv2d_5", 0): ("getitem_1", 0),
                    },
                    "subgraph_data": {
                        "subgraph_input_ids": {("conv2d_1", 0)},
                        "subgraph_output_ids": {("relu_1", 0)},
                    },
                },
            ),
            (
                "conv2d_2",
                {
                    "collected_inputs": {
                        ("conv2d", 0): ("concat", 0),
                        ("conv2d_1", 0): ("max_pool2d", 0),
                        ("conv2d_2", 0): ("relu_1", 0),
                        ("conv2d_3", 0): ("getitem", 0),
                        ("conv2d_5", 0): ("getitem_1", 0),
                    },
                    "subgraph_data": {
                        "subgraph_input_ids": {("conv2d", 0), ("conv2d_2", 0)},
                        "subgraph_output_ids": {("getitem", 0), ("getitem_1", 0)},
                    },
                },
            ),
            (
                "conv2d_3",
                {
                    "collected_inputs": {
                        ("conv2d_3", 0): ("getitem", 0),
                        ("conv2d_5", 0): ("getitem_1", 0),
                    },
                    "subgraph_data": {
                        "subgraph_input_ids": {("conv2d_3", 0)},
                        "subgraph_output_ids": {("relu_2", 0)},
                    },
                },
            ),
            (
                "conv2d_5",
                {
                    "collected_inputs": {
                        ("conv2d_4", 0): ("relu_2", 0),
                        ("conv2d_5", 0): ("getitem_1", 0),
                    },
                    "subgraph_data": {
                        "subgraph_input_ids": {("conv2d_4", 0), ("conv2d_5", 0)},
                        "subgraph_output_ids": {("add_3", 0), ("concat_1", 0)},
                    },
                },
            ),
            (
                "conv2d_9",
                {
                    "collected_inputs": {
                        ("conv2d_7", 0): ("conv_6", 0),
                        ("conv2d_8", 0): ("add_2", 0),
                        ("conv2d_9", 0): ("concat", 0),
                    },
                    "subgraph_data": {
                        "subgraph_input_ids": {
                            ("conv2d_7", 0),
                            ("conv2d_8", 0),
                            ("conv2d_9", 0),
                        },
                        "subgraph_output_ids": {("concat_2", 0)},
                    },
                },
            ),
            (
                "matmul",
                {
                    "collected_inputs": {
                        ("matmul", 0): ("reshape", 0),
                    },
                    "subgraph_data": {
                        "subgraph_input_ids": {("matmul", 0)},
                        "subgraph_output_ids": {("reshape_1", 0), ("add_4", 0)},
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
                    ("conv2d", 0): ("concat", 0),
                    ("concat", 1): ("x", 0),
                },
            ),
            (
                MultipleConvTestModel,
                {
                    ("conv2d", 0): ("x", 0),
                    ("conv2d_2", 0): ("x", 0),
                },
            ),
            (ConvTestModel, {("conv2d", 0): ("x", 0)}),
            (DepthwiseConvTestModel, {("conv2d", 0): ("x", 0)}),
            (TransposeConvTestModel, {("conv_transpose2d", 0): ("x", 0)}),
        ),
    )
    def test_verify_collected_stat_inputs_map(self, model_cls, ref_stat_inputs_map, tmpdir):
        return super().test_verify_collected_stat_inputs_map(model_cls, ref_stat_inputs_map, tmpdir)
