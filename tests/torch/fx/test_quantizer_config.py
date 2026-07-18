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

import pytest
import torch

import nncf.torch.graph.operator_metatypes as om
from nncf.common.graph.patterns import GraphPattern
from nncf.common.utils.backend import BackendType
from nncf.experimental.torch.fx.nncf_graph_builder import GraphConverter
from nncf.quantization.algorithms.min_max.algorithm import MinMaxQuantization
from nncf.quantization.algorithms.min_max.torch_fx_backend import FXMinMaxAlgoBackend
from nncf.quantization.passes import transform_to_inference_graph
from tests.cross_fw.test_templates.models import NNCFGraphArithmeticDegree2
from tests.cross_fw.test_templates.models import NNCFGraphConstantBranchWithWeightedNode
from tests.cross_fw.test_templates.models import NNCFGraphModelWithEmbeddingsConstantPath
from tests.cross_fw.test_templates.models import NNCFGraphToTest
from tests.cross_fw.test_templates.models import NNCFGraphToTestDepthwiseConv
from tests.cross_fw.test_templates.models import NNCFGraphToTestSumAggregation
from tests.cross_fw.test_templates.models import NNCFGraphTransformer
from tests.cross_fw.test_templates.models import NNCFSplitGraphTransformer
from tests.cross_fw.test_templates.test_quantizer_config import TemplateTestQuantizerConfig
from tests.torch.function_hook.quantization.helper import get_single_conv_arithmetic_degree2_nncf_graph
from tests.torch.fx.helpers import get_depthwise_conv_nncf_graph
from tests.torch.fx.helpers import get_single_conv_nncf_graph
from tests.torch.fx.helpers import get_sum_aggregation_nncf_graph
from tests.torch.fx.helpers import get_torch_fx_model


class SplitGetItemDifferentNumelModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.producer = torch.nn.Conv2d(3, 4, 1, bias=False)
        self.consumer_a = torch.nn.Conv2d(1, 2, 1, bias=False)
        self.consumer_b = torch.nn.Conv2d(3, 2, 1, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = self.producer(x)
        chunks = torch.split(y, [1, 3], dim=1)
        return self.consumer_a(chunks[0]), self.consumer_b(chunks[1])


class SplitGetItemSameNumelModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.producer = torch.nn.Conv2d(3, 3, 1, bias=False)
        self.consumer = torch.nn.Conv2d(3, 2, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.producer(x)
        chunks = torch.split(y, [3], dim=1)
        return self.consumer(chunks[0])


class TestQuantizerConfig(TemplateTestQuantizerConfig):
    def get_algo_backend(self):
        return FXMinMaxAlgoBackend()

    def get_backend_type(self):
        return BackendType.TORCH_FX

    @pytest.fixture
    def single_conv_nncf_graph(self) -> NNCFGraphToTest:
        return get_single_conv_nncf_graph()

    @pytest.fixture
    def single_conv_arithmetic_degree2_nncf_graph(self) -> NNCFGraphArithmeticDegree2:
        return get_single_conv_arithmetic_degree2_nncf_graph()

    @pytest.fixture
    def depthwise_conv_nncf_graph(self) -> NNCFGraphToTestDepthwiseConv:
        return get_depthwise_conv_nncf_graph()

    @pytest.fixture
    def conv_sum_aggregation_nncf_graph(self) -> NNCFGraphToTestSumAggregation:
        return get_sum_aggregation_nncf_graph()

    @pytest.fixture
    def transformer_nncf_graph(self) -> NNCFGraphToTest:
        return NNCFGraphTransformer(
            matmul_metatype=om.PTMatMulMetatype,
            softmax_metatype=om.PTSoftmaxMetatype,
            mul_metatype=om.PTMulMetatype,
            const_metatype=om.PTConstNoopMetatype,
            transpose_metatype=om.PTTransposeMetatype,
        )

    @pytest.fixture
    def split_transformer_nncf_graph(self) -> NNCFSplitGraphTransformer:
        return NNCFSplitGraphTransformer(
            matmul_metatype=om.PTMatMulMetatype,
            conv_metatype=om.PTConv2dMetatype,
            split_metatype=om.PTSplitMetatype,
            softmax_metatype=om.PTSoftmaxMetatype,
            const_metatype=om.PTConstNoopMetatype,
            mul_metatype=om.PTMulMetatype,
        )

    @pytest.fixture
    def embedding_nncf_graph_shape_of(self) -> NNCFGraphToTest:
        return None

    @pytest.mark.skip("Torch does not have shape of subgraphs")
    def test_embedding_model_qconfig_shape_of(self, embedding_nncf_graph_shape_of):
        pass

    @pytest.fixture
    def embedding_nncf_graph_constant_path(self) -> NNCFGraphToTest:
        return NNCFGraphModelWithEmbeddingsConstantPath(
            const_metatype=om.PTConstNoopMetatype,
            embedding_metatype=om.PTEmbeddingMetatype,
            conv_metatype=om.PTConv2dMetatype,
            add_metatype=om.PTAddMetatype,
        )

    @pytest.fixture
    def constant_branch_nncf_graph(self) -> NNCFGraphToTest:
        return NNCFGraphConstantBranchWithWeightedNode(
            const_metatype=om.PTConstNoopMetatype,
            conv_metatype=om.PTConv2dMetatype,
            add_metatype=om.PTAddMetatype,
        )

    @staticmethod
    def _export_model_and_get_q_setup(model: torch.nn.Module):
        exported_model = get_torch_fx_model(model, torch.ones((1, 3, 4, 4)))
        nncf_graph = GraphConverter.create_nncf_graph(exported_model)

        min_max_algo = MinMaxQuantization()
        min_max_algo._backend_entity = FXMinMaxAlgoBackend()
        inference_nncf_graph = transform_to_inference_graph(
            nncf_graph,
            min_max_algo._backend_entity.get_start_nodes_for_activation_path_tracing(nncf_graph),
            min_max_algo._backend_entity.shapeof_metatypes,
            min_max_algo._backend_entity.noop_metatypes,
            min_max_algo._backend_entity.preserved_metatypes,
        )
        q_setup = min_max_algo._get_quantizer_setup(
            nncf_graph, inference_nncf_graph, hw_patterns=GraphPattern(), ignored_patterns=GraphPattern()
        )
        return exported_model, q_setup

    @staticmethod
    def _get_fx_getitem_node_names(exported_model: torch.fx.GraphModule) -> list[str]:
        return [
            node.name
            for node in exported_model.graph.nodes
            if node.op == "call_function" and node.target.__name__ == "getitem"
        ]

    @staticmethod
    def _get_activation_qpoints(q_setup):
        return {
            (qp.insertion_point.target_node_name, qp.insertion_point.input_port_id): sorted(
                qp.directly_quantized_operator_node_names
            )
            for qp in q_setup.quantization_points.values()
            if qp.is_activation_quantization_point()
        }

    def test_getitem_different_numel_stops_quantizer_propagation(self):
        exported_model, q_setup = self._export_model_and_get_q_setup(SplitGetItemDifferentNumelModel())

        assert self._get_fx_getitem_node_names(exported_model) == ["getitem", "getitem_1"]

        activation_qpoints = self._get_activation_qpoints(q_setup)
        expected_consumer_input_qpoints = {("conv2d_1", 0), ("conv2d_2", 0)}
        assert expected_consumer_input_qpoints <= set(activation_qpoints), activation_qpoints
        assert activation_qpoints[("conv2d_1", 0)] == ["conv2d_1"]
        assert activation_qpoints[("conv2d_2", 0)] == ["conv2d_2"]
        assert ("conv2d", None) not in activation_qpoints

    def test_getitem_same_numel_preserves_quantizer_propagation(self):
        exported_model, q_setup = self._export_model_and_get_q_setup(SplitGetItemSameNumelModel())

        assert self._get_fx_getitem_node_names(exported_model) == ["getitem"]

        activation_qpoints = self._get_activation_qpoints(q_setup)
        assert activation_qpoints[("conv2d", None)] == ["conv2d_1"]
        assert ("conv2d_1", 0) not in activation_qpoints
