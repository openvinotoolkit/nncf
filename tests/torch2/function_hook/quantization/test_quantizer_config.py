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

import pytest

import nncf.torch.graph.operator_metatypes as om
from nncf.common.utils.backend import BackendType
from nncf.quantization.algorithms.min_max.torch_backend import PTMinMaxAlgoBackend
from nncf.torch.graph.graph import PTNNCFGraph
from tests.cross_fw.test_templates.models import NNCFGraphConstantBranchWithWeightedNode
from tests.cross_fw.test_templates.models import NNCFGraphModelWithEmbeddingsConstantPath
from tests.cross_fw.test_templates.models import NNCFGraphToTest
from tests.cross_fw.test_templates.models import NNCFGraphToTestDepthwiseConv
from tests.cross_fw.test_templates.models import NNCFGraphToTestSumAggregation
from tests.cross_fw.test_templates.models import NNCFGraphTransformer
from tests.cross_fw.test_templates.test_quantizer_config import TemplateTestQuantizerConfig
from tests.torch2.function_hook.quantization.helper import get_depthwise_conv_nncf_graph
from tests.torch2.function_hook.quantization.helper import get_single_conv_nncf_graph
from tests.torch2.function_hook.quantization.helper import get_sum_aggregation_nncf_graph


class TestQuantizerConfig(TemplateTestQuantizerConfig):
    def get_algo_backend(self):
        return PTMinMaxAlgoBackend()

    def get_backend_type(self):
        return BackendType.TORCH

    @pytest.fixture
    def single_conv_nncf_graph(self) -> NNCFGraphToTest:
        return get_single_conv_nncf_graph()

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
            nncf_graph_cls=PTNNCFGraph,
        )

    @pytest.fixture
    def embedding_nncf_graph_shape_of(self) -> NNCFGraphToTest:
        return None

    @pytest.mark.skip("Torch2 does not have shape of subgraphs")
    def test_embedding_model_qconfig_shape_of(self, embedding_nncf_graph_shape_of):
        pass

    @pytest.fixture
    def embedding_nncf_graph_constant_path(self) -> NNCFGraphToTest:
        return NNCFGraphModelWithEmbeddingsConstantPath(
            const_metatype=om.PTConstNoopMetatype,
            embedding_metatype=om.PTEmbeddingMetatype,
            conv_metatype=om.PTConv2dMetatype,
            add_metatype=om.PTAddMetatype,
            nncf_graph_cls=PTNNCFGraph,
        )

    @pytest.fixture
    def constant_branch_nncf_graph(self) -> NNCFGraphToTest:
        return NNCFGraphConstantBranchWithWeightedNode(
            const_metatype=om.PTConstNoopMetatype,
            conv_metatype=om.PTConv2dMetatype,
            add_metatype=om.PTAddMetatype,
            nncf_graph_cls=PTNNCFGraph,
        )
