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
from nncf.quantization.algorithms.min_max.torch_fx_backend import FXMinMaxAlgoBackend
from tests.cross_fw.test_templates.models import NNCFGraphToTest
from tests.cross_fw.test_templates.models import NNCFGraphToTestDepthwiseConv
from tests.cross_fw.test_templates.models import NNCFGraphToTestSumAggregation
from tests.cross_fw.test_templates.models import NNCFGraphTransformer
from tests.cross_fw.test_templates.test_quantizer_config import TemplateTestQuantizerConfig
from tests.torch.fx.helpers import get_depthwise_conv_nncf_graph
from tests.torch.fx.helpers import get_single_conv_nncf_graph
from tests.torch.fx.helpers import get_sum_aggregation_nncf_graph


class TestQuantizerConfig(TemplateTestQuantizerConfig):
    def get_algo_backend(self):
        return FXMinMaxAlgoBackend()

    def get_backend_type(self):
        return BackendType.TORCH_FX

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
        )
