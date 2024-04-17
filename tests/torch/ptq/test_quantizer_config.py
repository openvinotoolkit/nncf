# Copyright (c) 2024 Intel Corporation
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

from nncf.quantization.algorithms.min_max.torch_backend import PTMinMaxAlgoBackend
from tests.post_training.test_templates.models import NNCFGraphToTest
from tests.post_training.test_templates.models import NNCFGraphToTestDepthwiseConv
from tests.post_training.test_templates.models import NNCFGraphToTestSumAggregation
from tests.post_training.test_templates.test_quantizer_config import TemplateTestQuantizerConfig
from tests.torch.ptq.helpers import get_depthwise_conv_nncf_graph
from tests.torch.ptq.helpers import get_single_conv_nncf_graph
from tests.torch.ptq.helpers import get_sum_aggregation_nncf_graph


class TestQuantizerConfig(TemplateTestQuantizerConfig):
    def get_algo_backend(self):
        return PTMinMaxAlgoBackend()

    @pytest.fixture
    def single_conv_nncf_graph(self) -> NNCFGraphToTest:
        return get_single_conv_nncf_graph()

    @pytest.fixture
    def depthwise_conv_nncf_graph(self) -> NNCFGraphToTestDepthwiseConv:
        return get_depthwise_conv_nncf_graph()

    @pytest.fixture
    def conv_sum_aggregation_nncf_graph(self) -> NNCFGraphToTestSumAggregation:
        return get_sum_aggregation_nncf_graph()
