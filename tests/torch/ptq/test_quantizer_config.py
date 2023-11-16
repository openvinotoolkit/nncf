# Copyright (c) 2023 Intel Corporation
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

from nncf.common.graph.transformations.commands import TargetType
from nncf.common.tensor_statistics.collectors import ReductionAxes
from nncf.experimental.common.tensor_statistics.collectors import MaxAggregator
from nncf.experimental.common.tensor_statistics.collectors import MeanAggregator
from nncf.experimental.common.tensor_statistics.collectors import MinAggregator
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.experimental.common.tensor_statistics.collectors import TensorReducerBase
from nncf.quantization.algorithms.min_max.torch_backend import PTMinMaxAlgoBackend
from tests.post_training.test_templates.models import NNCFGraphToTest
from tests.post_training.test_templates.models import NNCFGraphToTestDepthwiseConv
from tests.post_training.test_templates.models import NNCFGraphToTestSumAggregation
from tests.post_training.test_templates.test_quantizer_config import TemplateTestQuantizerConfig
from tests.torch.ptq.helpers import get_depthwise_conv_nncf_graph
from tests.torch.ptq.helpers import get_single_conv_nncf_graph
from tests.torch.ptq.helpers import get_sum_aggregation_nncf_graph

ParamsCls = TemplateTestQuantizerConfig.TestGetStatisticsCollectorParameters


class TestQuantizerConfig(TemplateTestQuantizerConfig):
    def get_algo_backend(self):
        return PTMinMaxAlgoBackend()

    def check_is_min_max_statistic_collector(self, tensor_collector: TensorCollector):
        aggrs = [aggr.__class__ for aggr in tensor_collector.aggregators.values()]
        assert len(aggrs) == 2
        assert MinAggregator in aggrs
        assert MaxAggregator in aggrs

    def check_is_mean_min_max_statistic_collector(self, tensor_collector: TensorCollector):
        aggrs = [aggr.__class__ for aggr in tensor_collector.aggregators.values()]
        assert len(aggrs) == 2
        assert MeanAggregator in aggrs
        assert aggrs[0].__class__ == aggrs[1].__class__

    def get_reduction_axes(self, reducer: TensorReducerBase) -> ReductionAxes:
        return reducer._reduction_axes

    @pytest.fixture(
        params=[
            (TargetType.PRE_LAYER_OPERATION, "/Sum_1_0", (0, 2), (0, 1, 2)),
            (TargetType.POST_LAYER_OPERATION, "/Conv_1_0", (0, 2, 3), (0, 1, 2, 3)),
            (TargetType.OPERATION_WITH_WEIGHTS, "/Conv_1_0", (1, 2, 3), (0, 1, 2, 3)),
        ]
    )
    def statistic_collector_parameters(self, request) -> ParamsCls:
        return ParamsCls(*request.param)

    @pytest.fixture
    def single_conv_nncf_graph(self) -> NNCFGraphToTest:
        return get_single_conv_nncf_graph()

    @pytest.fixture
    def depthwise_conv_nncf_graph(self) -> NNCFGraphToTestDepthwiseConv:
        return get_depthwise_conv_nncf_graph()

    @pytest.fixture
    def conv_sum_aggregation_nncf_graph(self) -> NNCFGraphToTestSumAggregation:
        return get_sum_aggregation_nncf_graph()
