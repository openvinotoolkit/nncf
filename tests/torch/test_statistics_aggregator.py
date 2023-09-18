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

from typing import List, Type

import numpy as np
import pytest
import torch
from torch import nn

from nncf import Dataset
from nncf.common.graph.transformations.commands import TargetType
from nncf.experimental.common.tensor_statistics.collectors import TensorReducerBase
from nncf.quantization.algorithms.fast_bias_correction.torch_backend import PTFastBiasCorrectionAlgoBackend
from nncf.quantization.algorithms.min_max.torch_backend import PTMinMaxAlgoBackend
from nncf.torch.graph.graph import PTTargetPoint
from nncf.torch.statistics.aggregator import PTStatisticsAggregator
from tests.common.test_statistics_aggregator import TemplateTestStatisticsAggregator
from tests.torch.ptq.helpers import get_nncf_network
from tests.torch.ptq.test_ptq_params import ToNNCFNetworkInterface

IDENTITY_NODE_NAME = "PTIdentityConvModel/__add___0"
CONV_NODE_NAME = "PTIdentityConvModel/NNCFConv2d[conv]/conv2d_0"
INPUT_SHAPE = [1, 3, 3, 3]


class PTIdentityConvModel(nn.Module, ToNNCFNetworkInterface):
    def __init__(self, kernel):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3)
        self.conv.weight.data = torch.tensor(kernel, dtype=torch.float32)

    def forward(self, x):
        return self.conv(x + 0.0)

    def get_nncf_network(self):
        return get_nncf_network(self, INPUT_SHAPE)


class TestStatisticsAggregator(TemplateTestStatisticsAggregator):
    def get_min_max_algo_backend_cls(self) -> Type[PTMinMaxAlgoBackend]:
        return PTMinMaxAlgoBackend

    def get_bias_correction_algo_backend_cls(self) -> None:
        pytest.skip("PTBiasCorrectionAlgoBackend is not implemented")

    def get_fast_bias_correction_algo_backend_cls(self) -> Type[PTFastBiasCorrectionAlgoBackend]:
        return PTFastBiasCorrectionAlgoBackend

    def get_backend_model(self, dataset_samples):
        sample = dataset_samples[0].reshape(INPUT_SHAPE[1:])
        conv_w = self.dataset_samples_to_conv_w(np.array(sample))
        return PTIdentityConvModel(conv_w).get_nncf_network()

    @pytest.fixture
    def is_backend_support_custom_estimators(self) -> bool:
        return False

    @pytest.fixture(scope="session")
    def test_params(self):
        return

    def get_statistics_aggregator(self, dataset):
        return PTStatisticsAggregator(dataset)

    def get_dataset(self, samples):
        def transform_fn(data_item):
            return data_item

        return Dataset(samples, transform_fn)

    def get_target_point(self, target_type: TargetType):
        target_node_name = IDENTITY_NODE_NAME
        port_id = 0
        if target_type == TargetType.OPERATION_WITH_WEIGHTS:
            target_node_name = CONV_NODE_NAME
            port_id = None
        return PTMinMaxAlgoBackend.target_point(target_type, target_node_name, port_id)

    def get_target_point_cls(self):
        return PTTargetPoint

    def reducers_map(self) -> List[TensorReducerBase]:
        return None

    @pytest.fixture
    def dataset_samples(self, dataset_values):
        input_shape = INPUT_SHAPE
        dataset_samples = [np.zeros(input_shape), np.ones(input_shape)]

        for i, value in enumerate(dataset_values):
            dataset_samples[0][0, i, 0, 0] = value["max"]
            dataset_samples[0][0, i, 0, 1] = value["min"]

        return torch.tensor(dataset_samples, dtype=torch.float32)

    @pytest.fixture
    def is_stat_in_shape_of_scale(self) -> bool:
        return True

    @pytest.fixture(params=[False], ids=["out_of_palce"])
    def inplace_statistics(self, request) -> bool:
        return request.param

    @pytest.mark.skip("Merging is not implemented yet")
    def test_statistics_merging_simple(self, dataset_samples, inplace_statistics, statistic_point_params):
        pass

    @pytest.mark.skip("Merging is not implemented yet")
    def test_statistic_merging(self, dataset_samples, inplace_statistics):
        pass

    @pytest.mark.skip("Merging is not implemented yet")
    def test_same_collectors_different_attrs_dont_merge(self, statistics_type, test_params, dataset_samples):
        pass
