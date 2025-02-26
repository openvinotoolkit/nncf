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

from typing import List, Type

import numpy as np
import pytest
import torch
from torch import nn

from nncf import Dataset
from nncf.common.graph.transformations.commands import TargetType
from nncf.experimental.common.tensor_statistics.collectors import TensorReducerBase
from nncf.experimental.torch.fx.statistics.aggregator import FXStatisticsAggregator
from nncf.quantization.algorithms.fast_bias_correction.torch_fx_backend import FXFastBiasCorrectionAlgoBackend
from nncf.quantization.algorithms.min_max.torch_fx_backend import FXMinMaxAlgoBackend
from nncf.torch.graph.graph import PTTargetPoint
from tests.common.test_statistics_aggregator import TemplateTestStatisticsAggregator
from tests.torch.fx.helpers import get_torch_fx_model

IDENTITY_NODE_NAME = "add"
CONV_NODE_NAME = "conv2d"
INPUT_SHAPE = [1, 3, 3, 3]


class IdentityConv(nn.Module):
    def __init__(self, kernel) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=3,
        )
        self.conv.weight.data = torch.tensor(kernel, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x + 0.0)


class TestStatisticsAggregator(TemplateTestStatisticsAggregator):
    @staticmethod
    def get_min_max_algo_backend_cls() -> Type[FXMinMaxAlgoBackend]:
        return FXMinMaxAlgoBackend

    def get_bias_correction_algo_backend_cls(self) -> None:
        pytest.skip("FXBiasCorrectionAlgoBackend is not implemented")

    def get_fast_bias_correction_algo_backend_cls(self) -> Type[FXFastBiasCorrectionAlgoBackend]:
        return FXFastBiasCorrectionAlgoBackend

    def get_backend_model(self, dataset_samples):
        sample = dataset_samples[0].reshape(INPUT_SHAPE[1:])
        conv_w = self.dataset_samples_to_conv_w(np.array(sample))
        return get_torch_fx_model(IdentityConv(conv_w), torch.ones(INPUT_SHAPE))

    def get_statistics_aggregator(self, dataset):
        return FXStatisticsAggregator(dataset)

    def get_dataset(self, samples):
        def transform_fn(data_item):
            return data_item

        return Dataset(samples, transform_fn)

    @staticmethod
    def get_target_point(target_type: TargetType):
        target_node_name = IDENTITY_NODE_NAME
        port_id = 0
        if target_type == TargetType.OPERATION_WITH_WEIGHTS:
            target_node_name = CONV_NODE_NAME
            port_id = 1
        return FXMinMaxAlgoBackend.target_point(target_type, target_node_name, port_id)

    def get_target_point_cls(self):
        return PTTargetPoint

    @pytest.fixture(scope="session")
    def test_params(self):
        return

    @pytest.fixture
    def dataset_samples(self, dataset_values):
        dataset_samples = [np.zeros(INPUT_SHAPE), np.ones(INPUT_SHAPE)]

        for i, value in enumerate(dataset_values):
            dataset_samples[0][0, i, 0, 0] = value["max"]
            dataset_samples[0][0, i, 0, 1] = value["min"]

        return torch.tensor(dataset_samples, dtype=torch.float32)

    @pytest.fixture(params=[False], ids=["out_of_palce"])
    def inplace_statistics(self, request) -> bool:
        return request.param

    @pytest.fixture
    def is_backend_support_custom_estimators(self) -> bool:
        return True

    def reducers_map(self) -> List[TensorReducerBase]:
        return None

    @pytest.mark.skip("Merging is not implemented yet")
    def test_statistic_merging(self, dataset_samples, inplace_statistics):
        pass

    @pytest.mark.skip("Merging is not implemented yet")
    def test_same_collectors_different_attrs_dont_merge(self, statistics_type, test_params, dataset_samples):
        pass
