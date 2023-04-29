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

from typing import Type

import numpy as np
import pytest

from nncf import Dataset
from nncf.common.graph.transformations.commands import TargetType
from nncf.onnx.graph.transformations.commands import ONNXTargetPoint
from nncf.onnx.statistics.aggregator import ONNXStatisticsAggregator
from nncf.quantization.algorithms.bias_correction.onnx_backend import ONNXBiasCorrectionAlgoBackend
from nncf.quantization.algorithms.fast_bias_correction.onnx_backend import ONNXFastBiasCorrectionAlgoBackend
from nncf.quantization.algorithms.min_max.onnx_backend import ONNXMinMaxAlgoBackend
from tests.common.test_statistics_aggregator import TemplateTestStatisticsAggregator
from tests.onnx.models import IdentityConvolutionalModel

INPUT_NAME = "X"
IDENTITY_NODE_NAME = "Identity"
CONV_NODE_NAME = "Conv1"
INPUT_SHAPE = [3, 3, 3]


class TestStatisticsAggregator(TemplateTestStatisticsAggregator):
    def get_min_max_algo_backend_cls(self) -> Type[ONNXMinMaxAlgoBackend]:
        return ONNXMinMaxAlgoBackend

    def get_bias_correction_algo_backend_cls(self) -> Type[ONNXBiasCorrectionAlgoBackend]:
        return ONNXBiasCorrectionAlgoBackend

    def get_fast_bias_correction_algo_backend_cls(self) -> Type[ONNXFastBiasCorrectionAlgoBackend]:
        return ONNXFastBiasCorrectionAlgoBackend

    def get_backend_model(self, dataset_samples):
        conv_w = self.dataset_samples_to_conv_w(dataset_samples[0])
        return IdentityConvolutionalModel(
            input_shape=[1] + INPUT_SHAPE, inp_ch=3, out_ch=3, kernel_size=3, conv_w=conv_w
        ).onnx_model

    def get_statistics_aggregator(self, dataset):
        return ONNXStatisticsAggregator(dataset)

    @pytest.fixture
    def is_backend_support_custom_estimators(self) -> bool:
        return False

    @pytest.fixture(scope="session")
    def test_params(self):
        return

    def get_dataset(self, samples):
        def transform_fn(data_item):
            inputs = data_item
            return {INPUT_NAME: [inputs]}

        return Dataset(samples, transform_fn)

    def get_target_point(self, target_type: TargetType):
        target_node_name = IDENTITY_NODE_NAME
        port_id = 0
        if target_type == TargetType.OPERATION_WITH_WEIGHTS:
            target_node_name = CONV_NODE_NAME
            port_id = 1
        return ONNXTargetPoint(target_type, target_node_name, port_id)

    def get_target_point_cls(self):
        return ONNXTargetPoint

    @pytest.fixture
    def dataset_samples(self, dataset_values):
        input_shape = INPUT_SHAPE
        dataset_samples = [np.zeros(input_shape), np.ones(input_shape)]

        for i, value in enumerate(dataset_values):
            dataset_samples[0][i, 0, 0] = value["max"]
            dataset_samples[0][i, 0, 1] = value["min"]

        return dataset_samples

    @pytest.fixture
    def is_stat_in_shape_of_scale(self) -> bool:
        return False

    @pytest.fixture(params=[False], ids=["out_of_palce"])
    def inplace_statistics(self, request) -> bool:
        return request.param

    @pytest.mark.skip("Merging is not implemented yet")
    def test_statistics_merging_simple(self, dataset_samples, inplace_statistics):
        pass

    @pytest.mark.skip("Merging is not implemented yet")
    def test_statistic_merging(self, dataset_samples, inplace_statistics):
        pass
