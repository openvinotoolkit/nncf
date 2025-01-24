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
import openvino.runtime as ov
import pytest
from openvino.runtime import opset13 as opset

from nncf import Dataset
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.commands import TargetType
from nncf.experimental.common.tensor_statistics.collectors import TensorReducerBase
from nncf.openvino.graph.transformations.commands import OVTargetPoint
from nncf.openvino.statistics.aggregator import OVStatisticsAggregator
from nncf.openvino.statistics.collectors import OV_REDUCERS_MAP
from nncf.openvino.statistics.collectors import OVBatchMeanReducer
from nncf.openvino.statistics.collectors import OVMeanPerChanelReducer
from nncf.quantization.algorithms.bias_correction.openvino_backend import OVBiasCorrectionAlgoBackend
from nncf.quantization.algorithms.fast_bias_correction.openvino_backend import OVFastBiasCorrectionAlgoBackend
from nncf.quantization.algorithms.min_max.openvino_backend import OVMinMaxAlgoBackend
from tests.common.test_statistics_aggregator import TemplateTestStatisticsAggregator
from tests.openvino.native.models import SharedConvModel
from tests.openvino.native.models import SplitConcatModel

INPUT_NAME = "Input"
CONV_NODE_NAME = "Conv1"
INPUT_SHAPE = [1, 3, 3, 3]


def get_StatisticAggregatorTestModel(input_shape, kernel):
    input_1 = opset.parameter(input_shape, name=INPUT_NAME)
    strides = [1, 1]
    pads = [0, 0]
    dilations = [1, 1]
    conv = opset.convolution(input_1, kernel.astype(np.float32), strides, pads, pads, dilations, name=CONV_NODE_NAME)

    result = opset.result(conv, name="Result")
    model = ov.Model([result], [input_1])
    return model


class TestStatisticsAggregator(TemplateTestStatisticsAggregator):
    @staticmethod
    def get_min_max_algo_backend_cls() -> Type[OVMinMaxAlgoBackend]:
        return OVMinMaxAlgoBackend

    def get_bias_correction_algo_backend_cls(self) -> Type[OVBiasCorrectionAlgoBackend]:
        return OVBiasCorrectionAlgoBackend

    def get_fast_bias_correction_algo_backend_cls(self) -> Type[OVFastBiasCorrectionAlgoBackend]:
        return OVFastBiasCorrectionAlgoBackend

    def get_backend_model(self, dataset_samples):
        sample = dataset_samples[0].reshape(INPUT_SHAPE[1:])
        conv_w = self.dataset_samples_to_conv_w(sample)
        return get_StatisticAggregatorTestModel(INPUT_SHAPE, conv_w)

    @pytest.fixture(scope="session")
    def test_params(self):
        return {
            "test_statistic_merging": {
                "split_concat": {"model": self._get_split_concat_backend_model},
                "shared_conv": {"model": self._get_shared_conv_model},
            }
        }

    def get_statistics_aggregator(self, dataset):
        return OVStatisticsAggregator(dataset)

    def get_target_point_cls(self):
        return OVTargetPoint

    def get_dataset(self, samples):
        return Dataset(samples, lambda data: {INPUT_NAME: data})

    @staticmethod
    def get_target_point(target_type: TargetType) -> TargetPoint:
        target_node_name = INPUT_NAME
        port_id = 0
        if target_type == TargetType.OPERATION_WITH_WEIGHTS:
            target_node_name = CONV_NODE_NAME
            port_id = 1
        if target_type == TargetType.PRE_LAYER_OPERATION:
            target_node_name = CONV_NODE_NAME
        return OVTargetPoint(target_type, target_node_name, port_id)

    @pytest.fixture
    def dataset_samples(self, dataset_values):
        input_shape = INPUT_SHAPE
        dataset_samples = [np.zeros(input_shape), np.ones(input_shape)]

        for i, value in enumerate(dataset_values):
            dataset_samples[0][0, i, 0, 0] = value["max"]
            dataset_samples[0][0, i, 0, 1] = value["min"]
        return dataset_samples

    @pytest.fixture
    def is_backend_support_custom_estimators(self) -> bool:
        return True

    @pytest.fixture(params=[True, False], ids=["inplace", "out_of_place"])
    def inplace_statistics(self, request) -> bool:
        return request.param

    def _get_split_concat_backend_model(self, dataset_samples):
        return SplitConcatModel(input_name=INPUT_NAME).ov_model

    def _get_shared_conv_model(self, dataset_samples):
        sample = dataset_samples[0].reshape(INPUT_SHAPE[1:])
        conv_w = self.dataset_samples_to_conv_w(sample)
        return SharedConvModel(input_name=INPUT_NAME, input_shape=INPUT_SHAPE, kernel=conv_w).ov_model

    def reducers_map(self) -> List[TensorReducerBase]:
        map_ = OV_REDUCERS_MAP.copy()
        map_.update({"batch_mean": OVBatchMeanReducer, "mean_per_ch": OVMeanPerChanelReducer})
        return map_
