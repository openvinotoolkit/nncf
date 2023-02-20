"""
 Copyright (c) 2023 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import pytest
import numpy as np
import openvino.runtime as ov
from openvino.runtime import opset9 as opset

from nncf import Dataset
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.experimental.openvino_native.statistics.aggregator import OVStatisticsAggregator
from nncf.experimental.openvino_native.graph.transformations.commands import OVTargetPoint
from nncf.experimental.openvino_native.quantization.algorithms.min_max.openvino_backend import\
    OVMinMaxAlgoBackend

from tests.common.test_statistics_aggregator import TemplateTestStatisticsAggregator


INPUT_NAME = 'Input'
CONV_NODE_NAME = 'Conv1'
INPUT_SHAPE = [1, 3, 3, 3]


def get_StatisticAgregatorTestModel(input_shape, kernel):
    input_1 = opset.parameter(input_shape, name=INPUT_NAME)
    strides = [1, 1]
    pads = [0, 0]
    dilations = [1, 1]
    conv = opset.convolution(input_1, kernel.astype(np.float32),
                             strides, pads, pads, dilations, name=CONV_NODE_NAME)

    result = opset.result(conv, name="Result")
    model = ov.Model([result], [input_1])
    return model


class TestStatisticsAggregator(TemplateTestStatisticsAggregator):
    def get_algo_backend_cls(self) -> OVMinMaxAlgoBackend:
        return OVMinMaxAlgoBackend

    def get_backend_model(self, dataset_samples):
        sample = dataset_samples[0].reshape(INPUT_SHAPE[1:])
        conv_w = self.dataset_samples_to_conv_w(sample)
        return get_StatisticAgregatorTestModel(INPUT_SHAPE, conv_w)

    def get_statistics_aggregator(self, dataset):
        return OVStatisticsAggregator(dataset)

    def get_dataset(self, samples):
        return Dataset(samples, lambda data: {INPUT_NAME: data})

    def get_target_point(self, target_type: TargetType) -> TargetPoint:
        target_node_name = INPUT_NAME
        port_id = 0
        if target_type == TargetType.OPERATION_WITH_WEIGHTS:
            target_node_name = CONV_NODE_NAME
            port_id = 1
        return OVTargetPoint(target_type, target_node_name, port_id)

    @pytest.fixture
    def dataset_samples(self, dataset_values):
        input_shape = INPUT_SHAPE
        dataset_samples = [np.zeros(input_shape), np.ones(input_shape)]

        for i, value in enumerate(dataset_values):
            dataset_samples[0][0, i, 0, 0] = value['max']
            dataset_samples[0][0, i, 0, 1] = value['min']
        return dataset_samples

    @pytest.fixture
    def is_stat_in_shape_of_scale(self) -> bool:
        return True
