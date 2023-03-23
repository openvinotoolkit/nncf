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

    def get_split_concat_backend_model(self):
        input_1 = opset.parameter([1, 3, 3, 3], name=INPUT_NAME)
        split = opset.split(input_1, 1, 3, name='split')
        add_const = np.array(1).astype(np.float32)
        add_1 = opset.add(split.output(0), add_const, name='add_1')
        add_2 = opset.add(split.output(1), add_const, name='add_2')
        add_3 = opset.add(split.output(2), add_const, name='add_3')
        concat = opset.concat([add_1, add_2, add_3], 1, name='concat')
        add_4 = opset.add(concat, add_const, name='add_4')
        add_5 = opset.add(concat, add_const, name='add_5')
        result_1 = opset.result(add_4, name="result_1")
        result_2 = opset.result(add_5, name="result_2")
        model = ov.Model([result_1, result_2], [input_1])
        return model

    def get_split_concat_target_points_and_refs(self):
        return [
            # Split output target points
            (OVTargetPoint(TargetType.POST_LAYER_OPERATION, 'split', 0),
             {'min_max': (-10, 10), 'mean_min_max': (-4.5, 5.5)}),
            (OVTargetPoint(TargetType.PRE_LAYER_OPERATION, 'add_1', 0),
             {'min_max': (-10, 10), 'mean_min_max': (-4.5, 5.5)}),

            (OVTargetPoint(TargetType.POST_LAYER_OPERATION, 'split', 1),
             {'min_max': (-1, 1), 'mean_min_max': (0, 1)}),
            (OVTargetPoint(TargetType.PRE_LAYER_OPERATION, 'add_2', 0),
             {'min_max': (-1, 1), 'mean_min_max': (0, 1)}),

            (OVTargetPoint(TargetType.POST_LAYER_OPERATION, 'split', 2),
             {'min_max': (-128, 128), 'mean_min_max': (-63.5, 64.5)}),
            (OVTargetPoint(TargetType.PRE_LAYER_OPERATION, 'add_3', 0),
             {'min_max': (-128, 128), 'mean_min_max': (-63.5, 64.5)}),

            # Concat input target points
            (OVTargetPoint(TargetType.POST_LAYER_OPERATION, 'add_1', 0),
             {'min_max': (-9, 9), 'mean_min_max': (-3.5, 5.5)}),
            (OVTargetPoint(TargetType.PRE_LAYER_OPERATION, 'concat', 0),
             {'min_max': (-9, 9), 'mean_min_max': (-3.5, 5.5)}),

            (OVTargetPoint(TargetType.POST_LAYER_OPERATION, 'add_2', 0),
             {'min_max': (0, 2), 'mean_min_max': (1, 1.55)}),
            (OVTargetPoint(TargetType.PRE_LAYER_OPERATION, 'concat', 1),
             {'min_max': (0, 2), 'mean_min_max': (1, 1.55)}),

            (OVTargetPoint(TargetType.POST_LAYER_OPERATION, 'add_3', 0),
             {'min_max': (-127, 129), 'mean_min_max': (-62.5, 65.5)}),
            (OVTargetPoint(TargetType.PRE_LAYER_OPERATION, 'concat', 2),
             {'min_max': (-127, 129), 'mean_min_max': (-62.5, 65.5)}),

            # One output to Several branch target points
            (OVTargetPoint(TargetType.POST_LAYER_OPERATION, 'concat', 0),
             {'min_max': (-127, 129), 'mean_min_max': (-62.5, 65.5)}),
            (OVTargetPoint(TargetType.PRE_LAYER_OPERATION, 'add_4', 0),
             {'min_max': (-127, 129), 'mean_min_max': (-62.5, 65.5)}),
            (OVTargetPoint(TargetType.PRE_LAYER_OPERATION, 'add_5', 0),
             {'min_max': (-127, 129), 'mean_min_max': (-62.5, 65.5)}),
        ]

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
        if target_type == TargetType.PRE_LAYER_OPERATION:
            target_node_name = CONV_NODE_NAME
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

    @pytest.fixture(params=[True, False],
                    ids=['inplace', 'out_of_place'])
    def inplace_statistics(self, request) -> bool:
        return request.param
