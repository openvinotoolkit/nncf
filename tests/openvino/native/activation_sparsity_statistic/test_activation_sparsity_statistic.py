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
import numpy as np
import pytest

import nncf
from nncf.common.tensor import NNCFTensor
from nncf.experimental.openvino_native.activation_sparsity_statistic.activation_sparsity_statistic import \
    activation_sparsity_statistic_impl
from nncf.experimental.openvino_native.activation_sparsity_statistic.algorithm import ActivationSparsityStatistic
from nncf.experimental.openvino_native.activation_sparsity_statistic.algorithm import \
    ActivationSparsityStatisticParameters
from nncf.experimental.openvino_native.statistics.collectors import OVNNCFCollectorTensorProcessor
from tests.openvino.native.models import SYNTHETIC_MODELS
from tests.shared.datasets import MockDataset


def test_algo():
    model_to_test = SYNTHETIC_MODELS.get("ConvModel")().ov_model
    dataset = nncf.Dataset(MockDataset([1, 3, 4, 2]), transform_func=lambda x: {"Input_1": x, "Input_2": x})
    activation_sparsity_statistic_impl(model_to_test, dataset, 1)


REF_STATIC_POINTS = {
    "ComparisonBinaryModel": {"Add": 1},
    "ConvModel": {"Conv": 1, "Mul": 1, "Conv_Add": 1, "Transpose": 1},
    "LinearModel": {"Add": 1, "MatMul": 1},
    "MatMul2DModel": {"Add": 1},
}


@pytest.mark.parametrize("model_cls_to_test", SYNTHETIC_MODELS.values())
def test_get_static_points(model_cls_to_test):
    model_to_test = model_cls_to_test().ov_model

    algo = ActivationSparsityStatistic(ActivationSparsityStatisticParameters(number_samples=1))
    statistic_points = algo.get_statistic_points(model_to_test)

    points_info = dict()
    for node_name, points in statistic_points.items():
        points_info[node_name] = len(points)

    assert points_info == REF_STATIC_POINTS[model_cls_to_test.__name__]


@pytest.mark.parametrize("tensor, ref", (([1, 1], 0.0), ([1, 0], 0.5), ([[0, 0], [1, 0]], 0.75), ([0, 0], 1.0)))
def test_percentage_of_zeros_tensor_processor(tensor, ref):
    result = OVNNCFCollectorTensorProcessor.percentage_of_zeros(NNCFTensor(np.array(tensor)))
    assert result == ref


def test_final_statistic_data():
    # TODO: test IR
    pass
