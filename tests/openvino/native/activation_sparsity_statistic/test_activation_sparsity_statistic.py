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
    estimate_activation_sparsity
from nncf.experimental.openvino_native.activation_sparsity_statistic.algorithm import ActivationSparsityStatistic
from nncf.experimental.openvino_native.activation_sparsity_statistic.algorithm import \
    ActivationSparsityStatisticParameters
from nncf.experimental.openvino_native.activation_sparsity_statistic.ov_backend import ACTIVATION_SPARSITY_STATISTIC
from nncf.experimental.openvino_native.statistics.collectors import OVNNCFCollectorTensorProcessor
from tests.openvino.native.models import SYNTHETIC_MODELS
from tests.shared.datasets import MockDataset


def test_algo():
    model = SYNTHETIC_MODELS.get("ConvModel")().ov_model
    dataset = nncf.Dataset(
        MockDataset([1, 3, 4, 2]), transform_func=lambda x: {"Input_1": x, "Input_2": x.reshape(1, 3, 2, 4)}
    )
    activation_sparsity_stats = estimate_activation_sparsity(
        model, dataset, 1, threshold=0, target_node_types=["Convolution", "Add"]
    )

    assert activation_sparsity_stats == {
        "Conv": [{"port_id": 0, "sparsity_level": 0.0}],
        "Conv_Add": [{"port_id": 0, "sparsity_level": 0.0}],
    }

REF_STATIC_POINTS = {
    "ConvModel": {"Conv": 1},
    "LinearModel": {"MatMul": 1},
}
REF_STATIC_POINTS_ADD = {
    "ConvModel": {"Conv_Add": 1},
    "LinearModel": {"Add": 1},
}
@pytest.mark.parametrize("model_cls_name", ("ConvModel", "LinearModel"))
@pytest.mark.parametrize("list_node_types", (None, [], ["Add"]))
def test_get_static_points(model_cls_name, list_node_types):
    model_to_test = SYNTHETIC_MODELS.get(model_cls_name)().ov_model

    algo = ActivationSparsityStatistic(
        ActivationSparsityStatisticParameters(number_samples=1, target_node_types=list_node_types, threshold=0.0)
    )
    statistic_points = algo.get_statistic_points(model_to_test)

    points_info = {}
    for node_name, points in statistic_points.items():
        if node_name.split("_")[0] in ["Gather", "Concat"]:
            # Node have not constant name if run pytest in parallel mode
            node_name = node_name.split("_")[0]
        points_info[node_name] = len(points)

    if not list_node_types:
        assert points_info == REF_STATIC_POINTS[model_cls_name]
    else:
        assert points_info == REF_STATIC_POINTS_ADD[model_cls_name]


@pytest.mark.parametrize("tensor, ref", (([1, 1], 0.0), ([1, 0], 0.5), ([[0, 0], [1, 0]], 0.75), ([0, 0], 1.0)))
def test_percentage_of_zeros_tensor_processor(tensor, ref):
    result = OVNNCFCollectorTensorProcessor.percentage_of_zeros(NNCFTensor(np.array(tensor)))
    assert result == ref
