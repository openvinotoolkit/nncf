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


import openvino.runtime as ov

from nncf.data import Dataset
from nncf.experimental.openvino_native.activation_sparsity_statistic.algorithm import ActivationSparsityStatistic
from nncf.experimental.openvino_native.activation_sparsity_statistic.algorithm import \
    ActivationSparsityStatisticParameters


def activation_sparsity_statistic_impl(model: ov.Model, dataset: Dataset, subset_size: int) -> ov.Model:
    """
    Implementation of the `activation_sparsity_statistic` method for the OpenVINO backend via the OpenVINO Runtime API.
    """

    parameters = ActivationSparsityStatisticParameters(number_samples=subset_size)
    activation_sparsity_statistic_algorithm = ActivationSparsityStatistic(parameters)

    modified_model = activation_sparsity_statistic_algorithm.apply(model, dataset=dataset)

    return modified_model
