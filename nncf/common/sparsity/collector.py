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

from abc import abstractmethod
from typing import List

import numpy as np

from nncf.common.collector import StatisticsCollector
from nncf.common.sparsity.statistics import SparsifiedLayerSummary
from nncf.common.sparsity.statistics import SparsifiedModelStatistics


class WeightDescription:
    """
    Contains information about the weight of the model.
    """

    def __init__(self, name: str, shape: List[int], num_nonzero: int, is_sparse: bool):
        """
        Initializes the description of the weight.

        :param name: Identifier of the weight.
        :param shape: Shape of the weight.
        :param num_nonzero: Number of nonzero parameters of the weight.
        :param is_sparse: It is a bool value that specifies if the
            weight is used in the sparsity algorithm or not.
        """
        self._name = name
        self._shape = shape
        self._num_nonzero = num_nonzero
        self._is_sparse = is_sparse

        self._num_params = np.prod(shape).item()
        self._num_zero = self._num_params - self._num_nonzero
        self._sparsity_level = self._num_zero / self._num_params

    @property
    def name(self) -> str:
        return self._name

    @property
    def shape(self) -> List[int]:
        return self._shape

    @property
    def num_params(self) -> int:
        return self._num_params

    @property
    def num_nonzero(self) -> int:
        return self._num_nonzero

    @property
    def num_zero(self) -> int:
        return self._num_zero

    @property
    def sparsity_level(self) -> float:
        return self._sparsity_level

    @property
    def is_sparse(self) -> bool:
        return self._is_sparse


def _calculate_sparsity_level_for_model(
    weight_descriptions: List[WeightDescription],
) -> float:
    """
    Calculates the sparsity level for the whole model.

    :param weight_descriptions: Descriptions for weights of the model.
    :return: Sparsity level for the whole model.
    """
    total_params = sum(w.num_params for w in weight_descriptions)
    total_num_zero = sum(w.num_zero for w in weight_descriptions)
    sparsity_level = total_num_zero / total_params

    return sparsity_level


class BaseSparseModelStatisticsCollector(StatisticsCollector):
    """
    Base class for the sparse model statistics collector.
    """

    @abstractmethod
    def _collect_weights_descriptions(self) -> List[WeightDescription]:
        """
        Collects descriptions of the weights of the model.

        :return: Descriptions of the weights of the model.
        """

    def collect(self) -> SparsifiedModelStatistics:
        """
        Collects statistics for the sparse model.

        :return: An instance of the `SparsifiedModelStatistics` class.
        """
        weights_descriptions = self._collect_weights_descriptions()
        sparsity_level_for_model = _calculate_sparsity_level_for_model(weights_descriptions)

        total_params = sum(w.num_params for w in weights_descriptions if w.is_sparse)
        total_num_zero = sum(w.num_zero for w in weights_descriptions if w.is_sparse)
        sparsity_level_for_sparse_layers = total_num_zero / total_params

        sparse_layers_summary = []
        for w in weights_descriptions:
            if not w.is_sparse:
                continue

            weight_percentage = 100 * (w.num_params / total_params)
            sparse_layers_summary.append(SparsifiedLayerSummary(w.name, w.shape, w.sparsity_level, weight_percentage))

        sparse_model_stats = SparsifiedModelStatistics(
            sparsity_level_for_model,
            sparsity_level_for_sparse_layers,
            sparse_layers_summary,
        )

        return sparse_model_stats
