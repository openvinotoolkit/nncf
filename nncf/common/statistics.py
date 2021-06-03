"""
 Copyright (c) 2021 Intel Corporation
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

from typing import Optional

from nncf.api.statistics import Statistics
from nncf.common.sparsity.statistics import MagnitudeSparsityStatistics
from nncf.common.sparsity.statistics import RBSparsityStatistics
from nncf.common.sparsity.statistics import ConstSparsityStatistics
from nncf.common.quantization.statistics import QuantizationStatistics
from nncf.common.pruning.statistics import FilterPruningStatistics


class NNCFStatistics(Statistics):
    """
    Groups statistics for all available NNCF compression algorithms.
    Statistics are present only if the algorithm has been started.
    """

    def __init__(self):
        """
        Initializes nncf statistics.
        """
        self._storage = {}

    @property
    def magnitude_sparsity(self) -> Optional[MagnitudeSparsityStatistics]:
        """
        Returns statistics of the magnitude sparsity algorithm. If statistics
        have not been collected, `None` will be returned.

        :return: Instance of the `MagnitudeSparsityStatistics` class.
        """
        return self._storage.get('magnitude_sparsity')

    @property
    def rb_sparsity(self) -> Optional[RBSparsityStatistics]:
        """
        Returns statistics of the RB-sparsity algorithm. If statistics
        have not been collected, `None` will be returned.

        :return: Instance of the `RBSparsityStatistics` class.
        """
        return self._storage.get('rb_sparsity')

    @property
    def const_sparsity(self) -> Optional[ConstSparsityStatistics]:
        """
        Returns statistics of the const sparsity algorithm. If statistics
        have not been collected, `None` will be returned.

        :return: Instance of the `ConstSparsityStatistics` class.
        """
        return self._storage.get('const_sparsity')

    @property
    def quantization(self) -> Optional[QuantizationStatistics]:
        """
        Returns statistics of the quantization algorithm. If statistics
        have not been collected, `None` will be returned.

        :return: Instance of the `QuantizationStatistics` class.
        """
        return self._storage.get('quantization')

    @property
    def filter_pruning(self) -> Optional[FilterPruningStatistics]:
        """
        Returns statistics of the filter pruning algorithm. If statistics
        have not been collected, `None` will be returned.

        :return: Instance of the `FilterPruningStatistics` class.
        """
        return self._storage.get('filter_pruning')

    @property
    def binarization(self) -> None:
        """
        Returns statistics of the binarization algorithm. If statistics
        have not been collected, `None` will be returned.

        :return: `None`.
        """
        return self._storage.get('binarization')

    def register(self, algorithm_name: str, stats: Statistics):
        """
        Registers statistics for the algorithm.

        :param algorithm_name: Name of the algorithm. Should be one of the following
            - magnitude_sparsity
            - rb_sparsity
            - const_sparsity
            - quantization
            - filter_pruning
            - binarization
        :param stats: Statistics of the algorithm.
        """

        available_algorithms = [
            'magnitude_sparsity', 'rb_sparsity', 'const_sparsity',
            'quantization', 'filter_pruning', 'binarization'
        ]
        if algorithm_name not in available_algorithms:
            raise ValueError('Can not register statistics for the algorithm. '
                             f'Unknown name of the algorithm: {algorithm_name}.')

        self._storage[algorithm_name] = stats

    def to_str(self) -> str:
        """
        Calls `to_str()` method for all registered statistics of the algorithm and returns
        a sum-up string.

        :return: A representation of the NNCF statistics as a human-readable string.
        """
        pretty_string = '\n\n'.join([stats.to_str() for stats in self._storage.values()])
        return pretty_string

    def __iter__(self):
        return iter(self._storage.items())
