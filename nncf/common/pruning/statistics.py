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

from typing import List

from nncf.api.statistics import Statistics
from nncf.common.utils.helpers import create_table


class PrunedLayerSummary:
    """
    Contains information about the pruned layer.
    """

    def __init__(self,
                 name: str,
                 weight_shape: List[int],
                 mask_shape: List[int],
                 weight_pruning_level: float,
                 mask_pruning_level: float,
                 filter_pruning_level: float):
        """
        Initializes a summary about the pruned layer.

        :param name: Layer's name.
        :param weight_shape: Weight's shape.
        :param mask_shape: Mask's shape.
        :param weight_pruning_level: Weight's pruning level.
        :param mask_pruning_level: Mask's pruning level.
        :param filter_pruning_level: Filter's pruning level.
        """
        self.name = name
        self.weight_shape = weight_shape
        self.mask_shape = mask_shape
        self.weight_pruning_level = weight_pruning_level
        self.mask_pruning_level = mask_pruning_level
        self.filter_pruning_level = filter_pruning_level


class PrunedModelStatistics(Statistics):
    """
    Contains statistics of the pruned model.
    """

    def __init__(self,
                 pruning_level: float,
                 pruned_layers_summary: List[PrunedLayerSummary]):
        """
        Initializes statistics of the pruned model.

        :param pruning_level: Pruning level of the whole model.
        :param pruned_layers_summary: Detailed summary for the
            pruned layers.
        """
        self.pruning_level = pruning_level
        self.pruned_layers_summary = pruned_layers_summary

    def to_str(self) -> str:
        model_string = create_table(
            header=['Statistic\'s name', 'Value'],
            rows=[
                ['Pruning level of the whole model', self.pruning_level],
            ]
        )

        header = ['Layer\'s name', 'Weight\'s shape', 'Mask\'s shape',
                  'Zeros in mask, %', 'Pruning level', 'Filter pruning level']
        rows = []
        for s in self.pruned_layers_summary:
            rows.append([
                s.name, s.weight_shape, s.mask_shape, 100 * s.mask_pruning_level,
                s.weight_pruning_level, s.filter_pruning_level
            ])

        layers_string = create_table(header, rows)

        pretty_string = (
            f'Statistics of the pruned model:\n{model_string}\n\n'
            f'Statistics by pruned layers:\n{layers_string}'
        )

        return pretty_string


class FilterPruningStatistics(Statistics):
    """
    Contains statistics of the filter pruning algorithm.
    """

    def __init__(self,
                 model_statistics: PrunedModelStatistics,
                 full_flops: int,
                 current_flops: int,
                 full_params_num: int,
                 current_params_num: int,
                 target_pruning_level: float):
        """
        Initializes statistics of the filter pruning algorithm.

        :param model_statistics: Statistics of the pruned model.
        :param full_flops: Full FLOPS.
        :param current_flops: Current FLOPS.
        :param full_params_num: Full number of weights.
        :param current_params_num: Current number of weights.
        :param target_pruning_level: A target level of the pruning
            for the algorithm for the current epoch.
        """
        self._giga = 1e9
        self._mega = 1e6
        self.model_statistics = model_statistics
        self.full_flops = full_flops
        self.current_flops = current_flops
        self.flops_pruning_level = 1 - self.current_flops / self.full_flops
        self.full_params_num = full_params_num
        self.current_params_num = current_params_num
        self.target_pruning_level = target_pruning_level

    def to_str(self) -> str:
        algorithm_string = create_table(
            header=['Statistic\'s name', 'Value'],
            rows=[
                ['FLOPS pruning level', self.flops_pruning_level],
                ['GFLOPS current / full', f'{self.current_flops / self._giga:.3f} /'
                                          f' {self.full_flops / self._giga:.3f}'],
                ['MParams current / full', f'{self.current_params_num / self._mega:.3f} /'
                                           f' {self.full_params_num / self._mega:.3f}'],
                ['A target level of the pruning for the algorithm for the current epoch', self.target_pruning_level],
            ]
        )

        pretty_string = (
            f'{self.model_statistics.to_str()}\n\n'
            f'Statistics of the filter pruning algorithm:\n{algorithm_string}'
        )
        return pretty_string
