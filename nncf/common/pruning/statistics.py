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
                 filter_pruning_level: float):
        """
        Initializes a summary about the pruned layer.

        :param name: Layer's name.
        :param weight_shape: Weight's shape.
        :param mask_shape: Mask's shape.
        :param filter_pruning_level: Filter's pruning level.
        """
        self.name = name
        self.weight_shape = weight_shape
        self.mask_shape = mask_shape
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

        header = ['Layer\'s name', 'Weight\'s shape', 'Mask\'s shape', 'Filter pruning level']
        rows = []
        for s in self.pruned_layers_summary:
            rows.append([s.name, s.weight_shape, s.mask_shape, s.filter_pruning_level])

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


class PrunedModelTheoreticalBorderline(Statistics):
    """
    Contains theoretical borderline statistics of the filter pruning algorithm.
    """

    def __init__(self,
                 num_pruned_layers: int,
                 num_prunable_layers: int,
                 max_prunable_flops: float,
                 max_prunable_params: float,
                 full_flops: int,
                 full_params_num: int):
        """
        Initializes statistics of the filter pruning theoretical borderline.

        :param num_pruned_layers: number of layers which was actually
            pruned.
        :param num_prunable_layers: number of layers which have
            prunable type.
        :param max_prunable_flops: number of flops for pruned
            model with pruning rate = 1.
        :param max_prunable_params: number of weights for pruned
            model with pruning rate = 1.
        :param full_flops: Full FLOPS.
        :param full_params_num: Full number of weights.
        """
        self._giga = 1e9
        self._mega = 1e6
        self.pruned_layers_num = num_pruned_layers
        self.prunable_layers_num = num_prunable_layers
        self.minimum_possible_flops = max_prunable_flops
        self.minimum_possible_params = max_prunable_params
        self.full_flops = full_flops
        self.full_params_num = full_params_num

    def to_str(self) -> str:
        algorithm_string = create_table(
            header=['Statistic\'s name', 'Value'],
            rows=[
                ['Pruned layers count / prunable layers count', f'{self.pruned_layers_num} /'
                                                                f' {self.prunable_layers_num}'],
                ['GFLOPS minimum possible after pruning / full', f'{self.minimum_possible_flops / self._giga:.3f} /'
                                                                 f' {self.full_flops / self._giga:.3f}'],
                ['MParams minimum possible after pruning / full', f'{self.minimum_possible_params / self._mega:.3f} /'
                                                                  f' {self.full_params_num / self._mega:.3f}'],
            ]
        )

        pretty_string = (
            f'Theoretical borderline of the filter pruning algorithm\nfor current model:\n{algorithm_string}'
        )
        return pretty_string
