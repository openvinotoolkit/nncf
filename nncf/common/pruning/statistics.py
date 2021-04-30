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

from typing import List, Dict, Any

from nncf.api.compression import Statistics
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
        :param weight_pruning_level: TODO
        :param mask_pruning_level: TODO
        :param filter_pruning_level: TODO
        """
        self.name = name
        self.weight_shape = weight_shape
        self.mask_shape = mask_shape
        self.weight_pruning_level = weight_pruning_level
        self.mask_pruning_level = mask_pruning_level
        self.filter_pruning_level = filter_pruning_level

    def as_dict(self) -> Dict[str, Any]:
        """
        Returns a representation of the pruned layer's summary as built-in data types.

        :return: A representation of the sparsified layer's summary as built-in data types.
        """
        summary = {
            'name': self.name,
            'weight_shape': self.weight_shape,
            'mask_shape': self.mask_shape,
            'weight_pruning_level': self.weight_pruning_level,
            'mask_pruning_level': self.mask_pruning_level,
            'filter_pruning_level': self.filter_pruning_level,
        }
        return summary


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

    def as_str(self) -> str:
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

    def as_dict(self) -> Dict[str, Any]:
        stats = {
            'pruning_level': self.pruning_level,
            'pruned_layers_summary': [s.as_dict() for s in self.pruned_layers_summary],
        }
        return stats


class FilterPruningStatistics(Statistics):
    """
    Contains statistics of the filter pruning algorithm.
    """

    def __init__(self,
                 model_statistics: PrunedModelStatistics,
                 full_flops: int,
                 current_flops: int):
        """
        Initializes statistics of the filter pruning algorithm.

        :param model_statistics: Statistics of the pruned model.
        :param full_flops: TODO
        :param current_flops: TODO
        """
        self.model_statistics = model_statistics
        self.full_flops = full_flops
        self.current_flops = current_flops
        self.flops_pruning_level = 1 - self.current_flops / self.full_flops

    def as_str(self) -> str:
        algorithm_string = create_table(
            header=['Statistic\'s name', 'Value'],
            rows=[
                ['FLOPS pruning level', self.flops_pruning_level],
                ['FLOPS current / full', f'{self.current_flops} / {self.full_flops}'],
            ]
        )

        pretty_string = (
            f'{self.model_statistics.as_str()}\n\n'
            f'Statistics of the filter pruning algorithm:\n{algorithm_string}'
        )
        return pretty_string

    def as_dict(self) -> Dict[str, Any]:
        algorithm = 'filter_pruning'
        model_statistics = self.model_statistics.as_dict()
        stats = {
            f'{algorithm}/pruning_level_for_model': model_statistics['pruning_level'],
            f'{algorithm}/pruning_statistic_by_layer': model_statistics['pruned_layers_summary'],
            f'{algorithm}/flops_pruning_level': self.flops_pruning_level,
            f'{algorithm}/full_flops': self.full_flops,
            f'{algorithm}/current_flops': self.current_flops,
        }
        return stats
