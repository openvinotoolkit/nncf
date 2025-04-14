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

from typing import List

from nncf.api.statistics import Statistics
from nncf.common.utils.api_marker import api
from nncf.common.utils.helpers import create_table


class PrunedLayerSummary:
    """
    Contains information about the pruned layer.
    """

    def __init__(self, name: str, weight_shape: List[int], mask_shape: List[int], filter_pruning_level: float):
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

    def __init__(
        self,
        full_flops: int,
        current_flops: int,
        full_params_num: int,
        current_params_num: int,
        full_filters_num: int,
        current_filters_num: int,
        pruned_layers_summary: List[PrunedLayerSummary],
    ):
        """
        Initializes statistics of the pruned model.

        :param full_flops: The total amount of FLOPs in the model.
        :param current_flops: Current amount of FLOPs in the model.
        :param full_params_num: The total amount of weights in the model.
        :param current_params_num: Current amount of weights in the model.
        :param full_filters_num: The total amount of filters in the model.
        :param current_filters_num: Current amount of filters in the model.
        :param pruned_layers_summary: Detailed summary for the
            pruned layers.
        """
        self._giga = 1e9
        self._mega = 1e6
        self.full_flops = full_flops
        self.current_flops = current_flops
        self.flops_pruning_level = 1 - self.current_flops / self.full_flops
        self.full_params_num = full_params_num
        self.current_params_num = current_params_num
        self.params_pruning_level = 1 - self.current_params_num / self.full_params_num
        self.full_filters_num = full_filters_num
        self.current_filters_num = current_filters_num
        self.filter_pruning_level = 1 - self.current_filters_num / self.full_filters_num
        self.pruned_layers_summary = pruned_layers_summary

    def to_str(self) -> str:
        model_string = create_table(
            header=["#", "Full", "Current", "Pruning level"],
            rows=[
                [
                    "GFLOPs",
                    f"{self.full_flops / self._giga:.3f}",
                    f"{self.current_flops / self._giga:.3f}",
                    self.flops_pruning_level,
                ],
                [
                    "MParams",
                    f"{self.full_params_num / self._mega:.3f}",
                    f"{self.current_params_num / self._mega:.3f}",
                    self.params_pruning_level,
                ],
                ["Filters", self.full_filters_num, self.current_filters_num, self.filter_pruning_level],
            ],
        )

        header = ["Layer's name", "Weight's shape", "Mask's shape", "Filter pruning level"]
        rows = []
        for s in self.pruned_layers_summary:
            rows.append([s.name, s.weight_shape, s.mask_shape, s.filter_pruning_level])

        layers_string = create_table(header, rows)

        pruning_level_desc = "Prompt: statistic pruning level = 1 - statistic current / statistic full."
        pretty_string = (
            f"Statistics by pruned layers:\n{layers_string}\n"
            f"Statistics of the pruned model:\n{model_string}\n" + pruning_level_desc
        )

        return pretty_string


@api()
class FilterPruningStatistics(Statistics):
    """
    Contains statistics of the filter pruning algorithm.

    :param model_statistics: Statistics of the pruned model.
    :param current_pruning_level: A current level of the pruning for the algorithm for the current epoch.
    :param target_pruning_level: A target level of the pruning for the algorithm.
    :param prune_flops: Is pruning algo sets flops pruning level or not (filter pruning level).
    """

    def __init__(
        self,
        model_statistics: PrunedModelStatistics,
        current_pruning_level: float,
        target_pruning_level: float,
        prune_flops: bool,
    ):
        self.model_statistics = model_statistics
        self.current_pruning_level = current_pruning_level
        self.target_pruning_level = target_pruning_level
        self.prune_flops = prune_flops

    def to_str(self) -> str:
        pruning_mode = "FLOPs" if self.prune_flops else "filter"
        algorithm_string = create_table(
            header=["Statistic's name", "Value"],
            rows=[
                [f"{pruning_mode.capitalize()} pruning level in current epoch", self.current_pruning_level],
                [f"Target {pruning_mode} pruning level", self.target_pruning_level],
            ],
        )

        pretty_string = (
            f"{self.model_statistics.to_str()}\nStatistics of the filter pruning algorithm:\n{algorithm_string}"
        )
        return pretty_string


class PrunedModelTheoreticalBorderline(Statistics):
    """
    Contains theoretical borderline statistics of the filter pruning algorithm.
    """

    def __init__(
        self,
        num_pruned_layers: int,
        num_prunable_layers: int,
        min_possible_flops: float,
        min_possible_params: float,
        total_flops: int,
        total_params: int,
    ):
        """
        Initializes statistics of the filter pruning theoretical borderline.

        :param num_pruned_layers: Number of layers which was actually pruned.
        :param num_prunable_layers: Number of layers which have prunable type.
        :param min_possible_flops: Number of flops for pruned model with pruning level = 1.
        :param min_possible_params: Number of weights for pruned model with pruning level = 1.
        :param total_flops: The total amount of FLOPS in the model.
        :param total_params: The total amount of weights in the model.
        """
        self._giga = 1e9
        self._mega = 1e6
        self.pruned_layers_num = num_pruned_layers
        self.prunable_layers_num = num_prunable_layers
        self.min_possible_flops = min_possible_flops
        self.min_possible_params = min_possible_params
        self.total_flops = total_flops
        self.total_params = total_params

    def to_str(self) -> str:
        algorithm_string = create_table(
            header=["Statistic's name", "Value"],
            rows=[
                [
                    "Pruned layers count / prunable layers count",
                    f"{self.pruned_layers_num} / {self.prunable_layers_num}",
                ],
                [
                    "GFLOPs minimum possible after pruning / total",
                    f"{self.min_possible_flops / self._giga:.3f} / {self.total_flops / self._giga:.3f}",
                ],
                [
                    "MParams minimum possible after pruning / total",
                    f"{self.min_possible_params / self._mega:.3f} / {self.total_params / self._mega:.3f}",
                ],
            ],
        )

        pretty_string = (
            f"Theoretical borderline of the filter pruning algorithm\nfor current model:\n{algorithm_string}"
        )
        return pretty_string
