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


@api()
class SparsifiedLayerSummary:
    """
    Contains information about the sparsified layer.

    :param name: Layer's name.
    :param weight_shape: Weight's shape.
    :param sparsity_level: Sparsity level of the sparsified layer.
    :param weight_percentage: Proportion of the layer's weights in the whole model.
    """

    def __init__(
        self,
        name: str,
        weight_shape: List[int],
        sparsity_level: float,
        weight_percentage: float,
    ):
        self.name = name
        self.weight_shape = weight_shape
        self.sparsity_level = sparsity_level
        self.weight_percentage = weight_percentage


@api()
class SparsifiedModelStatistics(Statistics):
    """
    Contains statistics of the sparsified model.

    :param sparsity_level: Sparsity level of the whole model.
    :param sparsity_level_for_layers: Sparsity level of all sparsified layers
      (i.e. layers for which the algorithm was applied).
    :param sparsified_layers_summary: Detailed summary for the sparsified layers.
    """

    def __init__(
        self,
        sparsity_level: float,
        sparsity_level_for_layers: float,
        sparsified_layers_summary: List[SparsifiedLayerSummary],
    ):
        self.sparsity_level = sparsity_level
        self.sparsity_level_for_layers = sparsity_level_for_layers
        self.sparsified_layers_summary = sparsified_layers_summary

    def to_str(self) -> str:
        model_string = create_table(
            header=["Statistic's name", "Value"],
            rows=[
                ["Sparsity level of the whole model", self.sparsity_level],
                [
                    "Sparsity level of all sparsified layers",
                    self.sparsity_level_for_layers,
                ],
            ],
        )

        layers_string = create_table(
            header=[
                "Layer's name",
                "Weight's shape",
                "Sparsity level",
                "Weight's percentage",
            ],
            rows=[
                [s.name, s.weight_shape, s.sparsity_level, s.weight_percentage] for s in self.sparsified_layers_summary
            ],
        )

        pretty_string = (
            f"Statistics of the sparsified model:\n{model_string}\n\n"
            f"Statistics by sparsified layers:\n{layers_string}"
        )
        return pretty_string


class LayerThreshold:
    def __init__(self, name: str, threshold: float):
        self.name = name
        self.threshold = threshold


@api()
class MagnitudeSparsityStatistics(Statistics):
    """
    Contains statistics of the magnitude sparsity algorithm.

    :param model_statistics: Statistics of the sparsified model.
    :param thresholds: List of the sparsity thresholds.
    :param target_sparsity_level: A target level of the sparsity for the algorithm for the current epoch.
    """

    def __init__(
        self,
        model_statistics: SparsifiedModelStatistics,
        thresholds: List[LayerThreshold],
        target_sparsity_level: float,
    ):
        self.model_statistics = model_statistics
        self.thresholds = thresholds
        self.target_sparsity_level = target_sparsity_level

    def to_str(self) -> str:
        thresholds_string = create_table(
            ["Layer's name", "Sparsity threshold"],
            [[s.name, s.threshold] for s in self.thresholds],
        )

        algorithm_string = create_table(
            header=["Statistic's name", "Value"],
            rows=[
                [
                    "A target level of the sparsity for the algorithm for the current epoch",
                    self.target_sparsity_level,
                ],
            ],
        )

        pretty_string = (
            f"{self.model_statistics.to_str()}\n\n"
            f"Statistics of the magnitude sparsity algorithm:\n{algorithm_string}\n{thresholds_string}"
        )
        return pretty_string


@api()
class ConstSparsityStatistics(Statistics):
    """
    Contains statistics of the const sparsity algorithm.

    :param model_statistics: Statistics of the sparsified model.
    """

    def __init__(self, model_statistics: SparsifiedModelStatistics):
        self.model_statistics = model_statistics

    def to_str(self) -> str:
        pretty_string = self.model_statistics.to_str()
        return pretty_string


@api()
class RBSparsityStatistics(Statistics):
    """
    Contains statistics of the RB-sparsity algorithm.

    :param model_statistics: Statistics of the sparsified model.
    :param target_sparsity_level: A target level of the sparsity for the algorithm for the current epoch.
    :param mean_sparse_prob: The probability that one weight will be zeroed.
    """

    def __init__(
        self,
        model_statistics: SparsifiedModelStatistics,
        target_sparsity_level: float,
        mean_sparse_prob: float,
    ):
        self.model_statistics = model_statistics
        self.target_sparsity_level = target_sparsity_level
        self.mean_sparse_prob = mean_sparse_prob

    def to_str(self) -> str:
        algorithm_string = create_table(
            header=["Statistic's name", "Value"],
            rows=[
                [
                    "A target level of the sparsity for the algorithm for the current epoch",
                    self.target_sparsity_level,
                ],
                [
                    "The probability that one weight will be zeroed",
                    self.mean_sparse_prob,
                ],
            ],
        )

        pretty_string = (
            f"{self.model_statistics.to_str()}\n\n Statistics of the RB-sparsity algorithm:\n{algorithm_string}"
        )
        return pretty_string


@api()
class MovementSparsityStatistics(Statistics):
    """
    Contains statistics of the movement-sparsity algorithm.

    :param model_statistics: Statistics of the sparsified model.
    :param importance_threshold: Importance threshold for sparsity binary mask.
    :param importance_regularization_factor: Penalty factor of importance score.
    """

    def __init__(
        self,
        model_statistics: SparsifiedModelStatistics,
        importance_threshold: float,
        importance_regularization_factor: float,
    ):
        self.model_statistics = model_statistics
        self.importance_threshold = importance_threshold
        self.importance_regularization_factor = importance_regularization_factor

    def to_str(self) -> str:
        algorithm_string = create_table(
            header=["Statistic's name", "Value"],
            rows=[
                ["Mask Importance Threshold", self.importance_threshold],
                [
                    "Importance Regularization Factor",
                    self.importance_regularization_factor,
                ],
            ],
        )

        pretty_string = (
            f"{self.model_statistics.to_str()}\n\n"
            f"Statistics of the movement-sparsity algorithm:\n{algorithm_string}"
        )
        return pretty_string
