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

from dataclasses import dataclass
from typing import Any, Optional, TypeVar

import nncf
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.common.utils.helpers import create_table
from nncf.data.dataset import Dataset
from nncf.parameters import PruneMode
from nncf.scopes import IgnoredScope

TModel = TypeVar("TModel")


def prune(
    model: TModel,
    mode: PruneMode,
    *,
    ratio: Optional[float] = None,
    ignored_scope: Optional[IgnoredScope] = None,
    examples_inputs: Optional[Any] = None,
) -> TModel:
    """
    Prunes the given model based on the specified mode and ratio.
    Insert binary masks for the parameters and calculate the initial masks by the given ratio.

    :param model: The model to be pruned.
    :param mode: The mode of pruning to be applied.
    :param ratio: The ratio of parameters to prune from the model.
    :param ignored_scope: A scope of parameters to ignore during pruning. Defaults to None.
    :param examples_inputs: Example inputs to guide the pruning process. Defaults to None.
    :return: The pruned model.
    """
    backend = get_backend(model)
    if backend == BackendType.TORCH:
        from nncf.torch.function_hook.pruning.prune_model import prune

        model = prune(model, mode, ratio, ignored_scope, examples_inputs)
    else:
        msg = f"Pruning is not supported for the {backend} backend."
        raise nncf.InternalError(msg)
    return model


def batch_norm_adaptation(
    model: TModel, calibration_dataset: Dataset, *, num_iterations: Optional[int] = None
) -> TModel:
    """
    Adapt the batch normalization layers of the given model using the provided dataset.
    This function runs a specified number of iterations through the model
    to update the running statistics of the batch normalization layers.

    :param model: The model to adapt.
    :param calibration_dataset: The dataset to use for the adaptation.
    :param num_iterations: The number of iterations to use for adaptation.
        If set to None, the adaptation will run for the entire dataset.
    """
    backend = get_backend(model)
    if backend == BackendType.TORCH:
        from nncf.torch.function_hook.pruning.batch_norm_adaptation import batch_norm_adaptation

        return batch_norm_adaptation(model, calibration_dataset=calibration_dataset, num_iterations=num_iterations)

    msg = f"Batch norm adaptation is not supported for the {backend} backend."
    raise nncf.InternalError(msg)


@dataclass
class TensorPruningStatistic:
    """
    Statistics about pruning for a single tensor.

    :param tensor_name: Name of the tensor.
    :param shape: Shape of the tensor.
    :param pruned_ratio: Ratio of pruned elements in the tensor.
    """

    tensor_name: str
    shape: tuple[int, ...]
    pruned_ratio: float


@dataclass
class ModelPruningStatistic:
    """
    Aggregated pruning statistics for a model.

    :param pruning_ratio: Overall pruning ratio for pruned parameters in the model.
    :param global_pruning_ratio: Overall pruning ratio for all parameters in the model.
    :param pruned_tensors: List of pruning statistics for each tensor.
    """

    pruning_ratio: float
    global_pruning_ratio: float
    pruned_tensors: list[TensorPruningStatistic]

    def __str__(self) -> str:
        total = [
            [None, None, None, None],
            ["Prunable parameters", None, self.pruning_ratio],
            ["All parameters", None, self.global_pruning_ratio],
        ]

        sorted_stat_per_tensor = sorted(self.pruned_tensors, key=lambda s: s.tensor_name)
        rows_per_tensor = [[s.tensor_name, s.shape, s.pruned_ratio] for s in sorted_stat_per_tensor]
        text = create_table(header=["Parameter's name", "Shape", "Pruning ratio"], rows=rows_per_tensor + total)
        return text


def pruning_statistic(model: TModel) -> ModelPruningStatistic:
    """
    Collects and returns pruning statistics for the given model.

    :param model: The pruned model.
    :return: A pruning statistic.
    """
    backend = get_backend(model)
    if backend == BackendType.TORCH:
        from nncf.torch.function_hook.pruning.statistics import pruning_statistic

        return pruning_statistic(model)
    msg = f"Pruning statistics collection is not supported for the {backend} backend."
    raise nncf.InternalError(msg)
