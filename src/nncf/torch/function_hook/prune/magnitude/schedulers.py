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

import weakref
from abc import ABC
from abc import abstractmethod
from typing import Optional

from torch import nn

import nncf
import nncf.errors
from nncf.parameters import PruneMode
from nncf.torch.function_hook.prune.magnitude.algo import update_pruning_ratio
from nncf.torch.function_hook.prune.scheduler_fns import exponential_ratio_scheduler
from nncf.torch.function_hook.prune.scheduler_fns import multi_step_ratio_scheduler


class _BaseMagnitudePruningScheduler(ABC):
    """
    Base class for pruning schedulers.

    :param ref_model: A weak reference to the model being pruned.
    :param mode: The mode of pruning to be applied.
    :param epoch: The current epoch number.
    :param current_ratio: The current pruning ratio applied to the model.
    """

    def __init__(self, model: nn.Module, mode: PruneMode) -> None:
        self.ref_model = weakref.ref(model)
        self.mode = mode
        self.epoch = 0
        self.current_ratio = 0.0

    def step(self, epoch: Optional[int] = None) -> None:
        """
        Updates the pruning schedule for the model based on the current epoch.

        :param epoch: The current epoch number. If None, the method uses the internal epoch counter.
        """
        if epoch is None:
            epoch = self.epoch
        else:
            self.epoch = epoch

        self.current_ratio = self.get_pruning_ratio(epoch)

        model = self.ref_model()
        if model is None:
            msg = "The referenced model is no longer available"
            raise nncf.InternalError(msg)

        update_pruning_ratio(model, mode=self.mode, ratio=self.current_ratio)
        self.epoch += 1

    @abstractmethod
    def get_pruning_ratio(self, epoch: int) -> float:
        """
        Calculate the sparsity ratio for a given epoch.

        :param epoch: The current epoch number.
        :return: The pruning ratio for the specified epoch.
        """


class MultiStepMagnitudePruningScheduler(_BaseMagnitudePruningScheduler):
    """
    A scheduler for controlling the pruning ration over multiple steps.

    Note: If the first key in the steps dictionary is not 0, the initial sparsity ratio will be set to
          the first value in the steps.

    :param model: The neural network model to be pruned.
    :param mode: The pruning mode to be used.
    :param steps: A dictionary mapping epochs to pruning ratios. The keys should be in ascending order,
                  and the values should be in the range [0, 1).
    """

    def __init__(self, model: nn.Module, *, mode: PruneMode, steps: dict[int, int]) -> None:
        super().__init__(model, mode)
        self.steps = steps

    def get_pruning_ratio(self, epoch: int) -> float:
        return multi_step_ratio_scheduler(epoch, steps=self.steps)


class ExponentialMagnitudePruningScheduler(_BaseMagnitudePruningScheduler):
    """
    A scheduler for controlling the pruning ration over exponential function.

    :param model: The neural network model to be pruned.
    :param mode: The pruning mode to be used (e.g., global, layer-wise).
    :param initial_ratio: The initial pruning ratio (should be in the range [0, 1)).
    :param target_ratio: The target pruning ratio (should be in the range (0, 1)).
    :param target_epoch: The epoch at which the target pruning ratio should be reached (should be a positive integer).
    """

    def __init__(
        self, model: nn.Module, *, mode: PruneMode, initial_ratio: float, target_ratio: float, target_epoch: int
    ) -> None:
        super().__init__(model, mode)
        self.initial_ratio = initial_ratio
        self.target_ratio = target_ratio
        self.target_epoch = target_epoch
        self.current_ratio = initial_ratio

    def get_pruning_ratio(self, epoch: int) -> float:
        return exponential_ratio_scheduler(
            epoch,
            initial_ratio=self.initial_ratio,
            target_ratio=self.target_ratio,
            target_epoch=self.target_epoch,
        )
