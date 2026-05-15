# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC
from abc import abstractmethod

from nncf.torch.function_hook.pruning.rb.losses import RBLoss
from nncf.torch.function_hook.pruning.scheduler_fns import exponential_ratio_scheduler
from nncf.torch.function_hook.pruning.scheduler_fns import multi_step_ratio_scheduler


class _BaseRBScheduler(ABC):
    """
    Base class for RB pruning schedulers.

    :param rb_loss: A instance of RBLoss.
    :param epoch: The current epoch number.
    :param current_ratio: The current pruning ratio applied to the model.
    """

    def __init__(self, rb_loss: RBLoss) -> None:
        self.rb_loss = rb_loss
        self.epoch = 0

    def step(self, epoch: int | None = None) -> None:
        """
        Updates the pruning schedule for the model based on the current epoch.

        :param epoch: The current epoch number. If None, the method uses the internal epoch counter.
        """
        if epoch is None:
            epoch = self.epoch
        else:
            self.epoch = epoch

        self.rb_loss.target_ratio = self.get_pruning_ratio(epoch)
        self.epoch += 1

    @abstractmethod
    def get_pruning_ratio(self, epoch: int) -> float:
        """
        Calculate the sparsity ratio for a given epoch.

        :param epoch: The current epoch number.d
        :return: The pruning ratio for the specified epoch.
        """

    @property
    def current_ratio(self) -> float:
        """
        Returns the current pruning ratio.

        :return: The current pruning ratio.
        """
        return self.rb_loss.current_ratio


class MultiStepRBPruningScheduler(_BaseRBScheduler):
    """
    A scheduler for controlling the pruning ration over multiple steps.

    Note: If the first key in the steps dictionary is not 0, the initial sparsity ratio will be set to
          the first value in the steps.

    :param rb_loss: A instance of RBLoss.
    :param steps: A dictionary mapping epochs to pruning ratios. The keys should be in ascending order,
                  and the values should be in the range [0, 1).
    """

    def __init__(self, rb_loss: RBLoss, steps: dict[int, int]) -> None:
        super().__init__(rb_loss)
        self.steps = steps

    def get_pruning_ratio(self, epoch: int) -> float:
        return multi_step_ratio_scheduler(epoch, steps=self.steps)


class ExponentialRBPruningScheduler(_BaseRBScheduler):
    """
    A scheduler for controlling the pruning ration over exponential function.

    :param rb_loss: A instance of RBLoss.
    :param initial_ratio: The initial pruning ratio (should be in the range [0, 1)).
    :param target_ratio: The target pruning ratio (should be in the range (0, 1)).
    :param target_epoch: The epoch at which the target pruning ratio should be reached (should be a positive integer).
    """

    def __init__(self, rb_loss: RBLoss, initial_ratio: float, target_ratio: float, target_epoch: int) -> None:
        super().__init__(rb_loss)
        self.initial_ratio = initial_ratio
        self.target_ratio = target_ratio
        self.target_epoch = target_epoch
        self.rb_loss.current_ratio = initial_ratio

    def get_pruning_ratio(self, epoch: int) -> float:
        return exponential_ratio_scheduler(
            epoch,
            initial_ratio=self.initial_ratio,
            target_ratio=self.target_ratio,
            target_epoch=self.target_epoch,
        )
