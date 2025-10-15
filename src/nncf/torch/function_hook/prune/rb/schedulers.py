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

from abc import ABC
from abc import abstractmethod
from typing import Optional

import nncf
import nncf.errors
from nncf.torch.function_hook.prune.rb.losses import RBLoss


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

    def step(self, epoch: Optional[int] = None) -> None:
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
    :param steps: A dictionary mapping epochs to sparsity ratios. The keys should be in ascending order,
                  and the values should be in the range [0, 1).
    """

    def __init__(self, rb_loss: RBLoss, steps: dict[int, int]) -> None:
        super().__init__(rb_loss)
        self.steps = steps
        if list(self.steps) != sorted(self.steps) or any(r >= 1 or r < 0 for r in self.steps.values()):
            msg = (
                "Invalid schedule_dict provided to SparsityScheduler."
                "Keys should be in ascending order and values should be in range [0, 1)."
            )
            raise nncf.InternalError(msg)

        self.rb_loss.current_ratio = self.steps[sorted(self.steps.keys())[0]]

    def get_pruning_ratio(self, epoch: int) -> float:
        for steps in sorted(self.steps.keys(), reverse=True):
            if epoch >= steps:
                return self.steps[steps]
        return self.current_ratio


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

        if initial_ratio < 0 or initial_ratio >= 1:
            msg = "initial_ratio should be in range [0, 1)."
            raise nncf.InternalError(msg)
        if target_ratio <= 0 or target_ratio >= 1:
            msg = "target_ratio should be in range (0, 1)."
            raise nncf.InternalError(msg)
        if target_epoch < 1:
            msg = "target_epoch should be positive integer."
            raise nncf.InternalError(msg)

    def get_pruning_ratio(self, epoch: int) -> float:
        if epoch == 0:
            return self.initial_ratio
        if epoch >= self.target_epoch:
            return self.target_ratio

        d_init = 1 - self.initial_ratio
        d_target = 1 - self.target_ratio
        d = d_init * float((d_target / d_init) ** (epoch / self.target_epoch))
        ratio = 1 - d
        return min(max(self.initial_ratio, ratio), self.target_epoch)
