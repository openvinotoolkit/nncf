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
from nncf.torch.function_hook.prune.prune_model import prune_update_ratio


class _BaseMagnitudePruningScheduler(ABC):
    def __init__(self, model: nn.Module, mode: PruneMode) -> None:
        self.ref_model = weakref.ref(model)
        self.mode = mode
        self.epoch = 0
        self.current_ratio = 0.0

    def step(self, epoch: int | None = None) -> None:
        if epoch is None:
            epoch = self.epoch
        else:
            self.epoch = epoch

        ratio = self.get_sparsity_ratio(epoch)
        if ratio is None:
            return

        self.current_ratio = ratio
        prune_update_ratio(self.ref_model(), mode=self.mode, ratio=self.current_ratio)

        self.epoch += 1

    @abstractmethod
    def get_sparsity_ratio(self, epoch: int) -> Optional[float]:
        pass


class MultiStepPruneScheduler(_BaseMagnitudePruningScheduler):
    """
    A scheduler for controlling the sparsity of a neural network model over multiple steps.

    :param model: The neural network model to be pruned.
    :param mode: The pruning mode to be used.
    :param steps: A dictionary mapping epochs to sparsity ratios. The keys should be in ascending order,
                  and the values should be in the range [0, 1).

    :raises nncf.InternalError: If the provided steps are not in ascending order or if any value is
                                outside the range [0, 1).
    """

    def __init__(self, model: nn.Module, *, mode: PruneMode, steps: dict[int, int]) -> None:
        super().__init__(model, mode)
        self.steps = steps
        if list(self.steps.keys()) != sorted(self.steps.keys()) or any(r >= 1 or r < 0 for r in self.steps.values()):
            msg = (
                "Invalid schedule_dict provided to SparsityScheduler."
                "Keys should be in ascending order and values should be in range [0, 1)."
            )
            raise nncf.InternalError(msg)

    def get_sparsity_ratio(self, epoch: int) -> float:
        for steps in sorted(self.steps.keys(), reverse=True):
            if epoch >= steps:
                return self.steps[steps]
        return None


class ExponentialPruneScheduler(_BaseMagnitudePruningScheduler):
    def __init__(
        self, model: nn.Module, *, mode: PruneMode, initial_ratio: float, target_ratio: float, target_epoch: int
    ) -> None:
        super().__init__(model, mode)
        self.initial_ratio = initial_ratio
        self.target_ratio = target_ratio
        self.target_epoch = target_epoch

        if initial_ratio < 0 or initial_ratio >= 1:
            msg = "initial_ratio should be in range [0, 1)."
            raise nncf.InternalError(msg)
        if target_ratio <= 0 or target_ratio >= 1:
            msg = "target_ratio should be in range (0, 1)."
            raise nncf.InternalError(msg)
        if target_epoch < 1:
            msg = "target_epoch should be positive integer."
            raise nncf.InternalError(msg)

    def get_sparsity_ratio(self, epoch: int) -> float:
        if epoch == 0:
            return self.initial_ratio
        if epoch >= self.target_epoch:
            return self.target_ratio
        d_init = 1 - self.initial_ratio
        d_target = 1 - self.target_ratio
        d = d_init * (d_target / d_init) ** (epoch / self.target_epoch)
        ratio = 1 - d
        return min(max(self.initial_ratio, ratio), self.target_epoch)
