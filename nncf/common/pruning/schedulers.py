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

from typing import Optional
import numpy as np
import scipy.optimize

from nncf.api.compression import CompressionScheduler
from nncf.common.utils.registry import Registry
from nncf.common.schedulers import ExponentialDecaySchedule

PRUNING_SCHEDULERS = Registry("pruning_schedulers")


class PruningScheduler(CompressionScheduler):
    """
    This is the class from which all pruning schedulers inherit.

    A pruning scheduler is an object which specifies the pruning
    level at each training epoch. It involves a scheduling algorithm,
    defined in the `_calculate_pruning_level()` method and a state
    (some parameters required for current pruning level calculation)
    defined in the `__init__()` method.
    """

    def __init__(self, controller, params: dict):
        """
        Initializes the internal state of the pruning scheduler specified by:
            - controller: Pruning algorithm controller.
            - initial_pruning: Pruning level which already has been
                applied to the model. It is the level at which the schedule begins.
            - target_pruning: Pruning level at which the schedule ends.
            - num_warmup_epochs: Number of epochs for model pre-training before pruning.
            - num_pruning_epochs: Number of epochs during which the pruning level
                is increased from `initial_pruning` to `target_pruning`.
            - freeze_epoch: Zero-based index of the epoch from which the pruning
                mask will be frozen and will not be trained.
            - current_pruning: Pruning level for the `current_epoch`.

        :param controller: Pruning algorithm controller.
        :param params: Parameters of the scheduler.
        """
        super().__init__()
        self.controller = controller
        self.initial_pruning = self.controller.pruning_init

        if self.controller.prune_flops:
            self.target_pruning = params.get('pruning_flops_target')
        else:
            self.target_pruning = params.get('pruning_target', 0.5)

        self.num_warmup_epochs = params.get('num_init_steps', 0)
        self.num_pruning_epochs = params.get('pruning_steps', 100)
        self.freeze_epoch = self.num_warmup_epochs + self.num_pruning_epochs
        self.current_pruning = self.initial_pruning

    def _maybe_freeze(self) -> None:
        """
        Checks if pruning mask needs to be frozen and not to be trained.
        If freezing is needed, then the internal state of the `controller`
        object will be changed.
        """
        if self.current_epoch >= self.freeze_epoch:
            self.controller.freeze()

    def _calculate_pruning_level(self) -> float:
        """
        Calculates a pruning level that should be applied to the model
        for the `current_epoch`.

        :return: Pruning level that should be applied to the model.
        """
        raise NotImplementedError(
            'PruningScheduler implementation must override _calculate_pruning_level method.')

    def epoch_step(self, next_epoch: Optional[int] = None) -> None:
        super().epoch_step(next_epoch)
        self.current_pruning = self._calculate_pruning_level()
        self.controller.set_pruning_rate(self.pruning_level)
        self._maybe_freeze()

    def step(self, next_step: Optional[int] = None) -> None:
        super().step(next_step)
        self.controller.step(next_step)

    @property
    def pruning_level(self) -> float:
        """
        Returns pruning level for the `current_epoch`.

        :return: Current sparsity level.
        """
        if self.current_epoch >= self.num_warmup_epochs:
            return self.current_pruning
        return 0


@PRUNING_SCHEDULERS.register("baseline")
class BaselinePruningScheduler(PruningScheduler):
    """
    Pruning scheduler which applies the same pruning level for each epoch.

    The model is trained without pruning during `num_warmup_epochs` epochs.
    Then scheduler sets `target_pruning` and freezes the algorithm.
    """

    def __init__(self, controller, params: dict):
        super().__init__(controller, params)
        self.freeze_epoch = self.num_warmup_epochs

    def _calculate_pruning_level(self) -> float:
        return self.target_pruning


@PRUNING_SCHEDULERS.register("exponential")
class ExponentialPruningScheduler(PruningScheduler):
    """
    Pruning scheduler with an exponential decay schedule.

    This scheduler applies exponential decay to the density level
    to calculate the pruning level for the `current_epoch`.
    The density level for the `current_epoch` is calculated as

        current_density = 1.0 - current_pruning
    """

    def __init__(self, controller, params: dict):
        """
        Initializes a pruning scheduler with an exponential decay schedule.

        :param controller: Sparsity algorithm controller.
        :param params: Parameters of the scheduler.
        """
        super().__init__(controller, params)
        initial_density = 1.0 - self.initial_pruning
        target_density = 1.0 - self.target_pruning
        self.schedule = ExponentialDecaySchedule(initial_density, target_density, self.num_pruning_epochs)

    def _calculate_pruning_level(self) -> float:
        current_density = self.schedule(self.current_epoch - self.num_warmup_epochs)
        current_pruning = 1.0 - current_density
        return min(current_pruning, self.target_pruning)


@PRUNING_SCHEDULERS.register("exponential_with_bias")
class ExponentialWithBiasPruningScheduler(PruningScheduler):
    """
    Calculates pruning rate progressively according to the formula
    P = a * exp(- k * epoch) + b
    Where:
    epoch - epoch number
    P - pruning rate for current epoch
    a, b, k - params
    """
    def __init__(self, controller, params: dict):
        super().__init__(controller, params)
        self.a, self.b, self.k = self._init_exp(self.num_pruning_epochs, self.initial_pruning, self.target_pruning)

    def _calculate_pruning_level(self) -> float:
        curr_pruning = self.a * np.exp(-self.k * (self.current_epoch - self.num_warmup_epochs - 1)) + self.b
        max_pruning = self.target_pruning
        return max_pruning if curr_pruning >= max_pruning else curr_pruning

    @staticmethod
    def _init_exp(E_max, P_min, P_max, D=1 / 8):
        """
        Find a, b, k for system (from SPFP paper):
        1. P_min = a + b
        2. P_max = a * exp(-k * E_max) + b
        3. 3/4 * P_max = a *  exp(-k * E_max * D) + b
        Where P_min, P_max - minimal and goal levels of pruning rate
        E_max - number of epochs for pruning
        """
        def get_b(a, k):
            return P_min - a

        def get_a(k):
            return (3 / 4 * P_max - P_min) / (np.exp(- D * k * E_max) - 1)

        def f_to_solve(x):
            y = np.exp(D * x * E_max)
            return 1 / 3 * y + 1 / (y ** 7) - 4 / 3

        k = scipy.optimize.fsolve(f_to_solve, [1])[0]
        a = get_a(k)
        b = get_b(a, k)
        return a, b, k
