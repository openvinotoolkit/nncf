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

from typing import Any, Optional, cast

import numpy as np
import scipy.optimize  # type: ignore[import-untyped]

from nncf.api.compression import CompressionAlgorithmController
from nncf.common.schedulers import BaseCompressionScheduler
from nncf.common.schedulers import ExponentialDecaySchedule
from nncf.common.utils.api_marker import api
from nncf.common.utils.registry import Registry
from nncf.config.schemata.defaults import PRUNING_NUM_INIT_STEPS
from nncf.config.schemata.defaults import PRUNING_STEPS
from nncf.config.schemata.defaults import PRUNING_TARGET

PRUNING_SCHEDULERS = Registry("pruning_schedulers")


@api()
class PruningScheduler(BaseCompressionScheduler):
    """
    This is the class from which all pruning schedulers inherit.

    A pruning scheduler is an object which specifies the pruning
    level at each training epoch. It involves a scheduling algorithm,
    defined in the `_calculate_pruning_level()` method and a state
    (some parameters required for current pruning level calculation)
    defined in the `__init__()` method.

    :param controller: Pruning algorithm controller.
    :param params: Parameters of the scheduler in the JSON-like dictionary form. Passed as-is from the corresponding
      section of the NNCF config file .json section (https://openvinotoolkit.github.io/nncf/schema).
    """

    def __init__(self, controller: CompressionAlgorithmController, params: dict[str, Any]):
        super().__init__()
        self._controller = controller
        self.initial_level = self._controller.pruning_init  # type: ignore[attr-defined]

        if self._controller.prune_flops:  # type: ignore[attr-defined]
            self.target_level = cast(float, params.get("pruning_flops_target"))
        else:
            self.target_level = cast(float, params.get("pruning_target", PRUNING_TARGET))

        self.num_warmup_epochs = cast(int, params.get("num_init_steps", PRUNING_NUM_INIT_STEPS))
        self.num_pruning_epochs = cast(int, params.get("pruning_steps", PRUNING_STEPS))
        self.freeze_epoch = self.num_warmup_epochs + self.num_pruning_epochs

    def _calculate_pruning_level(self) -> float:
        """
        Calculates a pruning level that should be applied to the model
        for the `current_epoch`.

        :return: Pruning level that should be applied to the model.
        """
        msg = "PruningScheduler implementation must override _calculate_pruning_level method."
        raise NotImplementedError(msg)

    def epoch_step(self, next_epoch: Optional[int] = None) -> None:
        """
        Should be called at the beginning of each training epoch to prepare
        the pruning method to continue training the model in the `next_epoch`.

        :param next_epoch: The epoch index for which the pruning scheduler
            will update the state of the pruning method.
        """
        super().epoch_step(next_epoch)
        self._controller.set_pruning_level(self.current_pruning_level)  # type: ignore[attr-defined]
        if self.current_epoch >= self.freeze_epoch:
            self._controller.freeze()  # type: ignore[attr-defined]

    def step(self, next_step: Optional[int] = None) -> None:
        """
        Should be called at the beginning of each training step to prepare
        the pruning method to continue training the model in the `next_step`.

        :param next_step: The global step index for which the pruning scheduler
            will update the state of the pruning method.
        """
        super().step(next_step)
        self._controller.step(next_step)  # type: ignore[attr-defined]

    @property
    def current_pruning_level(self) -> float:
        """
        Returns pruning level for the `current_epoch`.

        :return: Current sparsity level.
        """
        if self.current_epoch >= self.num_warmup_epochs:
            return self._calculate_pruning_level()
        return 0


@PRUNING_SCHEDULERS.register("baseline")
class BaselinePruningScheduler(PruningScheduler):
    """
    Pruning scheduler which applies the same pruning level for each epoch.

    The model is trained without pruning during `num_warmup_epochs` epochs.
    Then scheduler sets `target_level` and freezes the algorithm.
    """

    def __init__(self, controller: CompressionAlgorithmController, params: dict[str, Any]):
        super().__init__(controller, params)
        self.freeze_epoch = self.num_warmup_epochs

    def _calculate_pruning_level(self) -> float:
        return self.target_level


@PRUNING_SCHEDULERS.register("exponential")
class ExponentialPruningScheduler(PruningScheduler):
    """
    Pruning scheduler with an exponential decay schedule.

    This scheduler applies exponential decay to the density level
    to calculate the pruning level for the `current_epoch`.
    The density level for the `current_epoch` is calculated as

        current_density = 1.0 - current_level
    """

    def __init__(self, controller: CompressionAlgorithmController, params: dict[str, Any]):
        """
        Initializes a pruning scheduler with an exponential decay schedule.

        :param controller: Sparsity algorithm controller.
        :param params: Parameters of the scheduler.
        """
        super().__init__(controller, params)
        initial_density = 1.0 - self.initial_level
        target_density = 1.0 - self.target_level
        target_epoch = self.num_pruning_epochs - 1
        self.schedule = ExponentialDecaySchedule(initial_density, target_density, target_epoch)

    def _calculate_pruning_level(self) -> float:
        current_density = self.schedule(self.current_epoch - self.num_warmup_epochs)
        current_level = 1.0 - current_density
        return min(current_level, self.target_level)


@PRUNING_SCHEDULERS.register("exponential_with_bias")
class ExponentialWithBiasPruningScheduler(PruningScheduler):
    """
    Pruning scheduler which calculates pruning level for the current epoch
    according to the formula:

        current_level = a * exp(-k * epoch_idx) + b,

    where a, b, k is a params.
    """

    def __init__(self, controller: CompressionAlgorithmController, params: dict[str, Any]):
        """
        Initializes a pruning scheduler with an exponential (with bias) decay schedule.

        :param controller: Sparsity algorithm controller.
        :param params: Parameters of the scheduler.
        """
        super().__init__(controller, params)
        target_epoch = self.num_pruning_epochs - 1
        self.a, self.b, self.k = self._init_exp(target_epoch, self.initial_level, self.target_level)

    def _calculate_pruning_level(self) -> float:
        current_level = self.a * float(np.exp(-self.k * (self.current_epoch - self.num_warmup_epochs))) + self.b
        return min(current_level, self.target_level)

    @staticmethod
    def _init_exp(epoch_idx: int, p_min: float, p_max: float, factor: float = 0.125) -> tuple[float, float, float]:
        """
        Finds parameters a, b, k from the system:
            p_min = a + b
            p_max = a * exp(-k * epoch_idx) + b
            3/4 * p_max = a * exp(-k * factor * epoch_idx) + b
        For more details see [paper](https://arxiv.org/pdf/1808.07471.pdf).

        :param epoch_idx: Zero-based index of the epoch for which the a * exp(-k * epoch_idx) + b = p_max.
        :param p_min: Initial pruning level at which the schedule begins.
        :param p_max: Target pruning level at which the schedule ends.
        :param factor: Hyperparameter.
        """

        def get_b(a: float) -> float:
            return p_min - a

        def get_a(k: float) -> float:
            return (p_max - p_min) / (float(np.exp(-k * epoch_idx)) - 1)

        def f_to_solve(x: float) -> float:
            c = (0.75 * p_max - p_min) / (p_max - p_min)
            y = float(np.exp(-x * epoch_idx))
            return float(y**factor) - c * y + c - 1

        k = float(scipy.optimize.fsolve(f_to_solve, [1])[0])
        a = get_a(k)
        b = get_b(a)
        return a, b, k
