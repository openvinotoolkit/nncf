"""
 Copyright (c) 2022 Intel Corporation
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

from typing import Optional, Dict, Any

from nncf.common.utils.logger import logger
from nncf.common.utils.registry import Registry
from nncf.common.schedulers import PolynomialDecaySchedule
from nncf.common.schedulers import ExponentialDecaySchedule
from nncf.common.schedulers import MultiStepSchedule
from nncf.common.sparsity.controller import SparsityController
from nncf.common.schedulers import BaseCompressionScheduler

SPARSITY_SCHEDULERS = Registry('sparsity_schedulers')


class SparsityScheduler(BaseCompressionScheduler):
    """
    This is the class from which all sparsity schedulers inherit.

    A sparsity scheduler is an object which specifies the sparsity
    level at each training epoch or each training step. It involves a
    scheduling algorithm, defined in the `_calculate_sparsity_level()`
    method and a state (some parameters required for current sparsity
    level calculation) defined in the `__init__()` method.

    :param initial_level: Sparsity level which already has been
        applied to the model. It is the level at which the schedule begins.
    :param target_level: Sparsity level at which the schedule ends.
    :param target_epoch: Zero-based index of the epoch from which the
        sparsity level of the model will be equal to the `target_level`.
    :param freeze_epoch: Zero-based index of the epoch from which the sparsity
        mask will be frozen and will not be trained.
    """

    def __init__(self, controller: SparsityController, params: dict):
        """
        Initializes the internal state of the sparsity scheduler.

        :param controller: Sparsity algorithm controller.
        :param params: Parameters of the scheduler.
        """
        super().__init__()
        self._controller = controller
        self.initial_level = params.get('sparsity_init')
        self.target_level = params.get('sparsity_target', 0.5)
        self.target_epoch = params.get('sparsity_target_epoch', 90)
        self.freeze_epoch = params.get('sparsity_freeze_epoch', 100)

    def _calculate_sparsity_level(self) -> float:
        """
        Calculates a sparsity level that should be applied to the weights
        for the `current_epoch` or for step in the `current_epoch`.

        :return: Sparsity level that should be applied to the weights
            for the `current_epoch` or for step in the `current_epoch`.
        """
        raise NotImplementedError(
            'SparsityScheduler implementation must override _calculate_sparsity_level method.')

    def _update_sparsity_level(self) -> None:
        """
        Calculates the current sparsity level and updates the internal
        state of the `controller`.
        """
        if self.current_epoch >= self.freeze_epoch:
            self._controller.freeze()
        self._controller.set_sparsity_level(self._calculate_sparsity_level())

    @property
    def current_sparsity_level(self) -> float:
        """
        Returns sparsity level for the `current_epoch` or for step
        in the `current_epoch`.

        :return: Current sparsity level.
        """
        if self._current_epoch == -1:
            return self.initial_level
        return self._calculate_sparsity_level()


@SPARSITY_SCHEDULERS.register('polynomial')
class PolynomialSparsityScheduler(SparsityScheduler):
    """
    Sparsity scheduler with a polynomial decay schedule.

    Two ways are available for calculations of the sparsity:
        - per epoch
        - per step
    Parameters `update_per_optimizer_step` and `steps_per_epoch`
    should be provided in config for the per step calculation.
    If `update_per_optimizer_step` was only provided then scheduler
    will use first epoch to calculate `steps_per_epoch`
    parameter. In this case, `current_epoch` and `current_step` will
    not be updated on this epoch. The scheduler will start calculation
    after `steps_per_epoch` will be calculated.
    """

    def __init__(self, controller: SparsityController, params: dict):
        """
        Initializes a sparsity scheduler with a polynomial decay schedule.

        :param controller: Sparsity algorithm controller.
        :param params: Parameters of the scheduler.
        """
        super().__init__(controller, params)
        self.schedule = PolynomialDecaySchedule(self.initial_level, self.target_level, self.target_epoch,
                                                params.get('power', 0.9), params.get('concave', True))
        self._steps_in_current_epoch = 0
        self._update_per_optimizer_step = params.get('update_per_optimizer_step', False)
        self._steps_per_epoch = params.get('steps_per_epoch', None)
        self._should_skip = False

    def step(self, next_step: Optional[int] = None) -> None:
        self._steps_in_current_epoch += 1
        if self._should_skip:
            return

        super().step(next_step)
        if self._update_per_optimizer_step:
            self._update_sparsity_level()

    def epoch_step(self, next_epoch: Optional[int] = None) -> None:
        self._maybe_should_skip()
        if self._should_skip:
            return

        self._steps_in_current_epoch = 0

        super().epoch_step(next_epoch)
        if not self._update_per_optimizer_step:
            self._update_sparsity_level()

    def _calculate_sparsity_level(self) -> float:
        local_step = max(self._steps_in_current_epoch - 1, 0)
        return self.schedule(self.current_epoch, local_step, self._steps_per_epoch)

    def load_state(self, state: Dict[str, Any]) -> None:
        super().load_state(state)
        if self._update_per_optimizer_step:
            self._steps_per_epoch = state['_steps_per_epoch']

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        if self._update_per_optimizer_step:
            state['_steps_per_epoch'] = self._steps_per_epoch
        return state

    def _maybe_should_skip(self) -> None:
        """
        Checks if the first epoch (with index 0) should be skipped to calculate
        the steps per epoch. If the skip is needed, then the internal state
        of the scheduler object will not be changed.
        """
        self._should_skip = False
        if self._update_per_optimizer_step:
            if self._steps_per_epoch is None and self._steps_in_current_epoch > 0:
                self._steps_per_epoch = self._steps_in_current_epoch

            if self._steps_per_epoch is not None and self._steps_in_current_epoch > 0:
                if self._steps_per_epoch != self._steps_in_current_epoch:
                    raise Exception('Actual steps per epoch and steps per epoch from the scheduler '
                                    'parameters are different. Scheduling may be incorrect.')

            if self._steps_per_epoch is None:
                self._should_skip = True
                logger.warning('Scheduler set to update sparsity level per optimizer step, '
                               'but steps_per_epoch was not set in config. Will only start updating '
                               'sparsity level after measuring the actual steps per epoch as signaled '
                               'by a .epoch_step() call.')


@SPARSITY_SCHEDULERS.register('exponential')
class ExponentialSparsityScheduler(SparsityScheduler):
    """
    Sparsity scheduler with an exponential decay schedule.

    This scheduler applies exponential decay to the density level
    to calculate the sparsity level for the `current_epoch`.
    The density level for the `current_epoch` is calculated as

        current_density = 1.0 - current_level
    """

    def __init__(self, controller: SparsityController, params: dict):
        """
        Initializes a sparsity scheduler with an exponential decay schedule.

        :param controller: Sparsity algorithm controller.
        :param params: Parameters of the scheduler.
        """
        super().__init__(controller, params)
        initial_density = 1.0 - self.initial_level
        target_density = 1.0 - self.target_level
        self.schedule = ExponentialDecaySchedule(initial_density, target_density, self.target_epoch)

    def epoch_step(self, next_epoch: Optional[int] = None) -> None:
        super().epoch_step(next_epoch)
        self._update_sparsity_level()

    def _calculate_sparsity_level(self) -> float:
        current_density = self.schedule(self.current_epoch)
        current_level = 1.0 - current_density
        return min(current_level, self.target_level)


@SPARSITY_SCHEDULERS.register('adaptive')
class AdaptiveSparsityScheduler(SparsityScheduler):
    """
    Sparsity scheduler with an adaptive schedule.
    """
    def __init__(self, controller: SparsityController, params: dict):
        """
        Initializes a sparsity scheduler with an adaptive schedule.

        :param controller: Sparsity algorithm controller.
        :param params: Parameters of the scheduler.
        """
        super().__init__(controller, params)
        self.decay_step = params.get('step', 0.05)
        self.eps = params.get('eps', 0.03)
        self.patience = params.get('patience', 1)
        self.num_bad_epochs = 0
        self._current_level = self.initial_level

    @property
    def current_sparsity_level(self) -> float:
        """
        Returns sparsity level for the `current_epoch` or for step
        in the `current_epoch`.

        :return: Current sparsity level.
        """
        return self._current_level

    def epoch_step(self, next_epoch: Optional[int] = None) -> None:
        super().epoch_step(next_epoch)
        self._update_sparsity_level()

    def _calculate_sparsity_level(self) -> float:
        if self._controller.loss.current_sparsity >= self._current_level - self.eps:
            self.num_bad_epochs += 1

        current_level = self._current_level
        if self.num_bad_epochs >= self.patience:
            self.num_bad_epochs = 0
            current_level = current_level + self.decay_step

        self._current_level = min(current_level, self.target_level)

        return self._current_level

    def load_state(self, state: Dict[str, Any]) -> None:
        super().load_state(state)
        self.num_bad_epochs = state['num_bad_epochs']
        self._current_level = state['current_sparsity_level']

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state['num_bad_epochs'] = self.num_bad_epochs
        state['current_sparsity_level'] = self._current_level
        return state


@SPARSITY_SCHEDULERS.register('multistep')
class MultiStepSparsityScheduler(SparsityScheduler):
    """
    Sparsity scheduler with a piecewise constant schedule.
    """

    def __init__(self, controller: SparsityController, params: dict):
        """
        Initializes a sparsity scheduler with a piecewise constant schedule.

        :param controller: Sparsity algorithm controller.
        :param params: Parameters of the scheduler.
        """
        super().__init__(controller, params)
        self.schedule = MultiStepSchedule(
            sorted(params.get('multistep_steps', [90])), params.get('multistep_sparsity_levels', [0.1, 0.5]))
        self.target_level = self.schedule.values[-1]

    @property
    def current_sparsity_level(self) -> float:
        """
        Returns sparsity level for the `current_epoch` or for step
        in the `current_epoch`.

        :return: Current sparsity level.
        """
        if self._current_epoch == -1:
            return self.schedule.values[0]
        return self._calculate_sparsity_level()

    def epoch_step(self, next_epoch: Optional[int] = None) -> None:
        super().epoch_step(next_epoch)
        self._update_sparsity_level()

    def _calculate_sparsity_level(self) -> float:
        return self.schedule(self.current_epoch)
