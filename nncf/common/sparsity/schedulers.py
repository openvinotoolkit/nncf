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

from nncf.common.logging import nncf_logger
from nncf.common.utils.registry import Registry
from nncf.common.schedulers import PolynomialDecaySchedule
from nncf.common.schedulers import ExponentialDecaySchedule
from nncf.common.schedulers import MultiStepSchedule
from nncf.common.sparsity.controller import SparsityController
from nncf.common.schedulers import BaseCompressionScheduler
from nncf.config.schemata.defaults import SPARSITY_FREEZE_EPOCH
from nncf.config.schemata.defaults import SPARSITY_MULTISTEP_SPARSITY_LEVELS
from nncf.config.schemata.defaults import SPARSITY_MULTISTEP_STEPS
from nncf.config.schemata.defaults import SPARSITY_SCHEDULER_CONCAVE
from nncf.config.schemata.defaults import SPARSITY_SCHEDULER_PATIENCE
from nncf.config.schemata.defaults import SPARSITY_SCHEDULER_POWER
from nncf.config.schemata.defaults import SPARSITY_SCHEDULER_UPDATE_PER_OPTIMIZER_STEP
from nncf.config.schemata.defaults import SPARSITY_TARGET
from nncf.config.schemata.defaults import SPARSITY_TARGET_EPOCH

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
        self.target_level = params.get('sparsity_target', SPARSITY_TARGET)
        self.target_epoch = params.get('sparsity_target_epoch', SPARSITY_TARGET_EPOCH)
        self.freeze_epoch = params.get('sparsity_freeze_epoch', SPARSITY_FREEZE_EPOCH)

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
                                                params.get('power', SPARSITY_SCHEDULER_POWER),
                                                params.get('concave', SPARSITY_SCHEDULER_CONCAVE))
        self._steps_in_current_epoch = 0
        self._update_per_optimizer_step = params.get('update_per_optimizer_step',
                                                     SPARSITY_SCHEDULER_UPDATE_PER_OPTIMIZER_STEP)
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
                nncf_logger.warning('Scheduler set to update sparsity level per optimizer step, '
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
        self.patience = params.get('patience', SPARSITY_SCHEDULER_PATIENCE)
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
            sorted(params.get('multistep_steps', SPARSITY_MULTISTEP_STEPS)),
            params.get('multistep_sparsity_levels', SPARSITY_MULTISTEP_SPARSITY_LEVELS))
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


@SPARSITY_SCHEDULERS.register('threshold_polynomial_decay')
class PolynomialThresholdScheduler(BaseCompressionScheduler):
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
        TODO: revise docstring
        TODO: test epoch-wise stepping
        Initializes a sparsity scheduler with a polynomial decay schedule.

        :param controller: Sparsity algorithm controller.
        :param params: Parameters of the scheduler.
        """
        super().__init__()
        self._controller = controller
        self.init_importance_threshold: float = params.get('init_importance_threshold', -1.)
        self.final_importance_threshold: float = params.get('final_importance_threshold', 0.)
        self.warmup_start_epoch: int = params.get('warmup_start_epoch', 1)
        self.warmup_end_epoch: int = params.get('warmup_end_epoch', 2)
        self.importance_target_lambda: float = params.get('importance_regularization_factor', 0.1)
        self.enable_structured_masking: bool = params.get('enable_structured_masking', True)
        self._steps_per_epoch = params.get('steps_per_epoch', None)

        if self._steps_per_epoch is None and self.warmup_start_epoch < 1:
            raise ValueError('`warmup_start_epoch` must be >= 1 in order to enable the auto calculation of `steps_per_epoch`. '
                             'Please either change `warmup_start_epoch` to a larger number or specify `steps_per_epoch` in the config.'
            )

        self.schedule = PolynomialDecaySchedule(
            self.init_importance_threshold,
            self.final_importance_threshold,
            (self.warmup_end_epoch - self.warmup_start_epoch),
            params.get('power', 3),
            params.get('concave', True)
        )
        self.current_importance_threshold = self.init_importance_threshold
        self._cached_importance_threshold = None
        self._is_importance_frozen = False
        self._steps_in_current_epoch = 0
        self._should_skip = False

    @property
    def current_importance_lambda(self):
        return self.importance_target_lambda * (self.current_importance_threshold - self.init_importance_threshold) / (self.final_importance_threshold - self.init_importance_threshold)

    def _freeze_importance(self):
        for minfo in self._controller.sparsified_module_info:
            minfo.operand.requires_grad_(False)

    def _update_operand_importance_threshold(self):
        if self.current_importance_threshold != self._cached_importance_threshold:
            for minfo in self._controller.sparsified_module_info:
                minfo.operand.importance_threshold = self.current_importance_threshold
        self.cached_importance_threshold = self.current_importance_threshold

    def epoch_step(self, next_epoch: Optional[int] = None) -> None:
        super().epoch_step(next_epoch)
        self._maybe_should_skip()
        self._steps_in_current_epoch = 0
        if self._should_skip:
            return
        self.schedule_threshold(self.current_step + 1) # useful when update_per_optimizer_step=False

    def step(self, next_step: Optional[int] = None) -> None:
        super().step(next_step)
        self._steps_in_current_epoch += 1
        if self._should_skip:
            return
        self.schedule_threshold(self.current_step)

    def schedule_threshold(self, global_step: Optional[int] = None):
        if global_step is None:
            global_step = self.current_step
        if global_step < self.warmup_start_epoch * self._steps_per_epoch:
            self.current_importance_threshold = self.init_importance_threshold
        elif global_step < self.warmup_end_epoch * self._steps_per_epoch:
            self.current_importance_threshold = self._calculate_scheduled_threshold(global_step)
        else:
            self.current_importance_threshold = self.final_importance_threshold
            if not self._is_importance_frozen:
                self._freeze_importance()
                if self.enable_structured_masking:
                    self._controller.reset_independent_structured_mask()
                    self._controller.resolve_structured_mask()
                    self._controller.populate_structured_mask()
                self._is_importance_frozen = True

        self._update_operand_importance_threshold()

    def _calculate_scheduled_threshold(self, global_step: int) -> float:
        schedule_current_step = global_step - self.warmup_start_epoch * self._steps_per_epoch
        schedule_epoch = schedule_current_step // self._steps_per_epoch
        schedule_step = schedule_current_step % self._steps_per_epoch
        return self.schedule(schedule_epoch, schedule_step, self._steps_per_epoch)

    def load_state(self, state: Dict[str, Any]) -> None:
        super().load_state(state)
        self._steps_per_epoch = state['_steps_per_epoch']
        if self._steps_per_epoch is None:  # It is the first epoch and `steps_per_epoch` not specified
            self._steps_in_current_epoch = self._current_step + 1
            self._should_skip = True
        else:
            self._steps_in_current_epoch = self._current_step % self._steps_per_epoch + 1

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state['_steps_per_epoch'] = self._steps_per_epoch
        return state

    def _maybe_should_skip(self) -> None:
        """
        Checks if the first epoch (with index 0) should be skipped to calculate
        the steps per epoch. If the skip is needed, then the internal state
        of the scheduler object will not be changed.
        """
        self._should_skip = False

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
