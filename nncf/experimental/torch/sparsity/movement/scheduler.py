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
from nncf.common.schedulers import PolynomialDecaySchedule
from nncf.common.sparsity.controller import SparsityController
from nncf.common.schedulers import BaseCompressionScheduler


class MovementPolynomialThresholdScheduler(BaseCompressionScheduler):
    """
    Movement Sparsity scheduler with a polynomial decay schedule.

    The scheduler will update importance threshold and importance regularization
    factor per optimizer step. Parameter `steps_per_epoch` should be provided 
    in config for the per step calculation. If not provided, the scheduler will 
    use the first epoch to calculate `steps_per_epoch` parameter. In this case,
    parameter `warmup_start_epoch` must be equal to or larger than 1, and the 
    scheduler will start calculation only after `steps_per_epoch` is calculated.
    """

    def __init__(self, controller: 'MovementSparsityController', params: dict):
        """
        TODO: revise docstring
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
            raise ValueError('`warmup_start_epoch` must be >= 1 to enable the auto calculation of '
                             '`steps_per_epoch`. Please either change `warmup_start_epoch` to a larger '
                             'number or specify `steps_per_epoch` in the config.')

        self.schedule = PolynomialDecaySchedule(
            self.init_importance_threshold,
            self.final_importance_threshold,
            (self.warmup_end_epoch - self.warmup_start_epoch),
            params.get('power', 3),
            concave=True
        )
        self.current_importance_threshold = self.init_importance_threshold
        self._cached_importance_threshold = None
        self._is_importance_frozen = False
        self._steps_in_current_epoch = 0
        self._should_skip = False

    @property
    def current_importance_lambda(self):
        return self.importance_target_lambda * (self.current_importance_threshold - self.init_importance_threshold) / (
            self.final_importance_threshold - self.init_importance_threshold)

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
