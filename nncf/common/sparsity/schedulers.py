"""
 Copyright (c) 2019-2020 Intel Corporation
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

import numpy as np
from bisect import bisect_right

from nncf.common.utils.logger import logger
from nncf.common.utils.registry import Registry
from nncf.api.compression import CompressionLevel
from nncf.api.compression import CompressionScheduler

SPARSITY_SCHEDULERS = Registry("sparsity_schedulers")


class SparsityScheduler(CompressionScheduler):
    def __init__(self, controller, params: dict = None):
        super().__init__()
        if params is None:
            self._params = dict()
        else:
            self._params = params

        self.controller = controller
        self.initial_sparsity = self.controller.get_sparsity_init()
        self.target_sparsity = self._params.get('sparsity_target', 0.5)
        self.target_epoch = self._params.get('sparsity_target_epoch', 90)
        self.freeze_epoch = self._params.get('sparsity_freeze_epoch', 100)

    def _set_sparsity_level(self):
        if self.current_epoch >= self.freeze_epoch:
            self.controller.freeze()
        self.controller.set_sparsity_level(self.current_sparsity_level)

    @property
    def current_sparsity_level(self):
        raise NotImplementedError

    def compression_level(self) -> CompressionLevel:
        if self.current_sparsity_level == 0:
            return CompressionLevel.NONE
        if self.current_sparsity_level >= self.target_sparsity:
            return CompressionLevel.FULL
        return CompressionLevel.PARTIAL


@SPARSITY_SCHEDULERS.register("polynomial")
class PolynomialSparseScheduler(SparsityScheduler):
    def __init__(self, controller, params=None):
        super().__init__(controller, params)
        self._steps_in_current_epoch = 0
        self.power = self._params.get('power', 0.9)
        self.concave = self._params.get('concave', True)
        self._update_per_optimizer_step = self._params.get('update_per_optimizer_step', False)
        if self._update_per_optimizer_step:
            self._steps_per_epoch = self._params.get('steps_per_epoch')
            if self._steps_per_epoch is None:
                logger.warning("Optimizer set to update sparsity level per optimizer step,"
                               "but steps_per_epoch was not set in config. Will only start updating "
                               "sparsity level after measuring the actual steps per epoch as signaled "
                               "by a .epoch_step() call.")

    def step(self, next_step=None):
        super().step(next_step)
        self._steps_in_current_epoch += 1
        if self._update_per_optimizer_step and self._steps_per_epoch is not None:
            self._set_sparsity_level()

    def epoch_step(self, next_epoch=None):
        if self._update_per_optimizer_step:
            if self.current_epoch == 0 and self._steps_in_current_epoch > 0 and self._steps_per_epoch is None:
                self._steps_per_epoch = self._steps_in_current_epoch

                # Reset step and epoch step counters
                next_epoch = 0
            if self._steps_in_current_epoch != self._steps_per_epoch and self._steps_in_current_epoch > 0:
                self._steps_per_epoch = self._steps_in_current_epoch
                logger.warning("Actual optimizer steps per epoch is different than what is "
                               "specified by scheduler parameters! Scheduling may be incorrect. "
                               "Setting scheduler's global step count to (current epoch) * "
                               "(actual steps per epoch)")
                self.step(self._steps_per_epoch * (self.current_epoch - 1))

        self._steps_in_current_epoch = 0
        super().epoch_step(next_epoch)
        if not self._update_per_optimizer_step:
            self._set_sparsity_level()

    def get_state(self):
        sd = super().get_state()
        if self._update_per_optimizer_step:
            sd['_steps_per_epoch'] = self._steps_per_epoch
        return sd

    @property
    def current_sparsity_level(self):
        if self.target_epoch == 0:
            return self.target_sparsity

        if self._update_per_optimizer_step:
            if self._steps_per_epoch is None:
                return self.initial_sparsity  # Cannot do proper sparsity update until the steps in an epoch are counted

            fractional_epoch = self.current_epoch + self.current_step_in_current_epoch / self._steps_per_epoch
            progress = fractional_epoch / self.target_epoch
        else:
            progress = self.current_epoch / self.target_epoch
        progress = min(1.0, max(0.0, progress))

        if self.concave:
            current_sparsity = self.target_sparsity - (self.target_sparsity - self.initial_sparsity) * (
                (1 - progress) ** self.power)
        else:
            current_sparsity = self.initial_sparsity + (self.target_sparsity - self.initial_sparsity) * (
                progress ** self.power)

        return current_sparsity

    @property
    def current_step_in_current_epoch(self):
        return self._steps_in_current_epoch - 1 if self._steps_in_current_epoch > 0 else 0


@SPARSITY_SCHEDULERS.register("exponential")
class ExponentialSparsityScheduler(SparsityScheduler):
    def __init__(self, controller, params=None):
        super().__init__(controller, params)
        self.a, self.k = self._init_exp(self.initial_sparsity, self.target_sparsity,
                                        sparsity_steps=self.target_epoch)

    def epoch_step(self, next_epoch=None):
        super().epoch_step(next_epoch)
        self._set_sparsity_level()

    @property
    def current_sparsity_level(self):
        if self.target_epoch == 0:
            return self.target_sparsity

        if self.current_epoch == -1:
            return self.initial_sparsity

        curr_sparsity = 1 - self.a * np.exp(-self.k * self.current_epoch)
        return curr_sparsity if curr_sparsity <= self.target_sparsity else self.target_sparsity

    @staticmethod
    def _init_exp(initial_sparsity, max_sparsity, sparsity_steps=20):
        p1 = (0, 1 - initial_sparsity)
        p2 = (sparsity_steps, 1 - max_sparsity)
        k = np.log(p2[1] / p1[1]) / (p1[0] - p2[0])
        a = p1[1] / np.exp(-k * p1[0])
        return a, k


@SPARSITY_SCHEDULERS.register("adaptive")
class AdaptiveSparsityScheduler(SparsityScheduler):
    def __init__(self, controller, params=None):
        super().__init__(controller, params)
        self.sparsity_loss = controller.loss
        self.decay_step = params.get('step', 0.05)
        self.eps = params.get('eps', 0.03)
        self.patience = params.get('patience', 1)
        self.current_sparsity_target = self.initial_sparsity
        self.num_bad_epochs = 0

    def epoch_step(self, next_epoch=None):
        super().epoch_step(next_epoch)
        if self.sparsity_loss.current_sparsity >= self.current_sparsity_target - self.eps:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            self.num_bad_epochs = 0
            self.current_sparsity_target = min(self.current_sparsity_target + self.decay_step, self.target_sparsity)
        self._set_sparsity_level()

    def get_state(self):
        sd = super().get_state()
        sd['num_bad_epochs'] = self.num_bad_epochs
        sd['current_sparsity_level'] = self.current_sparsity_level
        return sd

    @property
    def current_sparsity_level(self):
        return self.current_sparsity_target


@SPARSITY_SCHEDULERS.register("multistep")
class MultiStepSparsityScheduler(SparsityScheduler):
    def __init__(self, controller, params):
        super().__init__(controller, params)
        self.sparsity_levels = self._params.get('multistep_sparsity_levels', [0.1, 0.5])
        self.steps = sorted(self._params.get('multistep_steps', [90]))

        if len(self.steps) + 1 != len(self.sparsity_levels):
            raise ValueError('number of sparsity levels must equal to number of steps + 1')

        self.sparsity_level = self.sparsity_levels[0]
        self.target_sparsity = self.sparsity_levels[-1]

    def epoch_step(self, next_epoch=None):
        super().epoch_step(next_epoch)
        ind = bisect_right(self.steps, self.current_epoch)
        self.sparsity_level = self.sparsity_levels[ind]
        self._set_sparsity_level()

    @property
    def current_sparsity_level(self):
        return self.sparsity_level
