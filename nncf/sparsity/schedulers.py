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
from ..algo_selector import Registry
from ..compression_method_api import CompressionScheduler, CompressionLevel

SPARSITY_SCHEDULERS = Registry("sparsity_schedulers")


class SparsityScheduler(CompressionScheduler):
    def __init__(self, sparsity_algo, params: dict = None):
        super().__init__()
        if params is None:
            self._params = dict()
        else:
            self._params = params

        self.algo = sparsity_algo
        self.initial_sparsity = self.algo.get_sparsity_init()
        self.sparsity_target = self._params.get('sparsity_target', 0.5)
        self.sparsity_target_epoch = self._params.get('sparsity_target_epoch', 90)
        self.sparsity_freeze_epoch = self._params.get('sparsity_freeze_epoch', 100)

    def epoch_step(self, next_epoch=None):
        super().epoch_step(next_epoch)
        self._set_sparsity_level()

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._set_sparsity_level()

    def _set_sparsity_level(self):
        if self.current_epoch >= self.sparsity_freeze_epoch:
            self.algo.freeze()
        self.algo.set_sparsity_level(self.current_sparsity_level)

    def _calc_density_level(self):
        return 1 - self.current_sparsity_level

    @property
    def current_sparsity_level(self):
        raise NotImplementedError

    @property
    def target_sparsity_level(self) -> float:
        return self.sparsity_target

    def compression_level(self) -> CompressionLevel:
        if self.current_sparsity_level == 0:
            return CompressionLevel.NONE
        if self.current_sparsity_level >= self.target_sparsity_level:
            return CompressionLevel.FULL
        return CompressionLevel.PARTIAL


@SPARSITY_SCHEDULERS.register("polynomial")
class PolynomialSparseScheduler(SparsityScheduler):
    def __init__(self, sparsity_algo, params=None):
        super().__init__(sparsity_algo, params)
        self.power = self._params.get('power', 0.9)
        self.concave = self._params.get('concave', False)
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
        if self._update_per_optimizer_step and self._steps_per_epoch is not None:
            self._set_sparsity_level()

    def epoch_step(self, next_epoch=None):
        if self._update_per_optimizer_step:
            if self.current_epoch == 0 and self._steps_in_current_epoch > 0 and self._steps_per_epoch is None:
                self._steps_per_epoch = self._steps_in_current_epoch

                # Reset step and epoch step counters
                next_epoch = -1
            if self._steps_in_current_epoch != self._steps_per_epoch and self._steps_in_current_epoch > 0:
                self._steps_per_epoch = self._steps_in_current_epoch
                logger.warning("Actual optimizer steps per epoch is different than what is "
                               "specified by scheduler parameters! Scheduling may be incorrect. "
                               "Setting scheduler's global step count to (current epoch) * "
                               "(actual steps per epoch)")
                self.step(self._steps_per_epoch * (self.current_epoch - 1))

        self._steps_in_current_epoch = 0
        super().epoch_step(next_epoch)

    def state_dict(self):
        sd = super().state_dict()
        if self._update_per_optimizer_step:
            sd['_steps_per_epoch'] = self._steps_per_epoch
        return sd

    @property
    def current_sparsity_level(self):
        if self.sparsity_target_epoch == 0 and not self._update_per_optimizer_step:
            return self.sparsity_target
        if self._update_per_optimizer_step:
            if self._steps_per_epoch is None:
                return self.initial_sparsity  # Cannot do proper sparsity update until the steps in an epoch are counted
            fractional_epoch = self.current_epoch + self.current_step_in_current_epoch / self._steps_per_epoch
            if self.sparsity_target_epoch != 0:
                progress = (min(self.sparsity_target_epoch, fractional_epoch) / self.sparsity_target_epoch)
            else:
                progress = fractional_epoch
        else:
            progress = (min(self.sparsity_target_epoch, self.current_epoch) / self.sparsity_target_epoch)

        if self.concave:
            current_sparsity = self.initial_sparsity + (self.sparsity_target - self.initial_sparsity) * (
                progress ** self.power)
        else:
            current_sparsity = self.sparsity_target - (self.sparsity_target - self.initial_sparsity) * (
                (1 - progress) ** self.power)
        return current_sparsity

    @property
    def current_step_in_current_epoch(self):
        return self._steps_in_current_epoch - 1 if self._steps_in_current_epoch > 0 else 0

@SPARSITY_SCHEDULERS.register("exponential")
class ExponentialSparsityScheduler(SparsityScheduler):
    def __init__(self, sparsity_algo, params=None):
        super().__init__(sparsity_algo, params)
        self.a, self.k = self._init_exp(self.initial_sparsity, self.sparsity_target,
                                        sparsity_steps=self.sparsity_target_epoch)

    @property
    def current_sparsity_level(self):
        if self.sparsity_target_epoch == 0:
            return self.sparsity_target
        curr_sparsity = 1 - self.a * np.exp(-self.k * self.current_epoch)
        return curr_sparsity if curr_sparsity <= self.sparsity_target else self.sparsity_target

    @staticmethod
    def _init_exp(initial_sparsity, max_sparsity, sparsity_steps=20):
        p1 = (0, 1 - initial_sparsity)
        p2 = (sparsity_steps, 1 - max_sparsity)
        k = np.log(p2[1] / p1[1]) / (p1[0] - p2[0])
        a = p1[1] / np.exp(-k * p1[0])
        return a, k


@SPARSITY_SCHEDULERS.register("adaptive")
class AdaptiveSparsityScheduler(SparsityScheduler):
    def __init__(self, sparsity_algo, params=None):
        super().__init__(sparsity_algo, params)
        self.sparsity_loss = sparsity_algo.loss
        from .rb.loss import SparseLoss
        if not isinstance(self.sparsity_loss, SparseLoss):
            raise TypeError('AdaptiveSparseScheduler expects SparseLoss, but {} is given'.format(
                self.sparsity_loss.__class__.__name__))
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
            self.current_sparsity_target = min(self.current_sparsity_target + self.decay_step, self.sparsity_target)
        self._set_sparsity_level()

    def state_dict(self):
        sd = super().state_dict()
        sd['num_bad_epochs'] = self.num_bad_epochs
        sd['current_sparsity_level'] = self.current_sparsity_level
        return sd

    @property
    def current_sparsity_level(self):
        return self.current_sparsity_target


@SPARSITY_SCHEDULERS.register("multistep")
class MultiStepSparsityScheduler(SparsityScheduler):
    def __init__(self, sparsity_algo, params):
        super().__init__(sparsity_algo, params)
        self.sparsity_levels = self._params.get('multistep_sparsity_levels', [0.1, 0.5])
        self.steps = sorted(self._params.get('multistep_steps', [90]))
        if len(self.steps) + 1 != len(self.sparsity_levels):
            raise AttributeError('number of sparsity levels must equal to number of steps + 1')

        self.initial_sparsity = self.sparsity_level = self.sparsity_levels[0]
        self.prev_ind = 0

    def epoch_step(self, next_epoch=None):
        super().epoch_step(next_epoch)
        ind = bisect_right(self.steps, self.current_epoch)
        if ind != self.prev_ind:
            self.sparsity_level = self.sparsity_levels[ind]
            self.prev_ind = ind
        self._set_sparsity_level()

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        ind = bisect_right(self.steps, self._current_epoch)
        if ind > 0:
            self.prev_ind = ind
            self.sparsity_level = self.sparsity_levels[ind]
            self._set_sparsity_level()

    @property
    def current_sparsity_level(self):
        return self.sparsity_level

    @property
    def target_sparsity_level(self):
        return self.sparsity_levels[-1]
