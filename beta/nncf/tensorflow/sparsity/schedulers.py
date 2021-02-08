"""
 Copyright (c) 2020 Intel Corporation
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

from bisect import bisect_right
import numpy as np

from beta.nncf.api.compression import CompressionScheduler
from nncf.common.utils.registry import Registry

SPARSITY_SCHEDULERS = Registry("sparsity_schedulers")


class SparsityScheduler(CompressionScheduler):
    def __init__(self, sparsity_algo, params: dict = None):
        super().__init__()
        if params is None:
            self._params = dict()
        else:
            self._params = params

        self.algo = sparsity_algo
        self.initial_sparsity = self._params.get('sparsity_init', 0)
        self.sparsity_target = self._params.get('sparsity_target', 0.5)
        self.sparsity_target_epoch = self._params.get('sparsity_target_epoch', 90)
        self.sparsity_freeze_epoch = self._params.get('sparsity_freeze_epoch', 100)

    def _calc_density_level(self):
        return 1 - self.current_sparsity_level

    def _maybe_freeze(self):
        if self.last_epoch + 1 >= self.sparsity_freeze_epoch:
            self.algo.freeze()

    @property
    def current_sparsity_level(self):
        raise NotImplementedError

    @property
    def target_sparsity_level(self):
        return self.sparsity_target


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
                raise RuntimeError('steps_per_epoch must be specified when update_per_optimizer_step is set true')

        self.algo.set_sparsity_level(self.current_sparsity_level)

    def load_state(self, initial_step, steps_per_epoch):
        if steps_per_epoch != self._steps_per_epoch:
            raise RuntimeError('Parameter steps_per_epoch {} doesn\'t equal to the one '
                               'provided in configuration file {}'.format(steps_per_epoch, self._steps_per_epoch))
        super().load_state(initial_step, steps_per_epoch)
        self._maybe_freeze()

    def step(self, last=None):
        super().step(last)
        if self._update_per_optimizer_step:
            self.algo.set_sparsity_level(self.current_sparsity_level)
            self._maybe_freeze()

    def epoch_step(self, epoch=None):
        super().epoch_step(epoch)
        if not self._update_per_optimizer_step:
            self.algo.set_sparsity_level(self.current_sparsity_level)
            self._maybe_freeze()

    def _maybe_freeze(self):
        if self._update_per_optimizer_step:
            if (self.last_epoch + 1 == self.sparsity_freeze_epoch
                    and self._last_local_step + 1 == self._steps_per_epoch) \
                    or self.last_epoch >= self.sparsity_freeze_epoch:
                self.algo.freeze()
        else:
            super()._maybe_freeze()

    @property
    def current_sparsity_level(self):
        if self.last_epoch == self.last_step == -1:
            return self.initial_sparsity

        sparsity_target_epoch_index = self.sparsity_target_epoch - 1
        if self._update_per_optimizer_step:
            sparsity_target_epoch_frac = sparsity_target_epoch_index \
                                         + (self._steps_per_epoch - 1) / self._steps_per_epoch
            fractional_epoch = self.last_epoch + self._last_local_step / self._steps_per_epoch
            progress = (min(sparsity_target_epoch_frac, fractional_epoch) / sparsity_target_epoch_frac)
        elif sparsity_target_epoch_index == 0:
            progress = 1
        else:
            progress = (min(sparsity_target_epoch_index, self.last_epoch) / sparsity_target_epoch_index)

        if self.concave:
            current_sparsity = self.initial_sparsity + (self.sparsity_target - self.initial_sparsity) * (
                progress ** self.power)
        else:
            current_sparsity = self.sparsity_target - (self.sparsity_target - self.initial_sparsity) * (
                (1 - progress) ** self.power)
        return current_sparsity

    @property
    def _last_local_step(self):
        return -1 if self.last_step == -1 \
            else self.last_step % self._steps_per_epoch


@SPARSITY_SCHEDULERS.register("exponential")
class ExponentialSparsityScheduler(SparsityScheduler):
    def __init__(self, sparsity_algo, params=None):
        super().__init__(sparsity_algo, params)
        self.a, self.k = self._init_exp(self.initial_sparsity, self.sparsity_target,
                                        sparsity_steps=self.sparsity_target_epoch - 1)
        self.algo.set_sparsity_level(self.initial_sparsity)

    def load_state(self, last_epoch, last_step):
        super().load_state(last_epoch, last_step)
        self._maybe_freeze()

    def epoch_step(self, epoch=None):
        super().epoch_step(epoch)
        self.algo.set_sparsity_level(self.current_sparsity_level)
        self._maybe_freeze()

    @property
    def current_sparsity_level(self):
        if self.last_epoch == -1:
            return self.initial_sparsity
        curr_sparsity = 1 - self.a * np.exp(-self.k * self.last_epoch)
        return min(curr_sparsity, self.sparsity_target)

    @staticmethod
    def _init_exp(initial_sparsity, max_sparsity, sparsity_steps=20):
        p1 = (0, 1 - initial_sparsity)
        p2 = (sparsity_steps, 1 - max_sparsity)
        if p1[0] - p2[0] == 0:
            k = 0
            a = p2[1]
        else:
            k = np.log(p2[1] / p1[1]) / (p1[0] - p2[0])
            a = p1[1] / np.exp(-k * p1[0])
        return a, k


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
        self.algo.set_sparsity_level(self.current_sparsity_level)

    def load_state(self, last_epoch, last_step):
        super().load_state(last_epoch, last_step)
        self._maybe_freeze()

    def epoch_step(self, last=None):
        super().epoch_step(last)
        ind = bisect_right(self.steps, self.last_epoch)
        if ind != self.prev_ind:
            self.sparsity_level = self.sparsity_levels[ind]
            self.prev_ind = ind
        self.algo.set_sparsity_level(self.current_sparsity_level)
        self._maybe_freeze()

    @property
    def current_sparsity_level(self):
        return self.sparsity_level

    @property
    def target_sparsity_level(self):
        return self.sparsity_levels[-1]
