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

import numpy as np

from nncf.api.compression import CompressionScheduler
from nncf.common.utils.registry import Registry

PRUNING_SCHEDULERS = Registry("pruning_schedulers")


class PruningScheduler(CompressionScheduler):
    def __init__(self, pruning_algo, params: dict = None):
        super().__init__()
        if params is None:
            self._params = dict()
        else:
            self._params = params

        self.algo = pruning_algo
        self.initial_pruning_rate = self.algo.pruning_init
        self.pruning_target = self._params.get('pruning_target', 0.5)
        self.pruning_steps = self._params.get('pruning_steps', 100)
        self.pruning_freeze_epoch = self.pruning_steps

    def _calc_density_level(self):
        return 1 - self.current_pruning_level

    def _maybe_freeze(self):
        if self.current_epoch + 1 >= self.pruning_freeze_epoch:
            self.algo.freeze()

    @property
    def current_pruning_level(self):
        raise NotImplementedError

    @property
    def target_pruning_level(self):
        return self.pruning_target

    def load_state(self, state):
        super().load_state(state)
        self._maybe_freeze()


@PRUNING_SCHEDULERS.register("polynomial")
class PolynomialSparseScheduler(PruningScheduler):
    def __init__(self, pruning_algo, params=None):
        super().__init__(pruning_algo, params)
        self.power = self._params.get('power', 0.9)
        self.concave = self._params.get('concave', False)
        self._update_per_optimizer_step = self._params.get('update_per_optimizer_step', False)
        if self._update_per_optimizer_step:
            self._steps_per_epoch = self._params.get('steps_per_epoch')
            if self._steps_per_epoch is None:
                raise RuntimeError('steps_per_epoch must be specified when update_per_optimizer_step is set true')

        self.algo.set_pruning_rate(self.current_pruning_level)

    def step(self, last=None):
        super().step(last)
        if self._update_per_optimizer_step:
            self.algo.set_pruning_rate(self.current_pruning_level)
            self._maybe_freeze()

    def epoch_step(self, epoch=None):
        super().epoch_step(epoch)
        if not self._update_per_optimizer_step:
            self.algo.set_pruning_rate(self.current_pruning_level)
            self._maybe_freeze()

    def _maybe_freeze(self):
        if self._update_per_optimizer_step:
            if (self.current_epoch + 1 == self.pruning_freeze_epoch
                    and self._last_local_step + 1 == self._steps_per_epoch) \
                    or self.current_epoch >= self.pruning_freeze_epoch:
                self.algo.freeze()
        else:
            super()._maybe_freeze()

    @property
    def current_pruning_level(self):
        if self.current_epoch == self.current_step == -1:
            return self.initial_pruning_rate

        sparsity_target_epoch_index = self.pruning_steps - 1
        if self._update_per_optimizer_step:
            sparsity_target_epoch_frac = sparsity_target_epoch_index \
                                         + (self._steps_per_epoch - 1) / self._steps_per_epoch
            fractional_epoch = self.current_epoch + self._last_local_step / self._steps_per_epoch
            progress = (min(sparsity_target_epoch_frac, fractional_epoch) / sparsity_target_epoch_frac)
        elif sparsity_target_epoch_index == 0:
            progress = 1
        else:
            progress = (min(sparsity_target_epoch_index, self.current_epoch) / sparsity_target_epoch_index)

        if self.concave:
            current_sparsity = self.initial_pruning_rate + (self.pruning_target - self.initial_pruning_rate) * (
                progress ** self.power)
        else:
            current_sparsity = self.pruning_target - (self.pruning_target - self.initial_pruning_rate) * (
                (1 - progress) ** self.power)
        return current_sparsity

    @property
    def _last_local_step(self):
        return -1 if self.current_step == -1 \
            else self.current_step % self._steps_per_epoch


@PRUNING_SCHEDULERS.register("exponential")
class ExponentialSparsityScheduler(PruningScheduler):
    def __init__(self, pruning_algo, params=None):
        super().__init__(pruning_algo, params)
        self.a, self.k = self._init_exp(self.initial_pruning_rate, self.pruning_target,
                                        pruning_steps=self.pruning_steps - 1)
        self.algo.set_pruning_rate(self.initial_pruning_rate)

    def epoch_step(self, epoch=None):
        super().epoch_step(epoch)
        self.algo.set_pruning_rate(self.current_pruning_level)
        self._maybe_freeze()

    @property
    def current_pruning_level(self):
        if self.current_epoch == -1:
            return self.initial_pruning_rate
        curr_sparsity = 1 - self.a * np.exp(-self.k * self.current_epoch)
        return min(curr_sparsity, self.pruning_target)

    @staticmethod
    def _init_exp(initial_pruning_rate, max_pruning_rate, pruning_steps=20):
        p1 = (0, 1 - initial_pruning_rate)
        p2 = (pruning_steps, 1 - max_pruning_rate)
        if p1[0] - p2[0] == 0:
            k = 0
            a = p2[1]
        else:
            k = np.log(p2[1] / p1[1]) / (p1[0] - p2[0])
            a = p1[1] / np.exp(-k * p1[0])
        return a, k
