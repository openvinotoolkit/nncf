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

import tensorflow as tf

from examples.tensorflow.common.logger import logger


class StepLearningRateWithLinearWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Class to generate learning rate tensor"""

    def __init__(self, params):
        """Creates the step learning rate tensor with linear warmup"""
        super().__init__()
        self._params = params

    def __call__(self, global_step):
        warmup_lr = self._params.warmup_learning_rate
        warmup_steps = self._params.warmup_steps
        init_lr = self._params.init_learning_rate
        lr_levels = self._params.learning_rate_levels
        lr_steps = self._params.learning_rate_steps

        linear_warmup = warmup_lr + tf.cast(global_step, tf.float32) / warmup_steps * (init_lr - warmup_lr)
        learning_rate = tf.where(global_step < warmup_steps, linear_warmup, init_lr)

        for next_learning_rate, start_step in zip(lr_levels, lr_steps):
            learning_rate = tf.where(global_step >= start_step, next_learning_rate, learning_rate)

        return learning_rate

    def get_config(self):
        return {"params": self._params.as_dict()}


class MultiStepLearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, init_lr, steps, gamma=0.1):
        """
        Creates the multistep learning rate schedule.
        Decays learning rate by `gamma` once `global_step` reaches
        one of the milestones in the `steps` list.
        For init_lr = 0.01, steps = [10, 15] and gamma = 0.1
        lr = 0.01    if global_step < 10
        lr = 0.001   if 10 <= global_step < 15
        lr = 0.0001  if global_step >= 15
        Args:
            init_lr: Initial learning rate
            steps: List of step indices
            gamma: Learning rate decay rate
        """
        super().__init__()
        self._init_lr = init_lr
        self._steps = sorted(steps)
        self._gamma = gamma
        self._lr_values = [init_lr * self._gamma ** (i + 1) for i in range(len(self._steps))]

    def __call__(self, global_step):
        learning_rate = self._init_lr
        for next_learning_rate, start_step in zip(self._lr_values, self._steps):
            learning_rate = tf.where(global_step >= start_step, next_learning_rate, learning_rate)
        return learning_rate

    def get_config(self):
        return {"init_lr": self._init_lr, "steps": self._steps, "gamma": self._gamma}


def schedule_base_lr_check(schedule_type, base_lr):
    schedules_with_base_lr = ["exponential", "multistep", "step", "cosine"]
    if schedule_type in schedules_with_base_lr and base_lr is None:
        raise ValueError("`base_lr` parameter must be specified for the %s scheduler" % schedule_type)


def build_scheduler(config, steps_per_epoch):
    optimizer_config = config.get("optimizer", {})
    schedule_type = optimizer_config.get("schedule_type", "step").lower()
    schedule_params = optimizer_config.get("schedule_params", {})
    gamma = schedule_params.get("gamma", optimizer_config.get("gamma", 0.1))
    base_lr = schedule_params.get("base_lr", optimizer_config.get("base_lr", 0.001))

    schedule_base_lr_check(schedule_type, base_lr)

    if schedule_type == "exponential":
        step = schedule_params.get("step", optimizer_config.get("step", 1))
        decay_steps = step * steps_per_epoch

        logger.info(
            "Using exponential learning rate with: initial lr: %f, decay steps: %d, decay rate: %f",
            base_lr,
            decay_steps,
            gamma,
        )
        lr = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=base_lr, decay_steps=decay_steps, decay_rate=gamma
        )

    elif schedule_type == "piecewise_constant":
        boundaries = schedule_params.get("boundaries", optimizer_config.get("boundaries", None))
        if boundaries is None:
            raise ValueError("`boundaries` parameter must be specified for the `piecewise_constant` scheduler")

        values = schedule_params.get("values", optimizer_config.get("values", None))
        if values is None:
            raise ValueError("`values` parameter must be specified for the `piecewise_constant` scheduler")

        logger.info(
            "Using Piecewise constant decay with warmup. Parameters: boundaries: %s, values: %s", boundaries, values
        )
        boundaries = [steps_per_epoch * x for x in boundaries]
        lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)

    elif schedule_type == "multistep":
        logger.info("Using MultiStep learning rate.")
        steps = schedule_params.get("steps", optimizer_config.get("steps", None))
        if steps is None:
            raise ValueError("`steps` parameter must be specified for the `multistep` scheduler")
        steps = [steps_per_epoch * x for x in steps]
        lr = MultiStepLearningRate(base_lr, steps, gamma=gamma)

    elif schedule_type == "step":
        step = schedule_params.get("step", optimizer_config.get("step", 1))
        decay_steps = step * steps_per_epoch

        logger.info(
            "Using Step learning rate with: base_lr: %f, decay steps: %d, gamma: %f", base_lr, decay_steps, gamma
        )
        lr = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=base_lr, decay_steps=decay_steps, decay_rate=gamma, staircase=True
        )

    elif schedule_type == "step_warmup":
        lr = StepLearningRateWithLinearWarmup(schedule_params)

    elif schedule_type == "cosine":
        decay_steps = steps_per_epoch * config.epochs
        logger.info("Using Cosine learning rate with: base_lr: %f, decay steps: %d, ", base_lr, decay_steps)
        lr = tf.keras.experimental.CosineDecay(initial_learning_rate=base_lr, decay_steps=decay_steps)

    else:
        raise KeyError(f"Unknown learning rate scheduler type: {schedule_type}")

    return lr
