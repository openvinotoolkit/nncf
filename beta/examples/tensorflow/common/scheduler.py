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

import tensorflow as tf

from examples.tensorflow.common.logger import logger


class StepLearningRateWithLinearWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Class to generate learning rate tensor"""

    def __init__(self, total_steps, params):
        """Creates the step learning rate tensor with linear warmup"""
        super().__init__()
        self._total_steps = total_steps
        self._params = params

    def __call__(self, global_step):
        warmup_lr = self._params.warmup_learning_rate
        warmup_steps = self._params.warmup_steps
        init_lr = self._params.init_learning_rate
        lr_levels = self._params.learning_rate_levels
        lr_steps = self._params.learning_rate_steps

        linear_warmup = (warmup_lr + tf.cast(global_step, tf.float32) / warmup_steps * (init_lr - warmup_lr))
        learning_rate = tf.where(global_step < warmup_steps, linear_warmup, init_lr)

        for next_learning_rate, start_step in zip(lr_levels, lr_steps):
            learning_rate = tf.where(global_step >= start_step, next_learning_rate, learning_rate)

        return learning_rate

    def get_config(self):
        return {'_params': self._params.as_dict()}


def build_scheduler(config, epoch_size, batch_size, steps):
    optimizer_config = config.get('optimizer', {})
    schedule_type = optimizer_config.get('schedule_type', 'exponential').lower()
    schedule_params = optimizer_config.get("schedule_params", {})

    if schedule_type == 'exponential':
        decay_rate = schedule_params.get('decay_rate', None)
        if decay_rate is None:
            raise ValueError('decay_rate parameter must be specified '
                             'for the exponential scheduler')

        initial_lr = schedule_params.get('initial_lr', None)
        if initial_lr is None:
            raise ValueError('initial_lr parameter must be specified '
                             'for the exponential scheduler')

        decay_epochs = schedule_params.get('decay_epochs', None)
        decay_steps = decay_epochs * steps if decay_epochs is not None else 0

        logger.info('Using exponential learning rate with: '
                    'initial_learning_rate: {initial_lr}, decay_steps: {decay_steps}, '
                    'decay_rate: {decay_rate}'.format(initial_lr=initial_lr,
                                                      decay_steps=decay_steps,
                                                      decay_rate=decay_rate))
        lr = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            decay_rate=decay_rate)
    elif schedule_type == 'piecewise_constant':
        boundaries = schedule_params.get('boundaries', None)
        if boundaries is None:
            raise ValueError('boundaries parameter must be specified '
                             'for the piecewise_constant scheduler')

        values = schedule_params.get('values', None)
        if values is None:
            raise ValueError('values parameter must be specified '
                             'for the piecewise_constant')

        logger.info('Using Piecewise constant decay with warmup. '
                    'Parameters: batch_size: {batch_size}, epoch_size: {epoch_size}, '
                    'boundaries: {boundaries}, values: {values}'.format(batch_size=batch_size,
                                                                        epoch_size=epoch_size,
                                                                        boundaries=boundaries,
                                                                        values=values))
        steps_per_epoch = epoch_size // batch_size
        boundaries = [steps_per_epoch * x for x in boundaries]
        lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
    elif schedule_type == 'step':
        lr = StepLearningRateWithLinearWarmup(steps, schedule_params)

    return lr
