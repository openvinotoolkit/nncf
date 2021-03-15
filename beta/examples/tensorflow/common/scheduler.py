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

from beta.examples.tensorflow.common.logger import logger


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

        linear_warmup = (warmup_lr + tf.cast(global_step, tf.float32) / warmup_steps * (init_lr - warmup_lr))
        learning_rate = tf.where(global_step < warmup_steps, linear_warmup, init_lr)

        for next_learning_rate, start_step in zip(lr_levels, lr_steps):
            learning_rate = tf.where(global_step >= start_step, next_learning_rate, learning_rate)

        return learning_rate

    def get_config(self):
        return {'params': self._params.as_dict()}


class StepLearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, init_lr, step, gamma=0.1):
        """
        Creates the step learning rate tensor
        Args:
            init_lr: Initial learning rate
            step: Period of learning rate decay
            gamma: Learning rate decay rate
        """
        super().__init__()
        self._init_lr = init_lr
        self._step = step
        self._gamma = gamma

    def __call__(self, global_step):
        return self._init_lr * self._gamma ** tf.cast(global_step // self._step, tf.float32)

    def get_config(self):
        return {'init_lr': self._init_lr,
                'step': self._step,
                'gamma': self._gamma}


class MultiStepLearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, init_lr, steps, gamma=0.1):
        """
        Creates the multistep learning rate tensor
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
        return {'init_lr': self._init_lr,
                'steps': self._steps,
                'gamma': self._gamma}


def build_scheduler(config, steps_per_epoch):
    optimizer_config = config.get('optimizer', {})
    schedule_type = optimizer_config.get('schedule_type', 'step').lower()
    schedule_params = optimizer_config.get('scheduler_params', {})
    decay_rate = schedule_params.get('gamma', optimizer_config.get('gamma', 0.1))
    initial_lr = schedule_params.get('base_lr', optimizer_config.get('base_lr', None))

    if schedule_type == 'exponential':
        if initial_lr is None:
            raise ValueError('`base_lr` parameter must be specified '
                             'for the exponential scheduler')

        decay_epochs = schedule_params.get('decay_epochs', optimizer_config.get('decay_epochs', None))
        decay_steps = decay_epochs * steps_per_epoch if decay_epochs is not None else steps_per_epoch

        logger.info('Using exponential learning rate with: '
                    'base_lr: %f, decay_steps: %d, '
                    'decay_rate: %f', initial_lr, decay_steps, decay_rate)
        lr = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            decay_rate=decay_rate)

    elif schedule_type == 'piecewise_constant':
        boundaries = schedule_params.get('boundaries', None)
        if boundaries is None:
            raise ValueError('`boundaries` parameter must be specified '
                             'for the `piecewise_constant` scheduler')

        values = schedule_params.get('values', None)
        if values is None:
            raise ValueError('`values` parameter must be specified '
                             'for the `piecewise_constant` scheduler')

        logger.info('Using Piecewise constant decay with warmup. '
                    'Parameters: boundaries: %s, values: %s', boundaries, values)
        boundaries = [steps_per_epoch * x for x in boundaries]
        lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)

    elif schedule_type == 'multistep':
        logger.info('Using MultiStep learning rate.')
        if initial_lr is None:
            raise ValueError('`base_lr` parameter must be specified '
                             'for the `multistep` scheduler')
        steps = schedule_params.get('steps', optimizer_config.get('steps', None))
        if steps is None:
            raise ValueError('`steps` parameter must be specified '
                             'for the `multistep` scheduler')
        steps = [steps_per_epoch * x for x in steps]
        lr = MultiStepLearningRate(initial_lr, steps, gamma=0.1)

    elif schedule_type == 'step':
        logger.info('Using Step learning rate.')
        if initial_lr is None:
            raise ValueError('`base_lr` parameter must be specified '
                             'for the `step` scheduler')
        step = schedule_params.get('step', optimizer_config.get('step', None))
        if initial_lr is None:
            raise ValueError('`step` parameter must be specified '
                             'for the `step` scheduler')
        step = step * steps_per_epoch
        lr = StepLearningRate(initial_lr, step, gamma=0.1)
    elif schedule_type == 'step_warmup':
        lr = StepLearningRateWithLinearWarmup(schedule_params)
    else:
        raise KeyError(f'Unknown learning rate scheduler type: {schedule_type}')

    return lr
