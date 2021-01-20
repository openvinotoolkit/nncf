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
import tensorflow_addons as tfa

from beta.examples.tensorflow.common.logger import logger


def build_optimizer(config, scheduler):
    optimizer_config = config.get('optimizer', {})

    optimizer_type = optimizer_config.get('type', 'adam').lower()
    optimizer_params = optimizer_config.get("optimizer_params", {})

    logger.info('Building %s optimizer with params %s', optimizer_type, optimizer_params)

    if optimizer_type == 'sgd':
        logger.info('Using SGD optimizer')
        nesterov = optimizer_params.get('nesterov', False)
        optimizer = tf.keras.optimizers.SGD(learning_rate=scheduler,
                                            nesterov=nesterov)
    elif optimizer_type == 'momentum':
        logger.info('Using momentum optimizer')
        nesterov = optimizer_params.get('nesterov', False)
        momentum = optimizer_params.get('momentum', 0.9)
        optimizer = tf.keras.optimizers.SGD(learning_rate=scheduler,
                                            momentum=momentum,
                                            nesterov=nesterov)
    elif optimizer_type == 'rmsprop':
        logger.info('Using RMSProp')
        rho = optimizer_params.get('rho', 0.9)
        momentum = optimizer_params.get('momentum', 0.9)
        epsilon = optimizer_params.get('epsilon', 1e-07)
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=scheduler,
                                                rho=rho,
                                                momentum=momentum,
                                                epsilon=epsilon)
    elif optimizer_type == 'adam':
        logger.info('Using Adam')
        beta_1 = optimizer_params.get('beta_1', 0.9)
        beta_2 = optimizer_params.get('beta_2', 0.999)
        epsilon = optimizer_params.get('epsilon', 1e-07)
        optimizer = tf.keras.optimizers.Adam(learning_rate=scheduler,
                                             beta_1=beta_1,
                                             beta_2=beta_2,
                                             epsilon=epsilon)
    elif optimizer_type == 'adamw':
        logger.info('Using AdamW')
        weight_decay = optimizer_params.get('weight_decay', 0.01)
        beta_1 = optimizer_params.get('beta_1', 0.9)
        beta_2 = optimizer_params.get('beta_2', 0.999)
        epsilon = optimizer_params.get('epsilon', 1e-07)
        optimizer = tfa.optimizers.AdamW(weight_decay=weight_decay,
                                         learning_rate=scheduler,
                                         beta_1=beta_1,
                                         beta_2=beta_2,
                                         epsilon=epsilon)
    else:
        raise ValueError('Unknown optimizer %s' % optimizer_type)

    moving_average_decay = optimizer_params.get('moving_average_decay', 0.)
    if moving_average_decay > 0.:
        logger.info('Including moving average decay.')
        optimizer = tfa.optimizers.MovingAverage(
            optimizer,
            average_decay=moving_average_decay,
            num_updates=None)
    if optimizer_params.get('lookahead', None):
        logger.info('Using lookahead optimizer.')
        optimizer = tfa.optimizers.Lookahead(optimizer)

    return optimizer
