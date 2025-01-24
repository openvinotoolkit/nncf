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


def build_optimizer(config, scheduler):
    optimizer_config = config.get("optimizer", {})

    optimizer_type = optimizer_config.get("type", "adam").lower()
    optimizer_params = optimizer_config.get("optimizer_params", {})

    logger.info("Building %s optimizer with params %s", optimizer_type, optimizer_params)

    if optimizer_type in ["sgd", "momentum"]:
        printable_names = {"sgd": "SGD", "momentum": "momentum"}
        logger.info("Using %s optimizer", printable_names[optimizer_type])

        default_momentum_value = 0.9 if optimizer_type == "momentum" else 0.0
        momentum = optimizer_params.get("momentum", default_momentum_value)
        nesterov = optimizer_params.get("nesterov", False)
        weight_decay = optimizer_config.get("weight_decay", None)
        common_params = {"learning_rate": scheduler, "nesterov": nesterov, "momentum": momentum}
        if weight_decay:
            optimizer = tf.keras.optimizers.SGD(**common_params, weight_decay=weight_decay)
        else:
            optimizer = tf.keras.optimizers.SGD(**common_params)
    elif optimizer_type == "rmsprop":
        logger.info("Using RMSProp optimizer")
        rho = optimizer_params.get("rho", 0.9)
        momentum = optimizer_params.get("momentum", 0.9)
        epsilon = optimizer_params.get("epsilon", 1e-07)
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=scheduler, rho=rho, momentum=momentum, epsilon=epsilon)
    elif optimizer_type in ["adam", "adamw"]:
        printable_names = {"adam": "Adam", "adamw": "AdamW"}
        logger.info("Using %s optimizer", printable_names[optimizer_type])

        beta_1, beta_2 = optimizer_params.get("betas", [0.9, 0.999])
        epsilon = optimizer_params.get("eps", 1e-07)
        amsgrad = optimizer_params.get("amsgrad", False)
        w_decay_defaul_value = 0.01 if optimizer_type == "adamw" else None
        weight_decay = optimizer_config.get("weight_decay", w_decay_defaul_value)
        common_params = {
            "learning_rate": scheduler,
            "beta_1": beta_1,
            "beta_2": beta_2,
            "epsilon": epsilon,
            "amsgrad": amsgrad,
        }
        if weight_decay:
            optimizer = tf.keras.optimizers.AdamW(**common_params, weight_decay=weight_decay)
        else:
            optimizer = tf.keras.optimizers.Adam(**common_params)
    else:
        raise ValueError("Unknown optimizer %s" % optimizer_type)

    return optimizer
