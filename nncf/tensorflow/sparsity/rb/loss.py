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

from typing import Any, Dict

import tensorflow as tf

from nncf.api.compression import CompressionLoss


class SparseLoss(CompressionLoss):
    def __init__(self, target_ops, target=1.0, p=0.05):
        super().__init__()
        self._target_ops = target_ops
        self.target = tf.Variable(target, trainable=False)
        self.p = p
        self.disabled = tf.Variable(False, trainable=False)

    def disable(self):
        tf.cond(tf.cast(self.disabled, tf.bool), lambda: None, self._disable)

    def _disable(self):
        self.disabled.assign(True)

        for op, op_weights in self._target_ops:
            op.freeze(op_weights)

    def calculate(self, *args, **kwargs):
        return tf.cond(tf.cast(self.disabled, tf.bool), lambda: tf.constant(0.0), self._calculate)

    def _calculate(self):
        params = tf.constant(0)
        loss = tf.constant(0.0)
        for op, op_weights in self._target_ops:
            tf.debugging.assert_equal(
                op.get_trainable_weight(op_weights),
                tf.constant(True),
                "Invalid state of SparseLoss and SparsifiedWeight: mask is frozen for enabled loss",
            )
            mask = op.get_mask(op_weights)
            params = params + tf.size(mask)
            loss = loss + op.loss(op_weights)

        params = tf.cast(params, tf.float32)
        return tf.reshape(tf.math.pow(((loss / params - self.target) / self.p), 2), shape=[])

    @property
    def target_sparsity_rate(self):
        eager_target = tf.keras.backend.eval(self.target)
        rate = 1.0 - eager_target
        if rate < 0 or rate > 1:
            raise ValueError("Target is not within range [0, 1]")
        return rate

    def set_target_sparsity_loss(self, sparsity_level):
        self.target.assign(1 - sparsity_level)

    def load_state(self, state: Dict[str, Any]) -> None:
        self.target.assign(state["target"])
        self.disabled.assign(state["disabled"])
        self.p = state["p"]

    def get_state(self) -> Dict[str, Any]:
        return {
            "target": float(tf.keras.backend.eval(self.target)),
            "disabled": bool(tf.keras.backend.eval(tf.cast(self.disabled, tf.bool))),
            "p": self.p,
        }
