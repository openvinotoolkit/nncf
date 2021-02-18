"""
 Copyright (c) 2021 Intel Corporation
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

from beta.nncf.api.compression import CompressionLoss
from beta.nncf.tensorflow.layers.wrapper import NNCFWrapper
from beta.nncf.tensorflow.sparsity.rb.operation import OP_NAME


class SparseLoss(CompressionLoss):
    def __init__(self, sparse_layers: [NNCFWrapper] = None, target=1.0, p=0.05):
        super().__init__()
        self._sparse_layers = sparse_layers
        self.target = target
        self.p = p
        self.disabled = False
        self.current_sparsity = tf.constant(0.)
        self.mean_sparse_prob = tf.constant(0.)

    def set_layers(self, sparse_layers: [NNCFWrapper]):
        self._sparse_layers = sparse_layers

    def disable(self):
        if not self.disabled:
            self.disabled = True

            for sparse_layer in self._sparse_layers:
                op = sparse_layer.get_op_by_name(OP_NAME)
                op.freeze(sparse_layer.ops_weights[OP_NAME])

    def call(self, *args, **kwargs):
        if self.disabled:
            return 0

        params = tf.zeros((1, ), dtype=tf.int32)
        loss = tf.zeros((1, ))
        sparse_prob_sum = 0
        for sparse_layer in self._sparse_layers:
            if not self.disabled and not sparse_layer.get_op_by_name(OP_NAME).trainable:
                raise AssertionError(
                    "Invalid state of SparseLoss and SparsifiedWeight: mask is frozen for enabled loss")
            if sparse_layer.trainable:
                sw_loss, params_layer, mask = self._get_params_from_sparse_layer(sparse_layer)
                params = params + params_layer
                loss = loss + sw_loss
                sparse_prob_sum += tf.math.reduce_sum(tf.math.sigmoid(mask))

        params = tf.cast(params, dtype=tf.float32)
        self.mean_sparse_prob = (sparse_prob_sum / params)
        self.current_sparsity = 1 - loss / params
        return tf.reshape(tf.math.pow(((loss / params - self.target) / self.p), 2), shape=[])

    @property
    def target_sparsity_rate(self):
        rate = 1 - self.target
        if rate < 0 or rate > 1:
            raise IndexError("Target is not within range(0,1)")
        return rate

    @staticmethod
    def _get_params_from_sparse_layer(sparse_layer):
        op = sparse_layer.get_op_by_name(OP_NAME)
        mask = sparse_layer.ops_weights[OP_NAME]['mask']
        return op.loss(mask), tf.size(mask), mask

    def statistics(self, quickly_collected_only=False):
        return {'mean_sparse_prob': 1 - self.mean_sparse_prob}

    def set_target_sparsity_loss(self, sparsity_level):
        self.target = 1 - sparsity_level

    def as_metric(self, name='rb_sparse_loss', **kwargs):
        class SparseLossMetric(tf.keras.metrics.Metric):
            def __init__(self, loss):
                super().__init__(name=name, **kwargs)
                self.loss = loss
                self.val = self.add_weight(name='val', initializer='zeros',
                                           shape=[1])

            def update_state(self, *args, **kwargs):
                self.val.assign(self.loss)

            def result(self):
                return self.val

        return SparseLossMetric(self)



class SparseLossForPerLayerSparsity(SparseLoss):
    def __init__(self, sparse_layers: [NNCFWrapper] = None, target=1.0, p=0.05):
        super().__init__(sparse_layers, target, p)
        self.per_layer_target = {}
        for sparse_layer in self._sparse_layers:
            self.per_layer_target[sparse_layer] = self.target

    def call(self):
        if self.disabled:
            return 0

        params = tf.zeros((1, ), dtype=tf.int32)
        sparse_prob_sum = tf.zeros((1, ))
        sparse_layers_loss = 0
        for sparse_layer in self._sparse_layers:
            if not self.disabled: #and not sparse_layer.sparsify:
                raise AssertionError(
                    "Invalid state of SparseLoss and SparsifiedWeight: mask is frozen for enabled loss")
            # TODO: find out how it works
            if True:#sparse_layer.sparsify:
                sw_loss, params_layer, mask = self._get_params_from_sparse_layer(sparse_layer)
                params = params + params_layer
                sparse_layers_loss += tf.math.abs(sw_loss / params_layer - self.per_layer_target[sparse_layer])
                sparse_prob_sum += tf.math.reduce_sum(tf.math.sigmoid(sparse_layer.mask))

        params = tf.cast(params, dtype=tf.float32)
        self.mean_sparse_prob = (sparse_prob_sum / params)
        return tf.reshape(tf.math.pow((sparse_layers_loss / self.p), 2), shape=[])

    def set_target_sparsity_loss(self, target, sparse_layer):
        self.per_layer_target[sparse_layer] = 1 - target
