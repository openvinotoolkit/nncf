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

from .compression import CompressionLoss, CompressionScheduler,\
    CompressionAlgorithmController, CompressionAlgorithmBuilder
from ..tensorflow.graph.transformations.layout import TransformationLayout
from ..tensorflow.utils.save import save_model


class CompositeCompressionLoss(CompressionLoss):
    def __init__(self):
        super().__init__()
        self._child_losses = []

    @property
    def child_losses(self):
        return self._child_losses

    def add(self, child_loss):
        self._child_losses.append(child_loss)

    def call(self):
        result_loss = 0
        for loss in self._child_losses:
            result_loss += loss()
        return result_loss

    def statistics(self):
        stats = {}
        for loss in self._child_losses:
            stats.update(loss.statistics())
        return stats

    def get_config(self):
        config = {}
        child_loss_configs = []
        for child_loss in self.child_losses:
            child_loss_configs.append(
                tf.keras.utils.serialize_keras_object(child_loss)
            )
        config['child_losses'] = child_loss_configs
        return config

    @classmethod
    def from_config(cls, config):
        config = config.copy()

        child_loss_configs = config.pop('child_losses')

        loss = cls()
        for loss_config in child_loss_configs:
            loss.add(tf.keras.layers.deserialize(loss_config))
        return loss


class CompositeCompressionScheduler(CompressionScheduler):
    def __init__(self):
        super().__init__()
        self._child_schedulers = []

    @property
    def child_schedulers(self):
        return self._child_schedulers

    def add(self, child_scheduler):
        self._child_schedulers.append(child_scheduler)

    def step(self, last=None):
        super().step(last)
        for scheduler in self._child_schedulers:
            scheduler.step(last)

    def epoch_step(self, last=None):
        super().epoch_step(last)
        for scheduler in self._child_schedulers:
            scheduler.epoch_step(last)

    def get_config(self):
        config = {}
        child_scheduler_configs = []
        for child_scheduler in self.child_schedulers:
            child_scheduler_configs.append(
                tf.keras.utils.serialize_keras_object(child_scheduler)
            )
        config['child_schedulers'] = child_scheduler_configs
        return config

    @classmethod
    def from_config(cls, config):
        config = config.copy()

        child_scheduler_configs = config.pop('child_schedulers')

        scheduler = cls()
        for scheduler_config in child_scheduler_configs:
            scheduler.add(tf.keras.layers.deserialize(scheduler_config))
        return scheduler


class CompositeCompressionAlgorithmController(CompressionAlgorithmController):
    def __init__(self, target_model):
        super().__init__(target_model)
        self._child_ctrls = []
        self._loss = CompositeCompressionLoss()
        self._scheduler = CompositeCompressionScheduler()
        self._initializer = None

    @property
    def child_ctrls(self):
        return self._child_ctrls

    def add(self, child_ctrl):
        if child_ctrl.model is not self.model:
            raise RuntimeError("Cannot create a composite controller "
                               "from controllers belonging to different models!")

        self.child_ctrls.append(child_ctrl)
        self._loss.add(child_ctrl.loss)
        self._scheduler.add(child_ctrl.scheduler)

    def initialize(self, dataset=None, loss=None):
        for ctrl in self.child_ctrls:
            ctrl.initialize(dataset, loss)

    def statistics(self):
        stats = {}
        for ctrl in self.child_ctrls:
            stats.update(ctrl.statistics())
        return stats

    def export_model(self, save_path, save_format='frozen_graph'):
        stripped_model = self.model
        for ctrl in self.child_ctrls:
            stripped_model = ctrl.strip_model(stripped_model)
        save_model(stripped_model, save_path, save_format)


class CompositeCompressionAlgorithmBuilder(CompressionAlgorithmBuilder):
    def __init__(self, config=None):
        super().__init__(config)
        self._child_builders = []

    @property
    def child_builders(self):
        return self._child_builders

    def add(self, child_builder):
        self.child_builders.append(child_builder)

    def build_controller(self, model):
        composite_ctrl = CompositeCompressionAlgorithmController(model)
        for builder in self.child_builders:
            composite_ctrl.add(builder.build_controller(model))
        return composite_ctrl

    def get_transformation_layout(self, model):
        transformations = TransformationLayout()
        for builder in self.child_builders:
            transformations.update(builder.get_transformation_layout(model))
        return transformations
