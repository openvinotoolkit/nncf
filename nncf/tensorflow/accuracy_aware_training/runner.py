"""
 Copyright (c) 2022 Intel Corporation
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

import os.path as osp

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.optimizers import schedules  # pylint:disable=no-name-in-module

from nncf.common.accuracy_aware_training.runner import BaseAccuracyAwareTrainingRunner
from nncf.common.accuracy_aware_training.runner import BaseAdaptiveCompressionLevelTrainingRunner
from nncf.common.schedulers import StubCompressionScheduler
from nncf.common.logging import nncf_logger


class TFAccuracyAwareTrainingRunner(BaseAccuracyAwareTrainingRunner):
    """
    The Training Runner implementation for TensorFlow training code.
    """

    def retrieve_uncompressed_model_accuracy(self, model):
        if not hasattr(model, 'original_model_accuracy'):
            raise RuntimeError('Original model does not contain the pre-calculated reference metric value')
        self.uncompressed_model_accuracy = model.original_model_accuracy

    def validate(self, model):
        self.current_val_metric_value = self._validate_fn(model, epoch=self.cumulative_epoch_count)
        is_best = (not self.is_higher_metric_better) != (self.current_val_metric_value > self.best_val_metric_value)
        if is_best:
            self.best_val_metric_value = self.current_val_metric_value
        return self.current_val_metric_value

    def reset_training(self):
        self.configure_optimizers()

        if isinstance(self.optimizer, tfa.optimizers.MultiOptimizer):
            optimizers = [optimizer_spec.optimizer for optimizer_spec in self.optimizer.optimizer_specs]
        else:
            optimizers = self.optimizer if isinstance(self.optimizer, (tuple, list)) else [self.optimizer]

        for optimizer in optimizers:
            scheduler = optimizer.learning_rate
            if isinstance(scheduler, tf.Variable):
                scheduler = scheduler * self.base_lr_reduction_factor_during_search
                optimizer.learning_rate = scheduler
                optimizer.lr = scheduler
            elif isinstance(scheduler, (schedules.CosineDecay, schedules.ExponentialDecay)):
                scheduler.initial_learning_rate *= self.base_lr_reduction_factor_during_search
            elif isinstance(scheduler, schedules.PiecewiseConstantDecay):
                scheduler.values = [lr * self.base_lr_reduction_factor_during_search for lr in scheduler.values]
            else:
                nncf_logger.warning(f"Learning rate scheduler {scheduler} is not supported yet."
                                    f"Won't change the learning rate.")

        self.training_epoch_count = 0
        self.best_val_metric_value = 0
        self.current_val_metric_value = 0

    def update_learning_rate(self):
        if self._update_learning_rate_fn is not None:
            self._update_learning_rate_fn(self.lr_scheduler,
                                          self.training_epoch_count,
                                          self.current_val_metric_value,
                                          self.current_loss)

    def _save_checkpoint(self, model, compression_controller, checkpoint_path):
        model.save_weights(checkpoint_path)

    def _load_checkpoint(self, model, checkpoint_path):
        if self._load_checkpoint_fn is not None:
            self._load_checkpoint_fn(model, checkpoint_path)
        else:
            model.load_weights(checkpoint_path)

    def _make_checkpoint_path(self, is_best, compression_rate=None):
        extension = '.pt'
        return osp.join(self._checkpoint_save_dir, f'acc_aware_checkpoint_{"best" if is_best else "last"}{extension}')

    def add_tensorboard_scalar(self, key, data, step):
        if self.verbose and self._tensorboard_writer is not None:
            self._tensorboard_writer.add_scalar(key, data, step)

    def add_tensorboard_image(self, key, data, step):
        if self.verbose and self._tensorboard_writer is not None:
            self._tensorboard_writer.add_image(key, data, step)


class TFAdaptiveCompressionLevelTrainingRunner(BaseAdaptiveCompressionLevelTrainingRunner,
                                               TFAccuracyAwareTrainingRunner):
    def __init__(self, accuracy_aware_training_params, verbose=True, dump_checkpoints=True, lr_updates_needed=True,
                 minimal_compression_rate=0.05, maximal_compression_rate=0.95):
        super().__init__(accuracy_aware_training_params, verbose, dump_checkpoints, lr_updates_needed,
                         minimal_compression_rate=minimal_compression_rate,
                         maximal_compression_rate=maximal_compression_rate)

    def dump_checkpoint(self, model, compression_controller):
        # a workaround because for tensorflow backend disabling compression scheduler does not work properly
        is_best_checkpoint = (self.best_val_metric_value == self.current_val_metric_value and
                              isinstance(compression_controller.scheduler, StubCompressionScheduler))
        if not self.dump_checkpoints and not is_best_checkpoint:
            return

        if self._dump_checkpoint_fn is not None:
            checkpoint_path = self._dump_checkpoint_fn(model, compression_controller, self, self._checkpoint_save_dir)
        else:
            checkpoint_path = self._make_checkpoint_path(is_best=False)
            self._save_checkpoint(model, compression_controller, checkpoint_path)
        nncf_logger.info(f"Saved the checkpoint to {checkpoint_path}")

        if is_best_checkpoint:
            self._save_best_checkpoint(model, compression_controller)

    def _make_checkpoint_path(self, is_best, compression_rate=None):
        extension = '.pt'
        base_path = osp.join(self._checkpoint_save_dir, 'acc_aware_checkpoint')
        if is_best:
            if compression_rate is None:
                raise ValueError('Compression rate cannot be None')
            return f'{base_path}_best_{compression_rate:.3f}{extension}'
        return f'{base_path}_last{extension}'
