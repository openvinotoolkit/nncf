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
from shutil import rmtree
from shutil import copytree

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.optimizers import schedules

from nncf.common.accuracy_aware_training.runner import BaseAccuracyAwareTrainingRunner
from nncf.common.accuracy_aware_training.runner import BaseAdaptiveCompressionLevelTrainingRunner


class TFAccuracyAwareTrainingRunner(BaseAccuracyAwareTrainingRunner):
    """
    The Training Runner implementation for TensorFlow training code.
    """

    checkpoint_path_extension = ''

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
                raise NotImplementedError(f"Scheduler {type(scheduler)} is not supported yet")

        self.training_epoch_count = 0
        self.best_val_metric_value = 0
        self.current_val_metric_value = 0

    def update_learning_rate(self):
        if self.update_learning_rate_fn is not None:
            self.update_learning_rate_fn(self.lr_scheduler,
                                         self.training_epoch_count,
                                         self.current_val_metric_value)

    def _save_checkpoint(self, model, checkpoint_path, compression_controller=None):
        # for tensorflow checkpoints are saved in multiple shards, hence we save it in a separate subdirectory
        new_checkpoint_path = osp.join(checkpoint_path, "checkpoint")
        model.save_weights(new_checkpoint_path)

    def _load_checkpoint(self, model, checkpoint_path):
        if self.load_checkpoint_fn is not None:
            self.load_checkpoint_fn(model, checkpoint_path)
        else:
            # checkpoint is in the subdirectory
            new_checkpoint_path = osp.join(checkpoint_path, "checkpoint")
            model.load_weights(new_checkpoint_path)

    def _copy_checkpoint(self, source_path, destination_path):
        if osp.exists(destination_path):
            rmtree(destination_path)
        copytree(source_path, destination_path)


class TFAdaptiveCompressionLevelTrainingRunner(BaseAdaptiveCompressionLevelTrainingRunner,
                                               TFAccuracyAwareTrainingRunner):
    def __init__(self, accuracy_aware_training_params, verbose=True, dump_checkpoints=True, lr_updates_needed=True,
                 minimal_compression_rate=0.05, maximal_compression_rate=0.95):
        super().__init__(accuracy_aware_training_params, verbose, dump_checkpoints, lr_updates_needed,
                         minimal_compression_rate=minimal_compression_rate,
                         maximal_compression_rate=maximal_compression_rate)
