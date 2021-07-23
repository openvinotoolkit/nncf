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

import os.path as osp

import tensorflow as tf

from nncf.common.utils.logger import logger as nncf_logger
from nncf.common.accuracy_aware_training.runner import BaseAccuracyAwareTrainingRunner
from nncf.common.utils.helpers import configure_accuracy_aware_paths


# pylint: disable=E1101
class TFBaseTrainingRunner(BaseAccuracyAwareTrainingRunner):
    """
    The Training Runner implementation for TensorFlow training code.
    """

    def initialize_training_loop_fns(self, train_epoch_fn, validate_fn, configure_optimizers_fn=None,
                                     tensorboard_writer=None, log_dir=None):
        super().initialize_training_loop_fns(train_epoch_fn, validate_fn, configure_optimizers_fn,
                                             tensorboard_writer=tensorboard_writer, log_dir=log_dir)
        self._log_dir = self._log_dir if self._log_dir is not None \
            else 'runs'
        self._log_dir = configure_accuracy_aware_paths(self._log_dir)
        self._checkpoint_save_dir = self._log_dir

    def retrieve_original_accuracy(self, model):
        if not hasattr(model, 'original_model_accuracy'):
            raise RuntimeError('Original model does not contain the pre-calculated reference metric value')
        self.uncompressed_model_accuracy = model.original_model_accuracy
        self.minimal_tolerable_accuracy = self.uncompressed_model_accuracy * (1 - 0.01 * self.maximal_accuracy_drop)

    def train_epoch(self, model, compression_controller):
        compression_controller.scheduler.epoch_step()
        # assuming that epoch number is only used for logging in train_fn:
        self._train_epoch_fn(compression_controller,
                             model,
                             epoch=self.cumulative_epoch_count)

        statistics = compression_controller.statistics()

        self.current_val_metric_value = None
        if self.validate_every_n_epochs is not None and \
                self.training_epoch_count % self.validate_every_n_epochs == 0:
            self.current_val_metric_value = self.validate(model)

        if self.verbose:
            nncf_logger.info(statistics.to_str())
        # dump best checkpoint for current target compression rate
        self.dump_checkpoint(model)

        self.training_epoch_count += 1
        self.cumulative_epoch_count += 1
        return self.current_val_metric_value

    def validate(self, model):
        val_metric_value = self._validate_fn(model, epoch=self.cumulative_epoch_count)
        is_best = (not self.is_higher_metric_better) != (val_metric_value > self.best_val_metric_value)
        if is_best:
            self.best_val_metric_value = val_metric_value

        self.add_tensorboard_scalar('val/accuracy_aware/metric_value',
                                    data=val_metric_value, step=self.cumulative_epoch_count)

        return val_metric_value

    def reset_training(self):
        self.training_epoch_count = 0
        self.best_val_metric_value = 0

    def dump_checkpoint(self, model):
        checkpoint_path = osp.join(self._checkpoint_save_dir, 'acc_aware_checkpoint_last.pb')
        model.save_weights(checkpoint_path)

        if self.best_val_metric_value == self.current_val_metric_value:
            best_path = osp.join(self._checkpoint_save_dir,
                                 'acc_aware_checkpoint_best_compression_rate_'
                                 '{comp_rate:.3f}.pth'.format(comp_rate=self.compression_rate_target))
            self._best_checkpoints[self.compression_rate_target] = best_path
            model.save_weights(best_path)

    def add_tensorboard_scalar(self, key, data, step):
        tf.summary.scalar(key, data=data, step=step)

    def load_best_checkpoint(self, model):
        # load checkpoint with highest compression rate and positive acc budget
        possible_checkpoint_rates = [comp_rate for (comp_rate, acc_budget) in self._compressed_training_history
                                     if acc_budget >= 0]
        if not possible_checkpoint_rates:
            nncf_logger.warning('Could not produce a compressed model satisfying the set accuracy '
                                'degradation criterion during training. Increasing the number of training '
                                'epochs')
        best_checkpoint_compression_rate = max(possible_checkpoint_rates)
        resuming_checkpoint_path = self._best_checkpoints[best_checkpoint_compression_rate]
        nncf_logger.info('Loading the best checkpoint found during training '
                         '{}...'.format(resuming_checkpoint_path))
        model.load_weights(resuming_checkpoint_path)

    def configure_optimizers(self):
        pass


class TFAccuracyAwareTrainingRunner(TFBaseTrainingRunner, BaseAccuracyAwareTrainingRunner):

    def update_training_history(self, compression_rate, best_metric_value):
        best_accuracy_budget = best_metric_value - self.minimal_tolerable_accuracy
        self._compressed_training_history.append((compression_rate, best_accuracy_budget))

    @property
    def compressed_training_history(self):
        return self._compressed_training_history
