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

from nncf.common.utils.logger import logger as nncf_logger
from nncf.common.accuracy_aware_training.runner import BaseAccuracyAwareTrainingRunner
from nncf.common.accuracy_aware_training.runner import BaseAdaptiveCompressionLevelTrainingRunner
from nncf.common.utils.helpers import configure_accuracy_aware_paths


class TFAccuracyAwareTrainingRunner(BaseAccuracyAwareTrainingRunner):
    """
    The Training Runner implementation for TensorFlow training code.
    """

    def initialize_training_loop_fns(self, train_epoch_fn, validate_fn, configure_optimizers_fn=None,
                                     dump_checkpoint_fn=None, tensorboard_writer=None, log_dir=None):
        super().initialize_training_loop_fns(train_epoch_fn, validate_fn, configure_optimizers_fn, dump_checkpoint_fn,
                                             tensorboard_writer=tensorboard_writer, log_dir=log_dir)
        self._log_dir = self._log_dir if self._log_dir is not None \
            else 'runs'
        self._log_dir = configure_accuracy_aware_paths(self._log_dir)
        self._checkpoint_save_dir = self._log_dir

    def retrieve_uncompressed_model_accuracy(self, model):
        if not hasattr(model, 'original_model_accuracy'):
            raise RuntimeError('Original model does not contain the pre-calculated reference metric value')
        self.uncompressed_model_accuracy = model.original_model_accuracy

    def train_epoch(self, model, compression_controller):
        compression_controller.scheduler.epoch_step()
        # assuming that epoch number is only used for logging in train_fn:
        self._train_epoch_fn(compression_controller,
                             model,
                             epoch=self.cumulative_epoch_count)

        self.training_epoch_count += 1
        self.cumulative_epoch_count += 1

    def validate(self, model):
        self.current_val_metric_value = self._validate_fn(model, epoch=self.cumulative_epoch_count)
        is_best = (not self.is_higher_metric_better) != (self.current_val_metric_value > self.best_val_metric_value)
        if is_best:
            self.best_val_metric_value = self.current_val_metric_value

        self.add_tensorboard_scalar('val/accuracy_aware/metric_value',
                                    data=self.current_val_metric_value, step=self.cumulative_epoch_count)

        return self.current_val_metric_value

    def reset_training(self):
        self.training_epoch_count = 0
        self.best_val_metric_value = 0

    def dump_statistics(self, model, compression_controller):
        statistics = compression_controller.statistics()

        if self.verbose:
            nncf_logger.info(statistics.to_str())
        self.dump_checkpoint(model, compression_controller)

    def dump_checkpoint(self, model, compression_controller):
        checkpoint_path = osp.join(self._checkpoint_save_dir, 'acc_aware_checkpoint_last.pb')
        model.save_weights(checkpoint_path)

        if self.best_val_metric_value == self.current_val_metric_value:
            best_checkpoint_filename = 'acc_aware_checkpoint_best.ckpt'
            best_path = osp.join(self._checkpoint_save_dir, best_checkpoint_filename)
            self._best_checkpoint = best_path
            model.save_weights(best_path)

    def add_tensorboard_scalar(self, key, data, step):
        if self.verbose and self._tensorboard_writer is not None:
            self._tensorboard_writer({key: data}, step)

    def load_best_checkpoint(self, model):
        resuming_checkpoint_path = self._best_checkpoint
        nncf_logger.info('Loading the best checkpoint found during training '
                         '{}...'.format(resuming_checkpoint_path))
        model.load_weights(resuming_checkpoint_path)

    def configure_optimizers(self):
        pass

    def update_learning_rate(self):
        pass


class TFAdaptiveCompressionLevelTrainingRunner(TFAccuracyAwareTrainingRunner,
                                               BaseAdaptiveCompressionLevelTrainingRunner):

    def __init__(self, accuracy_aware_params, verbose=True,
                 minimal_compression_rate=0.05, maximal_compression_rate=0.95, dump_checkpoints=True):
        TFAccuracyAwareTrainingRunner.__init__(self, accuracy_aware_params,
                                               verbose, dump_checkpoints)
        BaseAdaptiveCompressionLevelTrainingRunner.__init__(self, accuracy_aware_params,
                                                            verbose,
                                                            minimal_compression_rate,
                                                            maximal_compression_rate,
                                                            dump_checkpoints)

    def update_training_history(self, compression_rate, best_metric_value):
        best_accuracy_budget = best_metric_value - self.minimal_tolerable_accuracy
        self._compressed_training_history.append((compression_rate, best_accuracy_budget))

    def dump_checkpoint(self, model, compression_controller):
        checkpoint_path = osp.join(self._checkpoint_save_dir, 'acc_aware_checkpoint_last.pb')
        model.save_weights(checkpoint_path)

        if self.best_val_metric_value == self.current_val_metric_value:
            best_path = osp.join(self._checkpoint_save_dir,
                                 'acc_aware_checkpoint_best_compression_rate_'
                                 '{comp_rate:.3f}.ckpt'.format(comp_rate=self.compression_rate_target))
            self._best_checkpoints[self.compression_rate_target] = best_path
            model.save_weights(best_path)

    def load_best_checkpoint(self, model):
        # load checkpoint with highest compression rate and positive acc budget
        possible_checkpoint_rates = [comp_rate for (comp_rate, acc_budget) in self._compressed_training_history
                                     if acc_budget >= 0]
        if not possible_checkpoint_rates:
            nncf_logger.warning('Could not produce a compressed model satisfying the set accuracy '
                                'degradation criterion during training. Increasing the number of training '
                                'epochs')
        best_checkpoint_compression_rate = sorted(possible_checkpoint_rates)[-1]
        resuming_checkpoint_path = self._best_checkpoints[best_checkpoint_compression_rate]
        nncf_logger.info('Loading the best checkpoint found during training '
                         '{}...'.format(resuming_checkpoint_path))
        model.load_weights(resuming_checkpoint_path)

    @property
    def compressed_training_history(self):
        return self._compressed_training_history
