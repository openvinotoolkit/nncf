"""
 Copyright (c) 2019-2020 Intel Corporation
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

from typing import TypeVar
import os.path as osp
from abc import ABC, abstractmethod
from copy import copy
from shutil import copyfile

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

import nncf.accuracy_aware_training.restricted_pickle_module as restricted_pickle_module
from nncf.initialization import TrainEpochArgs
from nncf.compression_method_api import PTStubCompressionScheduler
from nncf.compression_method_api import CompressionLevel
from nncf.common.utils.logger import logger as nncf_logger
from nncf.common.utils.registry import Registry
from nncf.api.composite_compression import CompositeCompressionAlgorithmController
from nncf.api.composite_compression import CompositeCompressionScheduler
from nncf.accuracy_aware_training.utils import is_main_process, print_statistics, configure_paths
from nncf.checkpoint_loading import load_state


ModelType = TypeVar('ModelType')
ACCURACY_AWARE_CONTROLLERS = Registry('accuracy_aware_controllers')


class TrainingRunner(ABC):

    @abstractmethod
    def train_epoch(self, model, compression_controller):
        pass

    @abstractmethod
    def validate(self, model, compression_controller):
        pass

    @abstractmethod
    def configure_optimizers(self):
        pass

    @abstractmethod
    def reset_training(self):
        pass


# pylint: disable=E1101
class AccuracyAwareRunner(TrainingRunner):

    def __init__(self, nncf_config, lr_updates_needed=True, verbose=True):

        self.accuracy_bugdet = None
        self.validate_every_n_epochs = None
        self.compression_rate_target = None
        self.best_compression_level = CompressionLevel.NONE
        self.was_compression_increased_on_prev_step = None
        self._compressed_training_history = []

        self.training_epoch_count = 0
        self.cumulative_epoch_count = 0
        self.best_val_metric_value = 0
        self._base_lr_reduction_factor_during_search = 0.5

        self.lr_updates_needed = lr_updates_needed
        self.verbose = verbose

        accuracy_aware_config = nncf_config.get('accuracy_aware_training_config', None)

        default_parameter_values = {
            'is_higher_metric_better': True,
            'initial_compression_rate_step': 0.1,
            'compression_rate_step_reduction_factor': 0.5,
            'minimal_compression_rate_step': 0.025,
            'patience_epochs': 10,
            'maximal_total_epochs': float('inf'),
        }

        for key in default_parameter_values:
            setattr(self, key, accuracy_aware_config.get(key, default_parameter_values[key]))

        self.minimal_tolerable_accuracy = accuracy_aware_config.get('minimal_tolerable_accuracy')
        self.initial_training_phase_epochs = accuracy_aware_config.get('initial_training_phase_epochs')

        self.compression_rate_step = self.initial_compression_rate_step
        self.step_reduction_factor = self.compression_rate_step_reduction_factor

        self.train_epoch_args = nncf_config.get_extra_struct(TrainEpochArgs)
        self.log_dir = self.train_epoch_args.log_dir if self.train_epoch_args.log_dir is not None \
            else 'runs'
        self.log_dir = configure_paths(self.log_dir)
        self.checkpoint_save_dir = self.log_dir

        self.tensorboard_writer = self.train_epoch_args.tensorboard_writer
        if self.tensorboard_writer is None:
            self.tensorboard_writer = SummaryWriter(self.log_dir)

    def train_epoch(self, model, compression_controller):
        compression_controller.scheduler.epoch_step()

        # assuming that epoch number is only used for logging in train_fn:
        self.train_epoch_args.train_epoch_fn(compression_controller,
                                             model,
                                             self.cumulative_epoch_count,
                                             self.optimizer,
                                             self.lr_scheduler)
        if self.lr_updates_needed:
            self.lr_scheduler.step(self.training_epoch_count if not isinstance(self.lr_scheduler, ReduceLROnPlateau)
                                   else self.best_val_metric_value)

        stats = compression_controller.statistics()

        self.current_val_metric_value = None
        if self.validate_every_n_epochs is not None and \
            self.training_epoch_count % self.validate_every_n_epochs == 0:
            self.current_val_metric_value = self.validate(model, compression_controller)

        if is_main_process():
            if self.verbose:
                print_statistics(stats)
            self.dump_checkpoint(model, compression_controller)
            # dump best checkpoint for current target compression rate
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    self.tensorboard_writer.add_scalar('compression/statistics/{0}'.format(key),
                                                       value, self.cumulative_epoch_count)

        self.training_epoch_count += 1
        self.cumulative_epoch_count += 1

        return self.current_val_metric_value

    def validate(self, model, compression_controller):
        val_metric_value = self.train_epoch_args.eval_fn(model, epoch=self.cumulative_epoch_count)

        compression_level = compression_controller.compression_level()
        is_better_by_accuracy = (not self.is_higher_metric_better) != (val_metric_value > self.best_val_metric_value)
        is_best_by_accuracy = is_better_by_accuracy and compression_level == self.best_compression_level
        is_best = is_best_by_accuracy or compression_level > self.best_compression_level
        if is_best:
            self.best_val_metric_value = val_metric_value
        self.best_compression_level = max(compression_level, self.best_compression_level)

        if is_main_process():
            self.tensorboard_writer.add_scalar('val/accuracy_aware/metric_value',
                                               val_metric_value, self.cumulative_epoch_count)

        return val_metric_value

    def configure_optimizers(self):
        self.optimizer, self.lr_scheduler = self.train_epoch_args.configure_optimizers_fn()

    def reset_training(self):
        self.configure_optimizers()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self._base_lr_reduction_factor_during_search
        self.training_epoch_count = 0
        self.best_val_metric_value = 0

    def dump_checkpoint(self, model, compression_controller):
        checkpoint_path = osp.join(self.checkpoint_save_dir, 'acc_aware_checkpoint_last.pth')
        checkpoint = {
            'epoch': self.cumulative_epoch_count + 1,
            'state_dict': model.state_dict(),
            'best_metric_val': self.best_val_metric_value,
            'current_val_metric_value': self.current_val_metric_value,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': compression_controller.scheduler.get_state()
        }
        torch.save(checkpoint, checkpoint_path)
        if self.best_val_metric_value == self.current_val_metric_value:
            best_path = osp.join(self.checkpoint_save_dir,
                                 'acc_aware_checkpoint_best_'
                                 'compression_rate_{comp_rate}.pth'.format(comp_rate=self.compression_rate_target))
            copyfile(checkpoint_path, best_path)

    def update_training_history(self, compression_rate, best_metric_value):
        best_accuracy_budget = best_metric_value - self.minimal_tolerable_accuracy
        self._compressed_training_history.append((compression_rate, best_accuracy_budget))

    def load_best_checkpoint(self, model):
        # load checkpoint with highest compression rate and positive acc budget
        possible_checkpoint_rates = [comp_rate for (comp_rate, acc_budget) in self._compressed_training_history
                                     if acc_budget >= 0]
        best_checkpoint_compression_rate = max(possible_checkpoint_rates)
        resuming_checkpoint_path = osp.join(self.checkpoint_save_dir,
                                            'acc_aware_checkpoint_best_compression_rate_'
                                            '{comp_rate}.pth'.format(comp_rate=best_checkpoint_compression_rate))
        nncf_logger.info('Loading the best checkpoint found during training '
                         '{}...'.format(resuming_checkpoint_path))
        resuming_checkpoint = torch.load(resuming_checkpoint_path, map_location='cpu',
                                         pickle_module=restricted_pickle_module)
        resuming_model_state_dict = resuming_checkpoint.get('state_dict', resuming_checkpoint,
                                                            pickle_module=restricted_pickle_module)
        load_state(model, resuming_model_state_dict, is_resume=True)


# pylint: disable=E1101
def run_accuracy_aware_compressed_training(model, compression_controller, nncf_config,
                                           minimal_compression_rate=0.05,
                                           maximal_compression_rate=0.95,
                                           lr_updates_needed=True):

    runner = AccuracyAwareRunner(nncf_config, lr_updates_needed=lr_updates_needed)

    accuracy_aware_controller = determine_compression_variable_controller(compression_controller)
    if accuracy_aware_controller is None:
        raise RuntimeError('No compression algorithm supported by the accuracy-aware training '
                           'runner was specified in the config')

    run_initial_training_phase(model, accuracy_aware_controller, runner)

    runner.validate_every_n_epochs = 1
    while runner.compression_rate_step >= runner.minimal_compression_rate_step and \
        runner.cumulative_epoch_count < runner.maximal_total_epochs:

        current_compression_rate = copy(runner.compression_rate_target)
        was_compression_rate_changed = update_target_compression_rate(accuracy_aware_controller, runner)
        nncf_logger.info('Current target compression rate value: '
                         '{comp_rate:.3f}'.format(comp_rate=runner.compression_rate_target))
        nncf_logger.info('Current accuracy budget value: {acc_budget:.3f}'.format(acc_budget=runner.accuracy_bugdet))
        nncf_logger.info('Current compression rate step value: '
                         '{comp_step:.3f}'.format(comp_step=runner.compression_rate_step))

        if was_compression_rate_changed:
            if runner.compression_rate_target < minimal_compression_rate:
                raise RuntimeError('Cannot produce a compressed model with a specified '
                                   'minimal tolerable accuracy')
            if runner.compression_rate_target > maximal_compression_rate:
                nncf_logger.info('Reached maximal possible compression rate '
                                 '{max_rate}'.format(max_rate=maximal_compression_rate))
                return model
            runner.update_training_history(compression_rate=current_compression_rate,
                                           best_metric_value=runner.best_val_metric_value)
            runner.reset_training()
            accuracy_aware_controller.set_compression_rate(runner.compression_rate_target)
            runner.tensorboard_writer.add_scalar('compression/accuracy_aware/target_compression_rate',
                                                 runner.compression_rate_target, runner.cumulative_epoch_count)
            runner.tensorboard_writer.add_scalar('compression/accuracy_aware/compression_rate_step',
                                                 runner.compression_rate_step, runner.cumulative_epoch_count)

        compressed_model_accuracy = runner.train_epoch(model, accuracy_aware_controller, )
        runner.accuracy_bugdet = compressed_model_accuracy - runner.minimal_tolerable_accuracy
        runner.tensorboard_writer.add_scalar('val/accuracy_aware/accuracy_bugdet', runner.accuracy_bugdet,
                                             runner.cumulative_epoch_count)

    runner.load_best_checkpoint(model)

    return model


def run_initial_training_phase(model, accuracy_aware_controller, runner):
    runner.configure_optimizers()
    for _ in range(runner.initial_training_phase_epochs):
        runner.train_epoch(model, accuracy_aware_controller)
    compressed_model_accuracy = runner.validate(model, accuracy_aware_controller)
    runner.accuracy_bugdet = compressed_model_accuracy - runner.minimal_tolerable_accuracy
    runner.tensorboard_writer.add_scalar('val/accuracy_aware/accuracy_bugdet', runner.accuracy_bugdet,
                                         runner.cumulative_epoch_count)
    nncf_logger.info('Accuracy budget value after training is {}'.format(runner.accuracy_bugdet))


# pylint: disable=W0212
def determine_compression_variable_controller(compression_controller):
    accuracy_aware_controllers = ACCURACY_AWARE_CONTROLLERS.registry_dict.values()
    if not isinstance(compression_controller, CompositeCompressionAlgorithmController):
        for ctrl_type in accuracy_aware_controllers:
            if isinstance(compression_controller, ctrl_type):
                return compression_controller
        return None
    for controller in compression_controller._child_ctrls:
        for ctrl_type in accuracy_aware_controllers:
            if isinstance(controller, ctrl_type):
                return controller
    return None


def update_target_compression_rate(accuracy_aware_controller, runner):
    if runner.compression_rate_target is None:
        runner.compression_rate_target = accuracy_aware_controller.compression_rate + \
            np.sign(runner.accuracy_bugdet) * runner.compression_rate_step
        runner.was_compression_increased_on_prev_step = np.sign(runner.accuracy_bugdet)
        disable_compression_schedulers(accuracy_aware_controller)
        return True
    if runner.training_epoch_count >= runner.patience_epochs:
        if runner.was_compression_increased_on_prev_step != np.sign(runner.accuracy_bugdet):
            runner.compression_rate_step *= runner.step_reduction_factor
        runner.compression_rate_target += np.sign(runner.accuracy_bugdet) * runner.compression_rate_step
        runner.was_compression_increased_on_prev_step = np.sign(runner.accuracy_bugdet)
        return True
    return False


# pylint: disable=W0212
def disable_compression_schedulers(accuracy_aware_controller):
    if not isinstance(accuracy_aware_controller, CompositeCompressionAlgorithmController):
        accuracy_aware_controller._scheduler = PTStubCompressionScheduler()
        accuracy_aware_controller._scheduler.target_level = 0.0
    else:
        accuracy_aware_controller._scheduler = CompositeCompressionScheduler()
        for child_controller in accuracy_aware_controller.child_ctrls:
            child_controller._scheduler = PTStubCompressionScheduler()
            child_controller._scheduler.target_level = 0.0
            accuracy_aware_controller._scheduler.add(child_controller._scheduler)
