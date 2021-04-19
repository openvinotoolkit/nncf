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

import io
import os.path as osp
from shutil import copyfile

import matplotlib.pyplot as plt
import PIL.Image
from torchvision.transforms import ToTensor

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

import nncf.accuracy_aware_training.restricted_pickle_module as restricted_pickle_module
from nncf.initialization import TrainEpochArgs
from nncf.api.compression import CompressionLevel
from nncf.common.utils.logger import logger as nncf_logger
from nncf.common.accuracy_aware_training.algo import TrainingRunner
from nncf.accuracy_aware_training.utils import is_main_process, print_statistics, configure_paths
from nncf.checkpoint_loading import load_state


# pylint: disable=E1101
class PTAccuracyAwareTrainingRunner(TrainingRunner):

    def __init__(self, nncf_config, lr_updates_needed=True, verbose=True,
                 minimal_compression_rate=0.05, maximal_compression_rate=0.95):

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
        self.minimal_compression_rate = minimal_compression_rate
        self.maximal_compression_rate = maximal_compression_rate

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

        self.maximal_accuracy_drop = accuracy_aware_config.get('maximal_accuracy_degradation')
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

    def retrieve_original_accuracy(self, model):
        if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            self.uncompressed_model_accuracy = model.module.original_model_accuracy
        else:
            self.uncompressed_model_accuracy = model.original_model_accuracy
        self.minimal_tolerable_accuracy = self.uncompressed_model_accuracy - self.maximal_accuracy_drop

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
        self.lr_scheduler.base_lrs = [base_lr * self._base_lr_reduction_factor_during_search
                                      for base_lr in self.lr_scheduler.base_lrs]
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
                                 'acc_aware_checkpoint_best_compression_rate_'
                                 '{comp_rate:.3f}.pth'.format(comp_rate=self.compression_rate_target))
            copyfile(checkpoint_path, best_path)

    def update_training_history(self, compression_rate, best_metric_value):
        best_accuracy_budget = best_metric_value - self.minimal_tolerable_accuracy
        self._compressed_training_history.append((compression_rate, best_accuracy_budget))

        plt.figure()
        plt.plot(self.compressed_training_history.keys(),
                 self.compressed_training_history.values())
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        image = PIL.Image.open(buf)
        image = ToTensor()(image)
        self.tensorboard_writer.add_image('compression/accuracy_aware/acc_budget_vs_comp_rate',
                                          image,
                                          global_step=len(self.compressed_training_history))

    @property
    def compressed_training_history(self):
        return dict(self._compressed_training_history)

    def load_best_checkpoint(self, model):
        # load checkpoint with highest compression rate and positive acc budget
        possible_checkpoint_rates = [comp_rate for (comp_rate, acc_budget) in self._compressed_training_history
                                     if acc_budget >= 0]
        best_checkpoint_compression_rate = max(possible_checkpoint_rates)
        resuming_checkpoint_path = osp.join(self.checkpoint_save_dir,
                                            'acc_aware_checkpoint_best_compression_rate_'
                                            '{comp_rate:.3f}.pth'.format(comp_rate=best_checkpoint_compression_rate))
        nncf_logger.info('Loading the best checkpoint found during training '
                         '{}...'.format(resuming_checkpoint_path))
        resuming_checkpoint = torch.load(resuming_checkpoint_path, map_location='cpu',
                                         pickle_module=restricted_pickle_module)
        resuming_model_state_dict = resuming_checkpoint.get('state_dict', resuming_checkpoint)
        load_state(model, resuming_model_state_dict, is_resume=True)
