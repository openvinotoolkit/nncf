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

import io
import os
import os.path as osp
from shutil import copyfile

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import PIL.Image
    from torchvision.transforms import ToTensor

    IMG_PACKAGES_AVAILABLE = True
except ImportError:
    IMG_PACKAGES_AVAILABLE = False

from nncf.api.compression import CompressionStage
from nncf.torch.checkpoint_loading import load_state
from nncf.torch.accuracy_aware_training.utils import is_main_process
from nncf.common.utils.helpers import configure_accuracy_aware_paths
from nncf.common.utils.logger import logger as nncf_logger
from nncf.common.utils.tensorboard import prepare_for_tensorboard
from nncf.common.accuracy_aware_training.runner import BaseAccuracyAwareTrainingRunner
from nncf.common.accuracy_aware_training.runner import BaseAdaptiveCompressionLevelTrainingRunner


# TODO: add type hints
# TODO: condition logging messages on self.verbose
# TODO: check tensorboard logs

class PTAccuracyAwareTrainingRunner(BaseAccuracyAwareTrainingRunner):
    """
    BaseAccuracyAwareTrainingRunner
    The Training Runner implementation for PyTorch training code.
    """

    def __init__(self, accuracy_aware_training_params,
                 lr_updates_needed=True, verbose=True,
                 dump_checkpoints=True):
        super().__init__(accuracy_aware_training_params, verbose, dump_checkpoints)

        self.base_lr_reduction_factor_during_search = 1.0
        self.lr_updates_needed = lr_updates_needed
        # TODO: is it possible to provide these in the constructor instead of assigning from .run() later on?
        self.load_checkpoint_fn = None
        self.early_stopping_fn = None
        self.update_learning_rate_fn = None

        self.current_val_metric_value = 0.0

    def initialize_training_loop_fns(self, train_epoch_fn, validate_fn, configure_optimizers_fn,
                                     dump_checkpoint_fn, tensorboard_writer=None, log_dir=None):
        super().initialize_training_loop_fns(train_epoch_fn, validate_fn, configure_optimizers_fn, dump_checkpoint_fn,
                                             tensorboard_writer=tensorboard_writer, log_dir=log_dir)
        self._log_dir = self._log_dir if self._log_dir is not None \
            else os.path.join(os.getcwd(), 'runs')
        # TODO:
        #  1) checkpoints are saved to log_dir, but not to config.checkpoint_save_dir
        #  2) resulting log_dir path:
        #       'runs/resnet18_cifar10_filter_pruning/2022-10-27__23-19-45/accuracy_aware_training/2022-10-27__23-19-45'
        if is_main_process():
            # Only the main process should create a log directory
            self._log_dir = configure_accuracy_aware_paths(self._log_dir)
            self._checkpoint_save_dir = self._log_dir
            if self._tensorboard_writer is None and TENSORBOARD_AVAILABLE:
                self._tensorboard_writer = SummaryWriter(self._log_dir)

    def retrieve_uncompressed_model_accuracy(self, model):
        if hasattr(model, 'original_model_accuracy') or hasattr(model.module, 'original_model_accuracy'):
            if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
                self.uncompressed_model_accuracy = model.module.original_model_accuracy
            else:
                self.uncompressed_model_accuracy = model.original_model_accuracy
        else:
            raise RuntimeError('Original model does not contain the pre-calculated reference metric value')

    def train_epoch(self, model, compression_controller):
        compression_controller.scheduler.epoch_step()
        # TODO: fix type hint for *_fns so that argument names are specified
        # assuming that epoch number is only used for logging in train_fn:
        self._train_epoch_fn(compression_controller,
                             model,
                             epoch=self.cumulative_epoch_count,
                             optimizer=self.optimizer,
                             lr_scheduler=self.lr_scheduler)
        self.training_epoch_count += 1
        self.cumulative_epoch_count += 1

    def validate(self, model):
        # TODO: this method should only compute and return the accuracy on the validation dataset
        with torch.no_grad():
            # TODO: fix type hint for *_fns so that argument names are specified
            self.current_val_metric_value = self._validate_fn(model, epoch=self.cumulative_epoch_count)
        is_better_by_accuracy = (not self.is_higher_metric_better) != (
                self.current_val_metric_value > self.best_val_metric_value)
        if is_better_by_accuracy:
            self.best_val_metric_value = self.current_val_metric_value

        if is_main_process():
            self.add_tensorboard_scalar('val/accuracy_aware/metric_value',
                                        self.current_val_metric_value, self.cumulative_epoch_count)

        return self.current_val_metric_value

    def update_learning_rate(self):
        if self.update_learning_rate_fn is not None:
            self.update_learning_rate_fn(self.lr_scheduler,
                                         self.training_epoch_count,
                                         self.current_val_metric_value)
        else:
            # FIXME:"The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible"
            if self.lr_scheduler is not None and self.lr_updates_needed:
                self.lr_scheduler.step(
                    self.training_epoch_count if not isinstance(self.lr_scheduler, ReduceLROnPlateau)
                    else self.best_val_metric_value)

    def configure_optimizers(self):
        self.optimizer, self.lr_scheduler = self._configure_optimizers_fn()

    def reset_training(self):
        self.configure_optimizers()

        optimizers = self.optimizer if isinstance(self.optimizer, (tuple, list)) else [self.optimizer]
        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= self.base_lr_reduction_factor_during_search

        lr_schedulers = self.lr_scheduler if isinstance(self.lr_scheduler, (tuple, list)) else [self.lr_scheduler]
        for lr_scheduler in lr_schedulers:
            if lr_scheduler is None:
                continue
            for attr_name in ['base_lrs', 'init_lr']:
                if hasattr(lr_scheduler, attr_name):
                    setattr(
                        lr_scheduler,
                        attr_name,
                        [base_lr * self.base_lr_reduction_factor_during_search
                        for base_lr in getattr(lr_scheduler, attr_name)]
                    )

        self.training_epoch_count = 0
        self.best_val_metric_value = 0
        # TODO: introducing a differentiation by compression rates for best val metric instead of zeroing it out will
        #   make an algorithm more flexible
        self.current_val_metric_value = 0

    def dump_statistics(self, model, compression_controller):
        if not is_main_process():
            return

        statistics = compression_controller.statistics()

        if self.verbose:
            nncf_logger.info(statistics.to_str())

        # TODO: pylint Expected type 'NNCFStatistics', got 'Statistics' instead
        for key, value in prepare_for_tensorboard(statistics).items():
            if isinstance(value, (int, float)):
                self.add_tensorboard_scalar('compression/statistics/{0}'.format(key),
                                            value, self.cumulative_epoch_count)

        # TODO: we shouldn't dump checkpoint here / rename the method to .dump_statistics_and_checkpoint()
        self.dump_checkpoint(model, compression_controller)

    def _save_best_checkpoint(self, checkpoint_path):
        best_checkpoint_filename = 'acc_aware_checkpoint_best.pth'
        best_path = osp.join(self._checkpoint_save_dir, best_checkpoint_filename)
        self._best_checkpoint = best_path
        copyfile(checkpoint_path, best_path)
        nncf_logger.info('Copy best checkpoint {} -> {}'.format(checkpoint_path, best_path))

    def dump_checkpoint(self, model, compression_controller):
        if not is_main_process():
            return

        # TODO: remove CompressionStage.FULLY_COMPRESSED? Ideally, we should save the best model even if we have not
        #   fully compressed the model yet.
        #  Additionally, for AdaptiveCompression loop where compression rate frequently changes, FULLY_COMPRESSED state
        #   does not make a lot of sense.
        is_best_checkpoint = (self.best_val_metric_value == self.current_val_metric_value and
                              compression_controller.compression_stage() == CompressionStage.FULLY_COMPRESSED)
        if not self.dump_checkpoints and not is_best_checkpoint:
            return

        if self._dump_checkpoint_fn is not None:
            checkpoint_path = self._dump_checkpoint_fn(model, compression_controller, self, self._checkpoint_save_dir)
        else:
            checkpoint = {
                'epoch': self.cumulative_epoch_count + 1,
                'state_dict': model.state_dict(),
                'compression_state': compression_controller.get_compression_state(),
                'best_metric_val': self.best_val_metric_value,
                'current_val_metric_value': self.current_val_metric_value,
                'optimizer': self.optimizer.state_dict(),
            }
            checkpoint_path = osp.join(self._checkpoint_save_dir, 'acc_aware_checkpoint_last.pth')
            torch.save(checkpoint, checkpoint_path)
        nncf_logger.info("The checkpoint is saved in {}".format(checkpoint_path))
        if is_best_checkpoint:
            self._save_best_checkpoint(checkpoint_path)

    def add_tensorboard_scalar(self, key, data, step):
        if self.verbose and self._tensorboard_writer is not None:
            self._tensorboard_writer.add_scalar(key, data, step)

    def add_tensorboard_image(self, key, data, step):
        if self.verbose and self._tensorboard_writer is not None:
            self._tensorboard_writer.add_image(key, data, step)

    def update_training_history(self, compression_rate, metric_value):
        accuracy_budget = metric_value - self.minimal_tolerable_accuracy
        self._compressed_training_history.append((compression_rate, accuracy_budget))

        if IMG_PACKAGES_AVAILABLE:
            plt.figure()
            plt.plot(self.compressed_training_history.keys(),
                     self.compressed_training_history.values())
            buf = io.BytesIO()
            plt.savefig(buf, format='jpeg')
            buf.seek(0)
            image = PIL.Image.open(buf)
            image = ToTensor()(image)
            self.add_tensorboard_image('compression/accuracy_aware/acc_budget_vs_comp_rate',
                                       image,
                                       len(self.compressed_training_history))

    @property
    def compressed_training_history(self):
        return dict(self._compressed_training_history)

    def resume_from_checkpoint(self, model, resuming_checkpoint_path):
        if self.load_checkpoint_fn is not None:
            self.load_checkpoint_fn(model, resuming_checkpoint_path)
        else:
            resuming_checkpoint = torch.load(resuming_checkpoint_path, map_location='cpu')
            resuming_model_state_dict = resuming_checkpoint.get('state_dict', resuming_checkpoint)
            load_state(model, resuming_model_state_dict, is_resume=True)

    def load_best_checkpoint(self, model):
        resuming_checkpoint_path = self._best_checkpoint
        nncf_logger.info('Loading the best checkpoint found during training {}...'.format(resuming_checkpoint_path))
        self.resume_from_checkpoint(model, resuming_checkpoint_path)

    def stop_training(self, compression_controller):
        if compression_controller.compression_stage() == CompressionStage.FULLY_COMPRESSED \
                and self.early_stopping_fn is not None:
            return self.early_stopping_fn(self.current_val_metric_value)
        return False


class PTAdaptiveCompressionLevelTrainingRunner(PTAccuracyAwareTrainingRunner,
                                               BaseAdaptiveCompressionLevelTrainingRunner):
    def __init__(self, accuracy_aware_training_params,
                 lr_updates_needed=True, verbose=True,
                 minimal_compression_rate=0.05,
                 maximal_compression_rate=0.95,
                 dump_checkpoints=True):

        # TODO: PTAccuracyAwareTrainingRunner and BaseAdaptiveCompressionLevelTrainingRunner have a common parent of
        #       BaseAccuracyAwareTrainingRunner. Due to explicit __init__ calls BAATR constructor is called twice.
        #       This is not critical but not clean.
        #   Call order:
        #        1. PTAdaptiveCompressionLevelTrainingRunner()
        #        2.     PTAccuracyAwareTrainingRunner()
        #        3.         BaseAdaptiveCompressionLevelTrainingRunner()
        #        4.             BaseAccuracyAwareTrainingRunner()   !
        #        5.     BaseAdaptiveCompressionLevelTrainingRunner()
        #        6.         BaseAccuracyAwareTrainingRunner()   !
        PTAccuracyAwareTrainingRunner.__init__(self, accuracy_aware_training_params,
                                               lr_updates_needed, verbose,
                                               dump_checkpoints)

        BaseAdaptiveCompressionLevelTrainingRunner.__init__(self, accuracy_aware_training_params,
                                                            verbose,
                                                            minimal_compression_rate=minimal_compression_rate,
                                                            maximal_compression_rate=maximal_compression_rate,
                                                            dump_checkpoints=dump_checkpoints)

    def dump_statistics(self, model, compression_controller):
        if not is_main_process():
            return

        # TODO: .update_training_history() call does not belong here as it is not part of dumping statistics
        self.update_training_history(self.compression_rate_target, self.current_val_metric_value)
        super().dump_statistics(model, compression_controller)

    def _save_best_checkpoint(self, checkpoint_path):
        # TODO: does not seem to differ a lot from the parent implementation, possible to combine?
        best_checkpoint_filename = 'acc_aware_checkpoint_best_compression_rate_' \
                                   '{comp_rate:.3f}.pth'.format(comp_rate=self.compression_rate_target)
        best_path = osp.join(self._checkpoint_save_dir, best_checkpoint_filename)
        self._best_checkpoints[self.compression_rate_target] = best_path
        copyfile(checkpoint_path, best_path)
        nncf_logger.info('Copy best checkpoint {} -> {}'.format(checkpoint_path, best_path))

    def load_best_checkpoint(self, model):
        # load checkpoint with the highest compression rate and positive acc budget
        possible_checkpoint_rates = self.get_compression_rates_with_positive_acc_budget()
        if len(possible_checkpoint_rates) == 0:
            # TODO: 'Increasing the number of training epochs': checkpoint loading method should not lead to
            #   this kind of logs; move this part out from this method / rephrase the log message
            nncf_logger.warning('Could not produce a compressed model satisfying the set accuracy '
                                'degradation criterion during training. Increasing the number of training '
                                'epochs')
            return

        best_checkpoint_compression_rate = max(possible_checkpoint_rates)
        if best_checkpoint_compression_rate not in self._best_checkpoints:
            # The checkpoint wasn't saved because it was not fully compressed
            # TODO: we should save these checkpoints nevertheless as they still may be useful to the user
            return

        resuming_checkpoint_path = self._best_checkpoints[best_checkpoint_compression_rate]
        nncf_logger.info('Loading the best checkpoint found during training {}...'.format(resuming_checkpoint_path))
        self.resume_from_checkpoint(model, resuming_checkpoint_path)
