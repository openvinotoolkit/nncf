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

from __future__ import annotations

import io
import os
import os.path as osp
import pathlib
from abc import ABC
from abc import abstractmethod
from typing import Callable, Dict, List, Optional, Union, Tuple, TypeVar, TYPE_CHECKING

from nncf.api.compression import CompressionAlgorithmController
from nncf.api.compression import CompressionStage
from nncf.common.utils.backend import infer_backend_from_compression_controller
from nncf.common.utils.backend import BackendType
from nncf.common.utils.helpers import configure_accuracy_aware_paths
from nncf.common.utils.logger import logger as nncf_logger
from nncf.common.utils.tensorboard import prepare_for_tensorboard
from nncf.config.schemata.defaults import AA_COMPRESSION_RATE_STEP_REDUCTION_FACTOR
from nncf.config.schemata.defaults import AA_INITIAL_COMPRESSION_RATE_STEP
from nncf.config.schemata.defaults import AA_INITIAL_TRAINING_PHASE_EPOCHS
from nncf.config.schemata.defaults import AA_LR_REDUCTION_FACTOR
from nncf.config.schemata.defaults import AA_MAXIMAL_TOTAL_EPOCHS
from nncf.config.schemata.defaults import AA_MINIMAL_COMPRESSION_RATE_STEP
from nncf.config.schemata.defaults import AA_PATIENCE_EPOCHS

if TYPE_CHECKING:
    from nncf.torch.accuracy_aware_training.runner import PTAdaptiveCompressionLevelTrainingRunner
    from nncf.tensorflow.accuracy_aware_training.runner import TFAdaptiveCompressionLevelTrainingRunner

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


TModel = TypeVar('TModel')
OptimizerType = TypeVar('OptimizerType')
LRSchedulerType = TypeVar('LRSchedulerType')
TensorboardWriterType = TypeVar('TensorboardWriterType')


class TrainingRunner(ABC):
    """
    Runner is an object that is used by a TrainingLoop instance to control the training process
    via wrapping user-supplied functions such as `train_epoch_fn` and `validate_fn`.
    """

    CHECKPOINT_PATH_EXTENSION: str
    uncompressed_model_accuracy: float
    maximal_total_epochs: int
    minimal_tolerable_accuracy: float

    @abstractmethod
    def train_epoch(self, model: TModel, compression_controller: CompressionAlgorithmController) -> None:
        """
        Calls train_epoch_fn and compression_controller.scheduler.epoch_step()

        :param model: The model to be fine-tuned
        :param compression_controller: The compression controller to be used during
        model fine-tuning
        """

    @abstractmethod
    def validate(self, model: TModel) -> float:
        """
        Compute the target metric value on the validation dataset for the supplied model.

        :param model: The model to be evaluated
        :return: Target validation metric value (float).
        """

    @abstractmethod
    def dump_statistics(self, model: TModel, compression_controller: CompressionAlgorithmController) -> None:
        """
        Dumps current statistics from compression_controller and dumps model's checkpoint.

        :param model: The model
        :param compression_controller: The compression controller to be used during
        model fine-tuning
        """

    @abstractmethod
    def dump_checkpoint(self, model: TModel, compression_controller: CompressionAlgorithmController) -> None:
        """
        Dump current model checkpoint on disk.

        :param model: The model to be saved
        :param compression_controller: The compression controller to be used during
        model fine-tuning
        """

    @abstractmethod
    def configure_optimizers(self) -> None:
        """
        Initialize the training optimizer object (and, optionally, the learning rate scheduler object).
        """

    @abstractmethod
    def update_learning_rate(self) -> None:
        """
        Update learning rate.
        """

    @abstractmethod
    def reset_training(self) -> None:
        """
        Initialize all-training related parameters (e.g. epoch count, optimizer, learning rate scheduler).
        """

    @abstractmethod
    def retrieve_uncompressed_model_accuracy(self, model: TModel) -> None:
        """
        :param model: The model object to retrieve the original accuracy value from.

        Retrive the original uncompressed model accuracy from the model instance and
        set the obtained value to the `uncompressed_model_accuracy` attribute of the TrainingRunner.
        """

    @abstractmethod
    def calculate_minimal_tolerable_accuracy(self, uncompressed_model_accuracy: float) -> None:
        """
        :param uncompressed_model_accuracy: The uncompressed model accuracy.

        Calculate the minimal tolerable accuracy from thr uncompressed_model_accuracy and
        set the obtained value to the `minimal_tolerable_accuracy` attribute of the TrainingRunner.
        """

    @abstractmethod
    def initialize_training_loop_fns(self, train_epoch_fn: Callable[[CompressionAlgorithmController, TModel,
                                                                     Optional[OptimizerType],
                                                                     Optional[LRSchedulerType],
                                                                     Optional[int]], None],
                                     validate_fn: Callable[[TModel, Optional[int]], float],
                                     configure_optimizers_fn: Callable[[], Tuple[OptimizerType, LRSchedulerType]],
                                     dump_checkpoint_fn: Callable[
                                         [TModel, CompressionAlgorithmController, 'TrainingRunner', str], None],
                                     tensorboard_writer: TensorboardWriterType = None,
                                     log_dir: Union[str, pathlib.Path] = None):
        """
        Register the user-supplied functions to be used to control the training process.

        :param train_epoch_fn: a method to fine-tune the model for a single epoch
        (to be called inside the `train_epoch` of the TrainingRunner).
        :param validate_fn: a method to evaluate the model on the validation dataset
        (to be called inside the `train_epoch` of the TrainingRunner).
        :param configure_optimizers_fn: a method to instantiate an optimizer and a learning
        rate scheduler (to be called inside the `configure_optimizers` of the TrainingRunner).
        :param dump_checkpoint_fn: a method to dump a checkpoint.
        :param tensorboard_writer: The tensorboard object to be used for logging.
        :param log_dir: The path to be used for logging and checkpoint saving.
        """

    @abstractmethod
    def load_best_checkpoint(self, model: TModel) -> None:
        """
        Load the most accurate model state from the fine-tuning history.

        param model: The model object in which the state will be loaded.
        """

    @abstractmethod
    def _save_checkpoint(self, model: TModel, compression_controller: CompressionAlgorithmController,
                         checkpoint_path: str) -> None:
        """
        Save a model to the disk.

        :param model: The model to be saved
        :param checkpoint_path: The path to save the checkpoint to
        :param compression_controller: The compression controller to be used during
        model fine-tuning
        """

    def _load_checkpoint(self, model, checkpoint_path):
        """
        Load model from path.

        :param model: The model object in which the state will be loaded.
        :param checkpoint_path: The path where model checkpoint is stored.
        """


class TrainingRunnerCreator(ABC):
    """
    Declares the factory method returning TrainingRunner object
    """

    @abstractmethod
    def create_training_loop(self) -> TrainingRunner:
        pass


class EarlyExitTrainingRunnerCreator(TrainingRunnerCreator):
    """
    Class creates an Early Exit Training Runner depending on an used backend.
    """

    def __init__(self, accuracy_aware_training_params: Dict[str, object],
                 compression_controller: CompressionAlgorithmController,
                 lr_updates_needed: bool, verbose: bool, dump_checkpoints: bool):
        self.accuracy_aware_training_params = accuracy_aware_training_params
        self.compression_controller = compression_controller
        self.lr_updates_needed = lr_updates_needed
        self.verbose = verbose
        self.dump_checkpoints = dump_checkpoints

    def create_training_loop(self) -> TrainingRunner:
        """
        Creates an object of AccuracyAwareTrainingRunner depending on user backend

        :return: AccuracyAwareTrainingRunner object
        """
        nncf_backend = infer_backend_from_compression_controller(self.compression_controller)
        if nncf_backend is BackendType.TORCH:
            from nncf.torch.accuracy_aware_training.runner import PTAccuracyAwareTrainingRunner #pylint: disable=cyclic-import
            return PTAccuracyAwareTrainingRunner(self.accuracy_aware_training_params, self.lr_updates_needed,
                                                 self.verbose, self.dump_checkpoints)
        if nncf_backend == BackendType.TENSORFLOW:
            from nncf.tensorflow.accuracy_aware_training.runner import TFAccuracyAwareTrainingRunner #pylint: disable=cyclic-import
            return TFAccuracyAwareTrainingRunner(self.accuracy_aware_training_params,
                                                 self.verbose, self.dump_checkpoints)
        raise RuntimeError('Got an unsupported value of nncf_backend')


class AdaptiveCompressionLevelTrainingRunnerCreator(TrainingRunnerCreator):
    """
    Class creates an Adaptive Compression Level Training Runner depending on an used backend.
    """

    def __init__(self, accuracy_aware_training_params: Dict[str, object],
                 compression_controller: CompressionAlgorithmController,
                 lr_updates_needed: bool, verbose: bool, minimal_compression_rate: float,
                 maximal_compression_rate: float, dump_checkpoints: bool):
        self.accuracy_aware_training_params = accuracy_aware_training_params
        self.compression_controller = compression_controller
        self.lr_updates_needed = lr_updates_needed
        self.verbose = verbose
        self.minimal_compression_rate = minimal_compression_rate
        self.maximal_compression_rate = maximal_compression_rate
        self.dump_checkpoints = dump_checkpoints

    def create_training_loop(self) -> Union[PTAdaptiveCompressionLevelTrainingRunner,
                                            TFAdaptiveCompressionLevelTrainingRunner]:
        """
        Creates an object of AdaptiveCompressionLevelTrainingRunner depending on user backend

        :return: AdaptiveCompressionLevelTrainingRunner object
        """
        nncf_backend = infer_backend_from_compression_controller(self.compression_controller)

        if nncf_backend is BackendType.TORCH:
            from nncf.torch.accuracy_aware_training.runner import PTAdaptiveCompressionLevelTrainingRunner #pylint: disable=cyclic-import
            return PTAdaptiveCompressionLevelTrainingRunner(self.accuracy_aware_training_params,
                                                            self.lr_updates_needed, self.verbose,
                                                            self.dump_checkpoints,
                                                            self.minimal_compression_rate,
                                                            self.maximal_compression_rate)
        if nncf_backend == BackendType.TENSORFLOW:
            from nncf.tensorflow.accuracy_aware_training.runner import TFAdaptiveCompressionLevelTrainingRunner #pylint: disable=cyclic-import
            return TFAdaptiveCompressionLevelTrainingRunner(self.accuracy_aware_training_params,
                                                            self.lr_updates_needed, self.verbose,
                                                            self.dump_checkpoints,
                                                            self.minimal_compression_rate,
                                                            self.maximal_compression_rate)
        raise RuntimeError('Got an unsupported value of nncf_backend')


class BaseAccuracyAwareTrainingRunner(TrainingRunner, ABC):
    """
    The base accuracy-aware training Runner object,
    initialized with the default parameters unless specified in the config.
    """

    def __init__(self, accuracy_aware_training_params: Dict[str, object], verbose=True,
                 dump_checkpoints=True, lr_updates_needed=True, **kwargs):
        self.maximal_relative_accuracy_drop = accuracy_aware_training_params.get(
            'maximal_relative_accuracy_degradation', 1.0)
        self.maximal_absolute_accuracy_drop = accuracy_aware_training_params.get(
            'maximal_absolute_accuracy_degradation')
        self.maximal_total_epochs = accuracy_aware_training_params.get('maximal_total_epochs', AA_MAXIMAL_TOTAL_EPOCHS)

        self.verbose = verbose
        self.dump_checkpoints = dump_checkpoints

        self.accuracy_budget = None
        self.is_higher_metric_better = True
        self._compressed_training_history = []
        self._best_checkpoint = None

        self.training_epoch_count = 0
        self.cumulative_epoch_count = 0
        self.best_val_metric_value = 0
        self.current_val_metric_value = 0

        self.optimizer = None
        self.lr_scheduler = None

        self.base_lr_reduction_factor_during_search = 1.0
        self.lr_updates_needed = lr_updates_needed
        self.load_checkpoint_fn = None
        self.early_stopping_fn = None
        self.update_learning_rate_fn = None

        self._train_epoch_fn = None
        self._validate_fn = None
        self._configure_optimizers_fn = None
        self._dump_checkpoint_fn = None

        self._log_dir = None
        self._checkpoint_save_dir = None
        self._tensorboard_writer = None

    def train_epoch(self, model, compression_controller):
        compression_controller.scheduler.epoch_step()
        # assuming that epoch number is only used for logging in train_fn:
        self._train_epoch_fn(compression_controller,
                             model,
                             epoch=self.cumulative_epoch_count,
                             optimizer=self.optimizer,
                             lr_scheduler=self.lr_scheduler)
        self.training_epoch_count += 1
        self.cumulative_epoch_count += 1

    def dump_statistics(self, model, compression_controller):
        statistics = compression_controller.statistics()

        if self.verbose:
            nncf_logger.info(statistics.to_str())

        self.add_tensorboard_scalar('val/accuracy_aware/metric_value',
                                    self.current_val_metric_value, self.cumulative_epoch_count)

        for key, value in prepare_for_tensorboard(statistics).items():
            if isinstance(value, (int, float)):
                self.add_tensorboard_scalar('compression/statistics/{0}'.format(key),
                                            value, self.cumulative_epoch_count)

        self.dump_checkpoint(model, compression_controller)

    def calculate_minimal_tolerable_accuracy(self, uncompressed_model_accuracy: float):
        if self.maximal_absolute_accuracy_drop is not None:
            self.minimal_tolerable_accuracy = uncompressed_model_accuracy - self.maximal_absolute_accuracy_drop
        else:
            self.minimal_tolerable_accuracy = uncompressed_model_accuracy * \
                                              (1 - 0.01 * self.maximal_relative_accuracy_drop)

    def dump_checkpoint(self, model, compression_controller):
        is_best_checkpoint = (self.best_val_metric_value == self.current_val_metric_value and
                              compression_controller.compression_stage() == CompressionStage.FULLY_COMPRESSED)
        if not self.dump_checkpoints and not is_best_checkpoint:
            return

        if self._dump_checkpoint_fn is not None:
            checkpoint_path = self._dump_checkpoint_fn(model, compression_controller, self, self._checkpoint_save_dir)
        else:
            checkpoint_path = osp.join(self._checkpoint_save_dir,
                                       f'acc_aware_checkpoint_last{self.CHECKPOINT_PATH_EXTENSION}')
            self._save_checkpoint(model, compression_controller, checkpoint_path)
        nncf_logger.info("The checkpoint is saved in {}".format(checkpoint_path))

        if is_best_checkpoint:
            self._save_best_checkpoint(model, compression_controller)

    def configure_optimizers(self):
        self.optimizer, self.lr_scheduler = self._configure_optimizers_fn()

    def initialize_training_loop_fns(self, train_epoch_fn: Callable[[CompressionAlgorithmController, TModel,
                                                                     Optional[OptimizerType],
                                                                     Optional[LRSchedulerType],
                                                                     Optional[int]], None],
                                     validate_fn: Callable[[TModel, Optional[float]], float],
                                     configure_optimizers_fn: Callable[[], Tuple[OptimizerType, LRSchedulerType]],
                                     dump_checkpoint_fn: Callable[
                                         [TModel, CompressionAlgorithmController, TrainingRunner, str], None],
                                     tensorboard_writer=None, log_dir=None):
        self._train_epoch_fn = train_epoch_fn
        self._validate_fn = validate_fn
        self._configure_optimizers_fn = configure_optimizers_fn
        self._dump_checkpoint_fn = dump_checkpoint_fn
        self._tensorboard_writer = tensorboard_writer

    def stop_training(self, compression_controller):
        if compression_controller.compression_stage() == CompressionStage.FULLY_COMPRESSED \
                and self.early_stopping_fn is not None:
            return self.early_stopping_fn(self.current_val_metric_value)
        return False

    def _save_best_checkpoint(self, model, compression_controller):
        best_path = osp.join(self._checkpoint_save_dir, f'acc_aware_checkpoint_best{self.CHECKPOINT_PATH_EXTENSION}')
        self._best_checkpoint = best_path
        self._save_checkpoint(model, compression_controller, best_path)
        nncf_logger.info('Saved the best model to {}'.format(best_path))

    def add_tensorboard_scalar(self, key, data, step):
        if self.verbose and self._tensorboard_writer is not None:
            self._tensorboard_writer.add_scalar(key, data, step)

    def add_tensorboard_image(self, key, data, step):
        if self.verbose and self._tensorboard_writer is not None:
            self._tensorboard_writer.add_image(key, data, step)

    def load_best_checkpoint(self, model):
        resuming_checkpoint_path = self._best_checkpoint
        nncf_logger.info('Loading the best checkpoint found during training {}...'.format(resuming_checkpoint_path))
        self._load_checkpoint(model, resuming_checkpoint_path)

    def _initialize_log_dir(self, log_dir):
        self._log_dir = log_dir if log_dir is not None else osp.join(os.getcwd(), 'runs')
        self._log_dir = configure_accuracy_aware_paths(self._log_dir)
        self._checkpoint_save_dir = self._log_dir
        if self._tensorboard_writer is None and TENSORBOARD_AVAILABLE:
            self._tensorboard_writer = SummaryWriter(self._log_dir)


class BaseAdaptiveCompressionLevelTrainingRunner(BaseAccuracyAwareTrainingRunner, ABC):
    """
    The base adaptive compression level accuracy-aware training Runner object,
    initialized with the default parameters unless specified in the config.
    """

    def __init__(self, accuracy_aware_training_params: Dict[str, object], verbose=True,
                 dump_checkpoints=True, lr_updates_needed=True,
                 minimal_compression_rate=0.05, maximal_compression_rate=0.95):
        super().__init__(accuracy_aware_training_params, verbose, dump_checkpoints, lr_updates_needed)

        self.compression_rate_step = accuracy_aware_training_params.get('initial_compression_rate_step',
                                                                        AA_INITIAL_COMPRESSION_RATE_STEP)
        self.compression_rate_step_reduction_factor = accuracy_aware_training_params.get(
            'compression_rate_step_reduction_factor', AA_COMPRESSION_RATE_STEP_REDUCTION_FACTOR)
        self.lr_reduction_factor = accuracy_aware_training_params.get('lr_reduction_factor', AA_LR_REDUCTION_FACTOR)
        self.minimal_compression_rate_step = accuracy_aware_training_params.get('minimal_compression_rate_step',
                                                                                AA_MINIMAL_COMPRESSION_RATE_STEP)
        self.patience_epochs = accuracy_aware_training_params.get('patience_epochs', AA_PATIENCE_EPOCHS)
        self.initial_training_phase_epochs = accuracy_aware_training_params.get('initial_training_phase_epochs',
                                                                                AA_INITIAL_TRAINING_PHASE_EPOCHS)

        self.minimal_compression_rate = minimal_compression_rate
        self.maximal_compression_rate = maximal_compression_rate

        self._best_checkpoints = {}
        self._compression_rate_target = None
        self.adaptive_controller = None
        self.was_compression_increased_on_prev_step = None

    def dump_statistics(self, model, compression_controller):
        self.update_training_history(self.compression_rate_target, self.current_val_metric_value)
        super().dump_statistics(model, compression_controller)

    def _save_best_checkpoint(self, model, compression_controller):
        best_checkpoint_filename = f'acc_aware_checkpoint_best_' \
                                   f'{self.compression_rate_target:.3f}{self.CHECKPOINT_PATH_EXTENSION}'
        best_path = osp.join(self._checkpoint_save_dir, best_checkpoint_filename)

        accuracy_budget = self.best_val_metric_value - self.minimal_tolerable_accuracy
        if self.compression_rate_target in self._best_checkpoints and \
                self._best_checkpoints[self.compression_rate_target][1] >= accuracy_budget:
            return

        self._best_checkpoints[self.compression_rate_target] = (best_path, accuracy_budget)
        self._save_checkpoint(model, compression_controller, best_path)
        nncf_logger.info('Saved the best model to {}'.format(best_path))

    def load_best_checkpoint(self, model):
        # load checkpoint with the highest compression rate and positive acc budget
        possible_checkpoint_rates = self.get_compression_rates_with_positive_acc_budget()
        if len(possible_checkpoint_rates) == 0:
            nncf_logger.warning('Could not produce a compressed model satisfying the set accuracy '
                                'degradation criterion during training. Increasing the number of training '
                                'epochs')
            return

        best_checkpoint_compression_rate = max(possible_checkpoint_rates)
        if best_checkpoint_compression_rate not in self._best_checkpoints:
            # The checkpoint wasn't saved because it was not fully compressed
            return

        resuming_checkpoint_path = self._best_checkpoints[best_checkpoint_compression_rate][0]
        nncf_logger.info('Loading the best checkpoint found during training {}...'.format(resuming_checkpoint_path))
        self._load_checkpoint(model, resuming_checkpoint_path)

    @property
    def compression_rate_target(self):
        if self._compression_rate_target is None:
            return self.adaptive_controller.compression_rate
        return self._compression_rate_target

    @compression_rate_target.setter
    def compression_rate_target(self, value):
        self._compression_rate_target = value

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
            self.add_tensorboard_image('compression/accuracy_aware/acc_budget_vs_comp_rate', image,
                                       len(self.compressed_training_history))

    @property
    def compressed_training_history(self):
        return dict(self._compressed_training_history)

    def get_compression_rates_with_positive_acc_budget(self) -> List[float]:
        return [comp_rate for (comp_rate, acc_budget) in self._compressed_training_history if acc_budget >= 0]
