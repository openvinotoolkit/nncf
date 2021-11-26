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

from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
from typing import Tuple
from typing import TypeVar
from abc import ABC
from abc import abstractmethod
import pathlib
from nncf.api.compression import CompressionAlgorithmController
from nncf.common.utils.backend import infer_backend_from_compression_controller
from nncf.common.utils.backend import BackendType

ModelType = TypeVar('ModelType')
OptimizerType = TypeVar('OptimizerType')
LRSchedulerType = TypeVar('LRSchedulerType')
TensorboardWriterType = TypeVar('TensorboardWriterType')


class TrainingRunner(ABC):
    """
    Runner is an object that is used by a TrainingLoop instance to control the training process
    via wrapping user-supplied functions such as `train_epoch_fn` and `validate_fn`.
    """

    uncompressed_model_accuracy: float
    maximal_total_epochs: int
    minimal_tolerable_accuracy: float

    @abstractmethod
    def train_epoch(self, model: ModelType, compression_controller: CompressionAlgorithmController) -> None:
        """
        Calls train_epoch_fn and compression_controller.scheduler.epoch_step()

        :param model: The model to be fine-tuned
        :param compression_controller: The compression controller to be used during
        model fine-tuning
        """

    @abstractmethod
    def validate(self, model: ModelType) -> float:
        """
        Compute the target metric value on the validation dataset for the supplied model.

        :param model: The model to be evaluated
        :return: Target validation metric value (float).
        """

    @abstractmethod
    def dump_statistics(self, model: ModelType, compression_controller: CompressionAlgorithmController) -> None:
        """
        Dumps current statistics from compression_controller and dumps model's checkpoint.

        :param model: The model
        :param compression_controller: The compression controller to be used during
        model fine-tuning
        """

    @abstractmethod
    def dump_checkpoint(self, model: ModelType, compression_controller: CompressionAlgorithmController) -> None:
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
    def reset_training(self) -> None:
        """
        Initialize all-training related parameters (e.g. epoch count, optimizer, learning rate scheduler).
        """

    @abstractmethod
    def retrieve_uncompressed_model_accuracy(self, model: ModelType) -> None:
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
    def initialize_training_loop_fns(self, train_epoch_fn: Callable[[CompressionAlgorithmController, ModelType,
                                                                     Optional[OptimizerType],
                                                                     Optional[LRSchedulerType],
                                                                     Optional[int]], None],
                                     validate_fn: Callable[[ModelType, Optional[float]], float],
                                     configure_optimizers_fn: Callable[[], Tuple[OptimizerType, LRSchedulerType]],
                                     dump_checkpoint_fn: Callable[
                                         [ModelType, CompressionAlgorithmController, 'TrainingRunner', str], None],
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
    def load_best_checkpoint(self, model: ModelType) -> None:
        """
        Load the most accurate model state from the fine-tuning history.

        :param model: The model object in which the state will be loaded.
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
            from nncf.torch.accuracy_aware_training.runner import PTAccuracyAwareTrainingRunner
            return PTAccuracyAwareTrainingRunner(self.accuracy_aware_training_params, self.lr_updates_needed,
                                                 self.verbose, self.dump_checkpoints)
        if nncf_backend == BackendType.TENSORFLOW:
            from nncf.tensorflow.accuracy_aware_training.runner import TFAccuracyAwareTrainingRunner
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

    def create_training_loop(self) -> TrainingRunner:
        """
        Creates an object of AdaptiveCompressionLevelTrainingRunner depending on user backend

        :return: AdaptiveCompressionLevelTrainingRunner object
        """
        nncf_backend = infer_backend_from_compression_controller(self.compression_controller)

        if nncf_backend is BackendType.TORCH:
            from nncf.torch.accuracy_aware_training.runner import PTAdaptiveCompressionLevelTrainingRunner
            return PTAdaptiveCompressionLevelTrainingRunner(self.accuracy_aware_training_params,
                                                            self.lr_updates_needed, self.verbose,
                                                            self.minimal_compression_rate,
                                                            self.maximal_compression_rate,
                                                            self.dump_checkpoints)
        if nncf_backend == BackendType.TENSORFLOW:
            from nncf.tensorflow.accuracy_aware_training.runner import TFAdaptiveCompressionLevelTrainingRunner
            return TFAdaptiveCompressionLevelTrainingRunner(self.accuracy_aware_training_params,
                                                            self.verbose,
                                                            self.minimal_compression_rate,
                                                            self.maximal_compression_rate,
                                                            self.dump_checkpoints)
        raise RuntimeError('Got an unsupported value of nncf_backend')


class BaseAccuracyAwareTrainingRunner(TrainingRunner):
    """
    The base accuracy-aware training Runner object,
    initialized with the default parameters unless specified in the config.
    """

    def __init__(self, accuracy_aware_params: Dict[str, object], verbose=True,
                 dump_checkpoints=True):
        self.maximal_relative_accuracy_drop = accuracy_aware_params.get('maximal_relative_accuracy_degradation', 1.0)
        self.maximal_absolute_accuracy_drop = accuracy_aware_params.get('maximal_absolute_accuracy_degradation')
        self.maximal_total_epochs = accuracy_aware_params.get('maximal_total_epochs', 10000)
        self.validate_every_n_epochs = accuracy_aware_params.get('validate_every_n_epochs', 1)

        self.verbose = verbose
        self.dump_checkpoints = dump_checkpoints

        self.accuracy_budget = None
        self.is_higher_metric_better = True
        self._compressed_training_history = []
        self._best_checkpoint = None

        self.training_epoch_count = 0
        self.cumulative_epoch_count = 0
        self.best_val_metric_value = 0

    def initialize_training_loop_fns(self, train_epoch_fn: Callable[[CompressionAlgorithmController, ModelType,
                                                                     Optional[OptimizerType],
                                                                     Optional[LRSchedulerType],
                                                                     Optional[int]], None],
                                     validate_fn: Callable[[ModelType, Optional[float]], float],
                                     configure_optimizers_fn: Callable[[], Tuple[OptimizerType, LRSchedulerType]],
                                     dump_checkpoint_fn: Callable[
                                         [ModelType, CompressionAlgorithmController, TrainingRunner, str], None],
                                     tensorboard_writer=None, log_dir=None):
        self._train_epoch_fn = train_epoch_fn
        self._validate_fn = validate_fn
        self._configure_optimizers_fn = configure_optimizers_fn
        self._dump_checkpoint_fn = dump_checkpoint_fn
        self._tensorboard_writer = tensorboard_writer
        self._log_dir = log_dir

    def calculate_minimal_tolerable_accuracy(self, uncompressed_model_accuracy: float):
        if self.maximal_absolute_accuracy_drop is not None:
            self.minimal_tolerable_accuracy = uncompressed_model_accuracy - self.maximal_absolute_accuracy_drop
        else:
            self.minimal_tolerable_accuracy = uncompressed_model_accuracy * \
                                              (1 - 0.01 * self.maximal_relative_accuracy_drop)


class BaseAdaptiveCompressionLevelTrainingRunner(BaseAccuracyAwareTrainingRunner):
    """
    The base adaptive compression level accuracy-aware training Runner object,
    initialized with the default parameters unless specified in the config.
    """

    def __init__(self, accuracy_aware_params: Dict[str, object], verbose=True,
                 minimal_compression_rate=0.05, maximal_compression_rate=0.95,
                 dump_checkpoints=True):
        super().__init__(accuracy_aware_params, verbose, dump_checkpoints)

        self.compression_rate_step = accuracy_aware_params.get('initial_compression_rate_step', 0.1)
        self.step_reduction_factor = accuracy_aware_params.get('compression_rate_step_reduction_factor', 0.5)
        self.minimal_compression_rate_step = accuracy_aware_params.get('minimal_compression_rate_step', 0.025)
        self.patience_epochs = accuracy_aware_params.get('patience_epochs')
        self.initial_training_phase_epochs = accuracy_aware_params.get('initial_training_phase_epochs')

        self.minimal_compression_rate = minimal_compression_rate
        self.maximal_compression_rate = maximal_compression_rate

        self._best_checkpoints = {}
        self.compression_rate_target = None
        self.was_compression_increased_on_prev_step = None

    def get_compression_rates_with_positive_acc_budget(self) -> List[float]:
        return [comp_rate for (comp_rate, acc_budget) in self._compressed_training_history if acc_budget >= 0]
