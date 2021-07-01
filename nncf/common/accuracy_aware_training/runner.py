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

from typing import TypeVar
from abc import ABC
from abc import abstractmethod
from nncf.api.compression import CompressionAlgorithmController

ModelType = TypeVar('ModelType')


class TrainingRunner(ABC):
    """
    Runner is an object that is used by a TrainingLoop instance to control the training process
    via wrapping user-supplied functions such as `train_epoch_fn` and `validate_fn`.
    """

    @abstractmethod
    def train_epoch(self, model: ModelType, compression_controller: CompressionAlgorithmController):
        """
        Train the supplied model for a single epoch (single dataset pass).

        :param model: The model to be fine-tuned
        :param compression_controller: The compression controller to be used during
        model fine-tuning
        """

    @abstractmethod
    def validate(self, model: ModelType):
        """
        Compute the target metric value on the validation dataset for the supplied model.

        :param model: The model to be evaluated
        :return: Target validation metric value (float).
        """

    @abstractmethod
    def dump_checkpoint(self, model: ModelType):
        """
        Dump current model checkpoint on disk.

        :param model: The model to be saved
        """

    @abstractmethod
    def configure_optimizers(self):
        """
        Initialize the training optimizer object (and, optionally, the learning rate scheduler object).

        :return: optimizer instance, learning rate scheduler instance (None if not applicable)
        """

    @abstractmethod
    def reset_training(self):
        """
        Initialize all-training related parameters (e.g. epoch count, optimizer, learning rate scheduler).
        """

    @abstractmethod
    def retrieve_original_accuracy(self, model):
        """
        :param model: The model object to retrieve the original accuracy value from.

        Retrive the original uncompressed model accuracy from the model instance and
        set the obtained value to the `uncompressed_model_accuracy` attribute of the TrainingRunner
        """

    @abstractmethod
    def initialize_training_loop_fns(self, train_epoch_fn, validate_fn, configure_optimizers_fn,
                                     tensorboard_writer=None, log_dir=None):
        """
        Register the user-supplied functions to be used to control the training process.

        :param train_epoch_fn: a method to fine-tune the model for a single epoch
        (to be called inside the `train_epoch` of the TrainingRunner).
        :param validate: a method to evaluate the model on the validation dataset
        (to be called inside the `train_epoch` of the TrainingRunner).
        :param configure_optimizers_fn: a method to instantiate an optimizer and a learning
        rate scheduler (to be called inside the `configure_optimizers` of the TrainingRunner).
        :param tensorboard_writer: The tensorboard object to be used for logging.
        :param log_dir: The path to be used for logging and checkpoint saving.
        """


class BaseEarlyStoppingTrainingRunner(TrainingRunner):
    """
    The base early stopping training Runner object, initialized with the default
    parameter values unless specified in the config.
    """

    def __init__(self, early_stopping_config, verbose=True,
                 validate_every_n_epochs=None, dump_checkpoints=True):
        self.accuracy_budget = None
        self.validate_every_n_epochs = None
        self._compressed_training_history = []
        self._best_checkpoints = {}

        self.training_epoch_count = 0
        self.cumulative_epoch_count = 0
        self.best_val_metric_value = 0

        self.validate_every_n_epochs = validate_every_n_epochs
        self.dump_checkpoints = dump_checkpoints
        self.verbose = verbose

        default_parameter_values = {
            'is_higher_metric_better': True,
            'maximal_total_epochs': float('inf'),
        }

        for key in default_parameter_values:
            setattr(self, key, early_stopping_config.get(key, default_parameter_values[key]))

        self.maximal_accuracy_drop = early_stopping_config.get('maximal_accuracy_degradation')
        self.maximal_total_epochs = early_stopping_config.get('maximal_total_epochs')

    def initialize_training_loop_fns(self, train_epoch_fn, validate_fn, configure_optimizers_fn,
                                     tensorboard_writer=None, log_dir=None):
        self._train_epoch_fn = train_epoch_fn
        self._validate_fn = validate_fn
        self._configure_optimizers_fn = configure_optimizers_fn
        self._tensorboard_writer = tensorboard_writer
        self._log_dir = log_dir


class BaseAccuracyAwareTrainingRunner(TrainingRunner):
    """
    The base accuracy-aware training Runner object, initialized with the default
    accuracy-aware parameter values unless specified in the config.
    """

    def __init__(self, accuracy_aware_config, verbose=True,
                 minimal_compression_rate=0.05, maximal_compression_rate=0.95,
                 validate_every_n_epochs=None, dump_checkpoints=True):
        self.accuracy_bugdet = None
        self.validate_every_n_epochs = None
        self.compression_rate_target = None
        self.was_compression_increased_on_prev_step = None
        self._compressed_training_history = []
        self._best_checkpoints = {}

        self.training_epoch_count = 0
        self.cumulative_epoch_count = 0
        self.best_val_metric_value = 0

        self.minimal_compression_rate = minimal_compression_rate
        self.maximal_compression_rate = maximal_compression_rate
        self.validate_every_n_epochs = validate_every_n_epochs
        self.dump_checkpoints = dump_checkpoints
        self.verbose = verbose

        default_parameter_values = {
            'is_higher_metric_better': True,
            'compression_rate_step': 0.1,
            'step_reduction_factor': 0.5,
            'minimal_compression_rate_step': 0.025,
            'patience_epochs': 10,
            'maximal_total_epochs': float('inf'),
        }

        for key in default_parameter_values:
            setattr(self, key, accuracy_aware_config.get(key, default_parameter_values[key]))

        self.maximal_accuracy_drop = accuracy_aware_config.get('maximal_accuracy_degradation')
        self.initial_training_phase_epochs = accuracy_aware_config.get('initial_training_phase_epochs')

    def initialize_training_loop_fns(self, train_epoch_fn, validate_fn, configure_optimizers_fn,
                                     tensorboard_writer=None, log_dir=None):
        self._train_epoch_fn = train_epoch_fn
        self._validate_fn = validate_fn
        self._configure_optimizers_fn = configure_optimizers_fn
        self._tensorboard_writer = tensorboard_writer
        self._log_dir = log_dir
