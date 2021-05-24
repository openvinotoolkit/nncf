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
        """
        pass

    @abstractmethod
    def validate(self, model: ModelType, compression_controller: CompressionAlgorithmController):
        """
        Compute the target metric value on the validation dataset for the supplied model
        :return: Target validation metric value.
        """
        pass

    @abstractmethod
    def dump_checkpoint(self, model: ModelType, compression_controller: CompressionAlgorithmController):
        """
        Dump current model checkpoint on disk.
        """
        pass

    @abstractmethod
    def configure_optimizers(self):
        """
        Initialize the training optimizer object (and, optionally, the learning rate scheduler object).
        """
        pass

    @abstractmethod
    def reset_training(self):
        """
        Initialize all-training related parameters (e.g. epoch count, optimizer, learning rate scheduler).
        """
        pass

    @abstractmethod
    def retrieve_original_accuracy(self, model):
        """
        Retrive the original uncompressed model accuracy from the model instance.
        """
        pass

    @abstractmethod
    def initialize_training_loop_fns(self, train_epoch_fn, validate_fn, configure_optimizers_fn,
                                     tensorboard_writer=None, log_dir=None):
        """
        Register the user-supplied functions to be used to control the training process.
        """
        pass


class BaseAccuracyAwareTrainingRunner(TrainingRunner):
    """
    The base accuracy-aware training Runner object, initialized with the default
    accuracy-aware parameter values unless specified in the config.
    """
    def __init__(self, accuracy_aware_config, verbose=True,
                 minimal_compression_rate=0.05, maximal_compression_rate=0.95):

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
        self.verbose = verbose

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

    def initialize_training_loop_fns(self, train_epoch_fn, validate_fn, configure_optimizers_fn,
                                     tensorboard_writer=None, log_dir=None):
        self._train_epoch_fn = train_epoch_fn
        self._validate_fn = validate_fn
        self._configure_optimizers_fn = configure_optimizers_fn
        self._tensorboard_writer = tensorboard_writer
        self._log_dir = log_dir
