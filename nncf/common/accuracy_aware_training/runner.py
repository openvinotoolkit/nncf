# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import os
import os.path as osp
import pathlib
from abc import ABC
from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union, cast

from nncf.api.compression import CompressionAlgorithmController
from nncf.api.compression import CompressionStage
from nncf.common.logging import nncf_logger
from nncf.common.plotting import noninteractive_plotting
from nncf.common.statistics import NNCFStatistics
from nncf.common.utils.helpers import configure_accuracy_aware_paths
from nncf.common.utils.tensorboard import prepare_for_tensorboard
from nncf.config.schemata.defaults import AA_COMPRESSION_RATE_STEP_REDUCTION_FACTOR
from nncf.config.schemata.defaults import AA_INITIAL_COMPRESSION_RATE_STEP
from nncf.config.schemata.defaults import AA_INITIAL_TRAINING_PHASE_EPOCHS
from nncf.config.schemata.defaults import AA_LR_REDUCTION_FACTOR
from nncf.config.schemata.defaults import AA_MAXIMAL_TOTAL_EPOCHS
from nncf.config.schemata.defaults import AA_MINIMAL_COMPRESSION_RATE_STEP
from nncf.config.schemata.defaults import AA_PATIENCE_EPOCHS

TModel = TypeVar("TModel")
OptimizerType = TypeVar("OptimizerType")
LRSchedulerType = TypeVar("LRSchedulerType")
TensorboardWriterType = TypeVar("TensorboardWriterType")
Checkpoint = TypeVar("Checkpoint")


Image = Any  # Default type for Image.

try:
    import matplotlib.pyplot as plt
    import PIL.Image
    from PIL.Image import Image

    IMG_PACKAGES_AVAILABLE = True
except ImportError:
    IMG_PACKAGES_AVAILABLE = False

ImageType = TypeVar("ImageType", bound=Image)


class TrainingRunner(ABC):
    """
    Runner is an object that is used by a TrainingLoop instance to control the training process
    via wrapping user-supplied functions such as `train_epoch_fn` and `validate_fn`.
    """

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
    def calculate_minimal_tolerable_accuracy(self, uncompressed_model_accuracy: float) -> None:
        """
        :param uncompressed_model_accuracy: The uncompressed model accuracy.

        Calculate the minimal tolerable accuracy from thr uncompressed_model_accuracy and
        set the obtained value to the `minimal_tolerable_accuracy` attribute of the TrainingRunner.
        """

    @abstractmethod
    def initialize_training_loop_fns(
        self,
        train_epoch_fn: Callable[
            [
                CompressionAlgorithmController,
                TModel,
                Optional[int],
                Optional[OptimizerType],
                Optional[LRSchedulerType],
            ],
            float,
        ],
        validate_fn: Callable[[TModel], float],
        configure_optimizers_fn: Optional[Callable[[], Tuple[OptimizerType, LRSchedulerType]]],
        dump_checkpoint_fn: Optional[
            Callable[
                [
                    TModel,
                    CompressionAlgorithmController,
                    "TrainingRunner",
                    Optional[Union[str, pathlib.Path]],
                ],
                None,
            ]
        ],
        load_checkpoint_fn: Optional[Callable[[Checkpoint, Union[str, pathlib.Path]], None]] = None,
        early_stopping_fn: Optional[
            Callable[
                [
                    float,
                ],
                bool,
            ]
        ] = None,
        update_learning_rate_fn: Optional[Callable[[], None]] = None,
    ) -> None:
        """
        Register the user-supplied functions to be used to control the training process.

        :param train_epoch_fn: a method to fine-tune the model for a single epoch
        (to be called inside the `train_epoch` of the TrainingRunner).
        :param validate_fn: a method to evaluate the model on the validation dataset
        (to be called inside the `train_epoch` of the TrainingRunner).
        :param configure_optimizers_fn: a method to instantiate an optimizer and a learning
        rate scheduler (to be called inside the `configure_optimizers` of the TrainingRunner).
        :param dump_checkpoint_fn: a method to dump a checkpoint.
        """

    @abstractmethod
    def initialize_logging(
        self,
        log_dir: Optional[Union[str, pathlib.Path]] = None,
        tensorboard_writer: Optional[TensorboardWriterType] = None,
    ) -> None:
        """
        Initialize logging related variables

        :param log_dir: The path to be used for logging and checkpoint saving.
        :param tensorboard_writer: The tensorboard object to be used for logging.
        """

    @abstractmethod
    def load_best_checkpoint(self, model: TModel) -> float:
        """
        Load the most accurate model state from the fine-tuning history.

        param model: The model object in which the state will be loaded.
        return: Compression rate of the returned model.
        """

    @abstractmethod
    def is_model_fully_compressed(self, compression_controller: CompressionAlgorithmController) -> bool:
        """
        Check if model is fully compressed

        param compression_controller: Compression controller used to compress the model
        return: True if the model is fully compressed, and False otherwise
        """


class BaseAccuracyAwareTrainingRunner(TrainingRunner):
    """
    The base accuracy-aware training Runner object,
    initialized with the default parameters unless specified in the config.
    """

    def __init__(
        self,
        accuracy_aware_training_params: Dict[str, Union[float, int]],
        uncompressed_model_accuracy: float,
        verbose: bool = True,
        dump_checkpoints: bool = True,
        lr_updates_needed: bool = True,
    ):
        self.uncompressed_model_accuracy = uncompressed_model_accuracy
        self.maximal_relative_accuracy_drop: float = accuracy_aware_training_params.get(
            "maximal_relative_accuracy_degradation", 1.0
        )
        self.maximal_absolute_accuracy_drop = accuracy_aware_training_params.get(
            "maximal_absolute_accuracy_degradation"
        )
        self.maximal_total_epochs = accuracy_aware_training_params.get("maximal_total_epochs", AA_MAXIMAL_TOTAL_EPOCHS)

        self.verbose: bool = verbose
        self.dump_checkpoints: bool = dump_checkpoints

        self.base_lr_reduction_factor_during_search = 1.0
        self.lr_updates_needed: bool = lr_updates_needed

        self.accuracy_budget: Optional[float] = None
        self.is_higher_metric_better: bool = True
        self.optimizer: OptimizerType  # type: ignore
        self.lr_scheduler: LRSchedulerType  # type: ignore

        self.training_epoch_count: int = 0
        self.cumulative_epoch_count: int = 0
        self.best_val_metric_value: float = 0
        self.current_val_metric_value: float = 0
        self.current_loss: float = 0

        self._compressed_training_history: List[Tuple[float, float]] = []
        self._best_checkpoint: Optional[Tuple[Union[str, pathlib.Path], float]] = None

        self._train_epoch_fn: Callable[
            [
                CompressionAlgorithmController,
                TModel,
                Optional[int],
                Optional[OptimizerType],
                Optional[LRSchedulerType],
            ],
            float,
        ]
        self._validate_fn: Callable[[TModel], float]
        self._configure_optimizers_fn: Callable[[], Tuple[OptimizerType, LRSchedulerType]]
        self._dump_checkpoint_fn: Optional[
            Callable[
                [
                    TModel,
                    CompressionAlgorithmController,
                    "TrainingRunner",
                    Optional[Union[str, pathlib.Path]],
                ],
                None,
            ]
        ]
        self._load_checkpoint_fn: Optional[Callable[[Checkpoint, Union[str, pathlib.Path]], None]] = None
        self._early_stopping_fn: Optional[Callable[[float], bool]] = None
        self._update_learning_rate_fn: Optional[Callable[[], None]] = None

        self._log_dir: Union[str, pathlib.Path]
        self._checkpoint_save_dir: Union[str, pathlib.Path]
        self._tensorboard_writer: TensorboardWriterType  # type: ignore

    def train_epoch(self, model: TModel, compression_controller: CompressionAlgorithmController) -> None:
        compression_controller.scheduler.epoch_step()
        # assuming that epoch number is only used for logging in train_fn:
        self.current_loss = self._train_epoch_fn(  # type: ignore
            compression_controller,
            model,
            epoch=self.cumulative_epoch_count,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
        )
        self.training_epoch_count += 1
        self.cumulative_epoch_count += 1

    def dump_statistics(self, model: TModel, compression_controller: CompressionAlgorithmController) -> None:
        statistics = cast(NNCFStatistics, compression_controller.statistics())

        if self.verbose:
            nncf_logger.info(statistics.to_str())

        self.add_tensorboard_scalar(
            "val/accuracy_aware/metric_value",
            self.current_val_metric_value,
            self.cumulative_epoch_count,
        )

        for key, value in prepare_for_tensorboard(statistics).items():
            if isinstance(value, (int, float)):
                self.add_tensorboard_scalar(
                    "compression/statistics/{0}".format(key),
                    value,
                    self.cumulative_epoch_count,
                )

        self.dump_checkpoint(model, compression_controller)

    def calculate_minimal_tolerable_accuracy(self, uncompressed_model_accuracy: float) -> None:
        if self.maximal_absolute_accuracy_drop is not None:
            self.minimal_tolerable_accuracy = uncompressed_model_accuracy - self.maximal_absolute_accuracy_drop
        else:
            self.minimal_tolerable_accuracy = uncompressed_model_accuracy * (
                1 - 0.01 * self.maximal_relative_accuracy_drop
            )

    def dump_checkpoint(self, model: TModel, compression_controller: CompressionAlgorithmController) -> None:
        is_best_checkpoint = (
            self.best_val_metric_value == self.current_val_metric_value
            and self.is_model_fully_compressed(compression_controller)
        )
        if not self.dump_checkpoints and not is_best_checkpoint:
            return

        if self._dump_checkpoint_fn is not None:
            checkpoint_path = self._dump_checkpoint_fn(model, compression_controller, self, self._checkpoint_save_dir)
        else:
            checkpoint_path = self._make_checkpoint_path(is_best=False)
            self._save_checkpoint(model, compression_controller, checkpoint_path)
        nncf_logger.info(f"Saved the checkpoint to {checkpoint_path}")

        if is_best_checkpoint:
            self._save_best_checkpoint(model, compression_controller)

    def configure_optimizers(self) -> None:
        self.optimizer, self.lr_scheduler = self._configure_optimizers_fn()

    def initialize_training_loop_fns(
        self,
        train_epoch_fn: Callable[
            [
                CompressionAlgorithmController,
                TModel,
                Optional[int],
                Optional[OptimizerType],
                Optional[LRSchedulerType],
            ],
            float,
        ],
        validate_fn: Callable[[TModel], float],
        configure_optimizers_fn: Optional[Callable[[], Tuple[OptimizerType, LRSchedulerType]]],
        dump_checkpoint_fn: Optional[
            Callable[
                [
                    TModel,
                    CompressionAlgorithmController,
                    "TrainingRunner",
                    Optional[Union[str, pathlib.Path]],
                ],
                None,
            ]
        ],
        load_checkpoint_fn: Optional[Callable[[Checkpoint, Union[str, pathlib.Path]], None]] = None,
        early_stopping_fn: Optional[Callable[[float], bool]] = None,
        update_learning_rate_fn: Optional[Callable[[], None]] = None,
    ) -> None:
        self._train_epoch_fn = train_epoch_fn  # type: ignore
        self._validate_fn = validate_fn  # type: ignore
        self._configure_optimizers_fn = configure_optimizers_fn  # type: ignore
        self._dump_checkpoint_fn = dump_checkpoint_fn  # type: ignore
        self._load_checkpoint_fn = load_checkpoint_fn  # type: ignore
        self._early_stopping_fn = early_stopping_fn
        self._update_learning_rate_fn = update_learning_rate_fn

    def initialize_logging(
        self,
        log_dir: Optional[Union[str, pathlib.Path]] = None,
        tensorboard_writer: Optional[TensorboardWriterType] = None,
    ) -> None:
        self._log_dir = log_dir if log_dir is not None else osp.join(os.getcwd(), "runs")
        self._log_dir = configure_accuracy_aware_paths(self._log_dir)
        self._checkpoint_save_dir = self._log_dir
        self._tensorboard_writer = tensorboard_writer

    def stop_training(self, compression_controller: CompressionAlgorithmController) -> bool:
        if self.is_model_fully_compressed(compression_controller) and self._early_stopping_fn is not None:
            return self._early_stopping_fn(self.current_val_metric_value)
        return False

    def _save_best_checkpoint(self, model: TModel, compression_controller: CompressionAlgorithmController) -> None:
        best_path = self._make_checkpoint_path(is_best=True)
        self._best_checkpoint = (best_path, compression_controller.compression_rate)
        self._save_checkpoint(model, compression_controller, best_path)
        nncf_logger.info(f"Saved the best model to {best_path}")

    def load_best_checkpoint(self, model: TModel) -> float:
        resuming_checkpoint_path: Union[str, pathlib.Path] = ""
        compression_rate: float = 0.0
        resuming_checkpoint_path, compression_rate = self._best_checkpoint  # type: ignore
        nncf_logger.info(f"Loading the best checkpoint found during training: {resuming_checkpoint_path}")
        self._load_checkpoint(model, resuming_checkpoint_path)
        return compression_rate

    def is_model_fully_compressed(self, compression_controller: CompressionAlgorithmController) -> bool:
        return compression_controller.compression_stage() == CompressionStage.FULLY_COMPRESSED

    @abstractmethod
    def add_tensorboard_scalar(self, key: str, data: Union[int, float], step: int) -> None:
        """
        Add a scalar to tensorboard

        :param key: Scalar key name
        :param data: Scalar data
        :param step: Logging step
        """

    @abstractmethod
    def add_tensorboard_image(self, key: str, data: ImageType, step: int) -> None:
        """
        Add an image to tensorboard

        :param key: Image key name
        :param data: Image data
        :param step: Logging step
        """

    @abstractmethod
    def _save_checkpoint(
        self,
        model: TModel,
        compression_controller: CompressionAlgorithmController,
        checkpoint_path: Union[str, pathlib.Path],
    ) -> None:
        """
        Save a model to the disk.

        :param model: The model to be saved
        :param checkpoint_path: The path to save the checkpoint to
        :param compression_controller: The compression controller to be used during
            model fine-tuning
        """

    @abstractmethod
    def _load_checkpoint(self, model: TModel, checkpoint_path: Union[str, pathlib.Path]) -> None:
        """
        Load model from path.

        :param model: The model object in which the state will be loaded.
        :param checkpoint_path: The path where model checkpoint is stored.
        """

    @abstractmethod
    def _make_checkpoint_path(
        self, is_best: bool, compression_rate: Optional[float] = None
    ) -> Union[str, pathlib.Path]:
        """
        Make a path to save the checkpoint there

        :param is_best: Whether this is the best model yet
        :param compression_rate: Compression rate of the model
        """


class BaseAdaptiveCompressionLevelTrainingRunner(BaseAccuracyAwareTrainingRunner):
    """
    The base adaptive compression level accuracy-aware training Runner object,
    initialized with the default parameters unless specified in the config.
    """

    def __init__(
        self,
        accuracy_aware_training_params: Dict[str, Union[float, int]],
        uncompressed_model_accuracy: float,
        verbose: bool = True,
        dump_checkpoints: bool = True,
        lr_updates_needed: bool = True,
        minimal_compression_rate: float = 0.0,
        maximal_compression_rate: float = 0.95,
    ):
        super().__init__(
            accuracy_aware_training_params,
            uncompressed_model_accuracy,
            verbose,
            dump_checkpoints,
            lr_updates_needed,
        )

        self.compression_rate_step: float = accuracy_aware_training_params.get(
            "initial_compression_rate_step", AA_INITIAL_COMPRESSION_RATE_STEP
        )
        self.compression_rate_step_reduction_factor: float = accuracy_aware_training_params.get(
            "compression_rate_step_reduction_factor",
            AA_COMPRESSION_RATE_STEP_REDUCTION_FACTOR,
        )

        self.lr_reduction_factor: float = accuracy_aware_training_params.get(
            "lr_reduction_factor", AA_LR_REDUCTION_FACTOR
        )
        self.minimal_compression_rate_step: float = accuracy_aware_training_params.get(
            "minimal_compression_rate_step", AA_MINIMAL_COMPRESSION_RATE_STEP
        )
        self.patience_epochs = accuracy_aware_training_params.get("patience_epochs", AA_PATIENCE_EPOCHS)
        self.initial_training_phase_epochs = accuracy_aware_training_params.get(
            "initial_training_phase_epochs", AA_INITIAL_TRAINING_PHASE_EPOCHS
        )

        self.minimal_compression_rate = minimal_compression_rate
        self.maximal_compression_rate = maximal_compression_rate

        self._best_checkpoints: Dict[float, Tuple[Union[str, pathlib.Path], float]] = {}
        self._compression_rate_target: Optional[float] = None
        self.adaptive_controller: CompressionAlgorithmController
        self.was_compression_increased_on_prev_step: Optional[float] = None

    def dump_statistics(self, model: TModel, compression_controller: CompressionAlgorithmController) -> None:
        self.update_training_history(self.compression_rate_target, self.current_val_metric_value)
        super().dump_statistics(model, compression_controller)

    def _save_best_checkpoint(self, model: TModel, compression_controller: CompressionAlgorithmController) -> None:
        best_path = self._make_checkpoint_path(is_best=True, compression_rate=self.compression_rate_target)

        accuracy_budget = self.best_val_metric_value - self.minimal_tolerable_accuracy
        if (
            self.compression_rate_target in self._best_checkpoints
            and self._best_checkpoints[self.compression_rate_target][1] >= accuracy_budget
        ):
            return

        self._best_checkpoints[self.compression_rate_target] = (
            best_path,
            accuracy_budget,
        )
        self._save_checkpoint(model, compression_controller, best_path)
        nncf_logger.info(f"Saved the best model to {best_path}")

    def load_best_checkpoint(self, model: TModel) -> float:
        # load checkpoint with the highest compression rate and positive acc budget
        possible_checkpoint_rates = self.get_compression_rates_with_positive_acc_budget()
        if len(possible_checkpoint_rates) == 0:
            nncf_logger.warning(
                "Could not produce a compressed model satisfying the set accuracy "
                "degradation criterion during training. Increasing the number of training "
                "epochs"
            )
            return self.compression_rate_target

        best_checkpoint_compression_rate = None
        for checkpoint_rate in sorted(possible_checkpoint_rates, key=lambda x: -x):
            if checkpoint_rate in self._best_checkpoints:
                best_checkpoint_compression_rate = checkpoint_rate
                break
        if best_checkpoint_compression_rate is None:
            nncf_logger.error(
                "Could not load the model - no models with positive accuracy budget in compression training history."
            )
            return self.compression_rate_target

        resuming_checkpoint_path: Union[str, pathlib.Path] = self._best_checkpoints[best_checkpoint_compression_rate][0]
        nncf_logger.info(f"Loading the best checkpoint found during training: {resuming_checkpoint_path}")
        self._load_checkpoint(model, resuming_checkpoint_path)
        return best_checkpoint_compression_rate

    @property
    def compression_rate_target(self) -> float:
        if self._compression_rate_target is None:
            return self.adaptive_controller.compression_rate
        return self._compression_rate_target

    @compression_rate_target.setter
    def compression_rate_target(self, value: float) -> None:
        self._compression_rate_target = value

    def update_training_history(self, compression_rate: float, metric_value: float) -> None:
        accuracy_budget = metric_value - self.minimal_tolerable_accuracy
        self._compressed_training_history.append((compression_rate, accuracy_budget))

        if IMG_PACKAGES_AVAILABLE:
            with noninteractive_plotting():
                fig = plt.figure()
                plt.plot(
                    cast(List[float], self.compressed_training_history.keys()),
                    cast(List[float], self.compressed_training_history.values()),
                )
                buf = io.BytesIO()
                plt.savefig(buf, format="jpeg")
                buf.seek(0)
                image = PIL.Image.open(buf)
                self.add_tensorboard_image(
                    "compression/accuracy_aware/acc_budget_vs_comp_rate",
                    image,
                    len(self.compressed_training_history),
                )
                plt.close(fig)

    @property
    def compressed_training_history(self) -> Dict[float, float]:
        return dict(self._compressed_training_history)

    def get_compression_rates_with_positive_acc_budget(self) -> List[float]:
        return [comp_rate for (comp_rate, acc_budget) in self._compressed_training_history if acc_budget >= 0]
