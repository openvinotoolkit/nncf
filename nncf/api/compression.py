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

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TypeVar, List, Tuple

from nncf import NNCFConfig
from nncf.api.statistics import Statistics
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.utils.ordered_enum import OrderedEnum

ModelType = TypeVar('ModelType')


class CompressionLoss(ABC):
    """
    Used to calculate the additional loss to be added to the base loss during the
    training process. It uses the model graph to measure variables and activations
    values of the layers during the loss construction. For example, the $L_0$-based
    sparsity algorithm calculates the number of non-zero weights in convolutional
    and fully-connected layers to construct the loss function.
    """

    @abstractmethod
    def calculate(self, *args, **kwargs) -> Any:
        """
        Calculates the compression loss value.

        :return: The compression loss value.
        """

    @abstractmethod
    def load_state(self, state: Dict[str, object]) -> None:
        """
        Loads the compression loss state.

        :param state: Output of `get_state()` method.
        """

    @abstractmethod
    def get_state(self) -> Dict[str, object]:
        """
        Returns the compression loss state.

        :return: The compression loss state.
        """

    def __call__(self, *args, **kwargs) -> Any:
        """
        Invokes the `CompressionLoss` instance.

        :return: The compression loss value.
        """
        return self.calculate(*args, **kwargs)


class CompressionScheduler(ABC):
    """
    Implements the logic of compression method control during the training process.
    May change the method hyperparameters in regards to the current training step
    or epoch. For example, the sparsity method can smoothly increase the sparsity
    rate over several epochs.

    The `step()` and `epoch_step()` methods of the compression scheduler must be
    called at the beginning of each training step and epoch, respectively.

    ```
    for epoch in range(0, num_epochs):
        scheduler.epoch_step()
        for i, (x, y) in enumerate(dataset):
             scheduler.step()
             ...
    ```
    """

    @abstractmethod
    def step(self, next_step: Optional[int] = None) -> None:
        """
        Should be called at the beginning of each training step to prepare
        the compression method to continue training the model in the `next_step`.

        :param next_step: The global step index for which the compression scheduler
            will update the state of the compression method.
        """

    @abstractmethod
    def epoch_step(self, next_epoch: Optional[int] = None) -> None:
        """
        Should be called at the beginning of each training epoch to prepare
        the compression method to continue training the model in the `next_epoch`.

        :param next_epoch: The epoch index for which the compression scheduler
            will update the state of the compression method.
        """

    @abstractmethod
    def load_state(self, state: Dict[str, object]) -> None:
        """
        Loads the compression scheduler state, but does not update the state of the
        compression method.

        :param state: Output of `get_state()` method.
        """

    @abstractmethod
    def get_state(self) -> Dict[str, object]:
        """
        Returns the compression scheduler state.

        :return: The compression scheduler state.
        """


class CompressionStage(OrderedEnum):
    """
    Specifies the compression stage for the model.
    """

    UNCOMPRESSED = 0
    PARTIALLY_COMPRESSED = 1
    FULLY_COMPRESSED = 2

    def __add__(self, other: 'CompressionStage') -> 'CompressionStage':
        """
        Defines compression stage of a composite compression controller, consist of
        two algorithms, where `self` is the compression stage of the first algorithm
        and other - compression stage of the second one.
            UNCOMPRESSED    & UNCOMPRESSED    = UNCOMPRESSED
            PARTIALLY_COMPRESSED & PARTIALLY_COMPRESSED = PARTIALLY_COMPRESSED
            FULLY_COMPRESSED    & FULLY_COMPRESSED    = FULLY_COMPRESSED
            UNCOMPRESSED    & PARTIALLY_COMPRESSED = PARTIALLY_COMPRESSED
            UNCOMPRESSED    & FULLY_COMPRESSED    = PARTIALLY_COMPRESSED
            PARTIALLY_COMPRESSED & FULLY_COMPRESSED    = PARTIALLY_COMPRESSED

        :param other: An instance of another compression stage.
        :return: The common compression stage of the two algorithms.
        """
        if self == other:
            return self
        return CompressionStage.PARTIALLY_COMPRESSED


class CompressionAlgorithmController(ABC):
    """
    Serves as a handle to the additional modules, parameters and hooks inserted
    into the original uncompressed model to enable algorithm-specific compression.
    Hosts entities that are to be used during the training process, such as
    compression scheduler and compression loss.
    """

    def __init__(self, target_model: ModelType):
        """
        Initializes the internal state of the compression algorithm controller.

        :param target_model: The model with additional modifications necessary
            to enable algorithm-specific compression during fine-tuning built
            by the `CompressionAlgorithmBuilder`.
        """
        self._model = target_model

    @property
    def model(self) -> ModelType:
        """
        :return: The target model.
        """
        return self._model

    @property
    @abstractmethod
    def loss(self) -> CompressionLoss:
        """
        :return: The instance of the `CompressionLoss`.
        """

    @property
    @abstractmethod
    def scheduler(self) -> CompressionScheduler:
        """
        :return: The instance of the `CompressionScheduler`.
        """

    @abstractmethod
    def load_state(self, state: Dict[str, object]) -> None:
        """
        Loads the compression controller state.

        :param state: Output of `get_state()` method.
        """

    @abstractmethod
    def get_state(self) -> Dict[str, object]:
        """
        Returns the compression controller state.

        :return: The compression controller state.
        """

    def compression_stage(self) -> CompressionStage:
        """
        Returns the compression stage. Should be used on saving best checkpoints
        to distinguish between uncompressed, partially compressed, and fully
        compressed models.

        :return: The compression stage of the target model.
        """

    @abstractmethod
    def statistics(self, quickly_collected_only: bool = False) -> Statistics:
        """
        Returns a `Statistics` class instance that contains compression algorithm statistics.

        :param quickly_collected_only: Enables collection of the statistics that
            don't take too much time to compute. Can be helpful for the case when
            need to keep track of statistics on each training batch/step/iteration.
        :return: A `Statistics` class instance that contains compression algorithm statistics.
        """

    def prepare_for_export(self) -> None:
        """
        Prepare the compressed model for deployment.
        """
        self._model = self.strip_model(self._model)

    @abstractmethod
    def export_model(self,
                     save_path: str,
                     save_format: Optional[str] = None,
                     input_names: Optional[List[str]] = None,
                     output_names: Optional[List[str]] = None,
                     model_args: Optional[Tuple[Any, ...]] = None) -> None:
        """
        Exports the compressed model to the specified format for deployment.

        Makes method-specific preparations of the model, (e.g. removing auxiliary
        layers that were used for the model compression), then exports the model to
        the specified path.

        :param save_path: The path where the model will be saved.
        :param save_format: Saving format. The default format will
            be used if `save_format` is not specified.
        :param input_names: Names to be assigned to the input tensors of the model.
        :param output_names: Names to be assigned to the output tensors of the model.
        :param model_args: Tuple of additional positional and keyword arguments
            which are required for the model's forward during export. Should be
            specified in the following format:
                - (a, b, {'x': None, 'y': y}) for positional and keyword arguments.
                - (a, b, {}) for positional arguments only.
                - ({'x': None, 'y': y},) for keyword arguments only.
        """

    def strip_model(self, model: ModelType) -> ModelType:
        """
        Strips auxiliary layers that were used for the model compression, as it's
        only needed for training. The method is used before exporting the model
        in the target format.

        :param model: The compressed model.
        :return: The stripped model.
        """
        return model


class CompressionAlgorithmBuilder(ABC):
    """
    Determines which modifications should be made to the original model in
    order to enable algorithm-specific compression during fine-tuning.
    """

    def __init__(self, config: NNCFConfig, should_init: bool = True):
        """
        Initializes internal state of the compression algorithm builder

        :param config: The dictionary that contains parameters of the compression
            method.
        :param should_init: If False, trainable parameter initialization will be
            skipped during building.
        """
        self.config = config
        self.should_init = should_init

    @abstractmethod
    def apply_to(self, model: ModelType) -> ModelType:
        """
        Applies algorithm-specific modifications to the model.

        :param model: The original uncompressed model.
        :return: The model with additional modifications necessary to enable
            algorithm-specific compression during fine-tuning.
        """

    @abstractmethod
    def build_controller(self, model: ModelType) -> CompressionAlgorithmController:
        """
        Builds `CompressionAlgorithmController` to handle the additional modules,
        parameters, and hooks inserted into the model to enable algorithm-specific
        compression.

        :param model: The model with additional modifications necessary to enable
            algorithm-specific compression during fine-tuning.
        :return: The instance of the `CompressionAlgorithmController`.
        """

    @abstractmethod
    def get_transformation_layout(self, model: ModelType) -> TransformationLayout:
        """
        Computes necessary model transformations to enable algorithm-specific
        compression.

        :param model: The original uncompressed model.
        :return: The instance of the `TransformationLayout` class containing
            a list of algorithm-specific modifications.
        """

    @abstractmethod
    def initialize(self, model: ModelType) -> None:
        """
        Initialize model parameters before training

        :param model: The model with additional modifications necessary to enable
            algorithm-specific compression during fine-tuning.
        """


class CompressionLevel(OrderedEnum):
    """
    Legacy class, now replaced by CompressionStage.
    Supports backward compatibility of older checkpoints produced with NNCF.
    CompressionLevel is deprecated and will be removed in future releases.
    """

    NONE = 0
    PARTIAL = 1
    FULL = 2

    @classmethod
    def map_legacy_level_to_stage(cls):
        return {
            CompressionLevel.NONE: CompressionStage.UNCOMPRESSED,
            CompressionLevel.PARTIAL: CompressionStage.PARTIALLY_COMPRESSED,
            CompressionLevel.FULL: CompressionStage.FULLY_COMPRESSED,
        }
