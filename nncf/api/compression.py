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
from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple, TypeVar

from nncf.api.statistics import Statistics
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.utils.api_marker import api
from nncf.common.utils.backend import copy_model

TModel = TypeVar("TModel")


@api()
class CompressionLoss(ABC):
    """
    Used to calculate the additional loss to be added to the base loss during the
    training process. It uses the model graph to measure variables and activations
    values of the layers during the loss construction. For example, the $L_0$-based
    sparsity algorithm calculates the number of non-zero weights in convolutional
    and fully-connected layers to construct the loss function.
    """

    @abstractmethod
    def calculate(self, *args: Any, **kwargs: Any) -> Any:
        """
        Calculates and returns the compression loss value.
        """

    @abstractmethod
    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Loads the compression loss state.

        :param state: The state of the compression loss, most likely obtained as the result of a `.get_state()` method
            call.
        """

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        Returns the compression loss state.
        """

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Calculates and returns the compression loss value. Same as `.calculate()`.
        """
        return self.calculate(*args, **kwargs)


@api()
class CompressionScheduler(ABC):
    """
    Implements the logic of compression method control during the training process.
    May change the method hyperparameters in regard to the current training step
    or epoch. For example, the sparsity method can smoothly increase the sparsity
    rate over several epochs.

    The `step()` and `epoch_step()` methods of the compression scheduler must be
    called at the beginning of each training step and epoch, respectively:

    ..  code-block:: python

        for epoch in range(0, num_epochs):
            scheduler.epoch_step()
            for i, (x, y) in enumerate(dataset):
                 scheduler.step()
                 ...

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
    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Loads the compression scheduler state, but does not update the state of the
        compression method.

        :param state: Output of `get_state()` method.
        """

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        Returns the compression scheduler state.
        """


@api()
class CompressionStage(IntEnum):
    """
    Specifies the compression stage for the model.
    """

    UNCOMPRESSED = 0
    PARTIALLY_COMPRESSED = 1
    FULLY_COMPRESSED = 2

    def __add__(self, other: int) -> CompressionStage:
        """
        Defines compression stage of a composite compression controller, consist of
        two algorithms, where `self` is the compression stage of the first algorithm
        and other - compression stage of the second one.

        * ``UNCOMPRESSED         & UNCOMPRESSED         == UNCOMPRESSED``
        * ``PARTIALLY_COMPRESSED & PARTIALLY_COMPRESSED == PARTIALLY_COMPRESSED``
        * ``FULLY_COMPRESSED     & FULLY_COMPRESSED     == FULLY_COMPRESSED``
        * ``UNCOMPRESSED         & PARTIALLY_COMPRESSED == PARTIALLY_COMPRESSED``
        * ``UNCOMPRESSED         & FULLY_COMPRESSED     == PARTIALLY_COMPRESSED``
        * ``PARTIALLY_COMPRESSED & FULLY_COMPRESSED     == PARTIALLY_COMPRESSED``

        :param other: An instance of another compression stage.
        :return: The common compression stage of the two algorithms.
        """
        if self == other:
            return self
        return CompressionStage.PARTIALLY_COMPRESSED


@api()
class CompressionAlgorithmController(ABC):
    """
    A handle to the compression-specific modifications made to the model.
    Hosts entities that are to be used during the training process, such as compression scheduler and compression loss.

    :param target_model: The model with additional modifications necessary
        to enable algorithm-specific compression during fine-tuning built by the `CompressionAlgorithmBuilder`.
    """

    def __init__(self, target_model: TModel):
        self._model = target_model

    @property
    def model(self) -> TModel:  # type: ignore[type-var]
        """
        The compressed model object with which this controller is associated.
        """
        return self._model

    @property
    @abstractmethod
    def loss(self) -> CompressionLoss:
        """
        The compression loss for this particular algorithm combination.
        """

    @property
    @abstractmethod
    def scheduler(self) -> CompressionScheduler:
        """
        The compression scheduler for this particular algorithm combination.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Name of the compression algorithm that is being controlled.
        Should be unique to identify the controller and its state among other controllers and their states.
        """

    @abstractmethod
    def load_state(self, state: Dict[str, Dict[str, Any]]) -> None:
        """
        Loads the compression controller state from the map of algorithm name to the dictionary with state attributes.

        :param state: map of the algorithm name to the dictionary with the corresponding state attributes.
        """

    @abstractmethod
    def get_state(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns the compression controller state, which is the map of the algorithm name to the dictionary with the
        corresponding state attributes.
        """

    @abstractmethod
    def get_compression_state(self) -> Dict[str, Any]:
        """
        Returns the compression state - builder and controller state.
        This state should be used to unambiguously resume compression via `compression_state` argument of
        `create_compressed_model` method.

        :return: Compression state of the model to  resume compression from it.
        """

    @abstractmethod
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
        """

    def strip_model(self, model: TModel, do_copy: bool = False) -> TModel:
        """
        Strips auxiliary layers that were used for the model compression, as it's
        only needed for training. The method is used before exporting the model
        in the target format.

        :param model: The compressed model.
        :param do_copy: Modify copy of the model, defaults to False.
        :return: The stripped model.
        """
        if do_copy:
            model = copy_model(model)
        return model

    def prepare_for_export(self) -> None:
        """
        Prepare the compressed model for exporting to a backend-specific model serialization format.
        """
        self._model = self.strip_model(self._model)

    def strip(self, do_copy: bool = True) -> TModel:  # type: ignore[type-var]
        """
        Returns the model object with as much custom NNCF additions as possible removed
        while still preserving the functioning of the model object as a compressed model.

        :param do_copy: If True (default), will return a copy of the currently associated model object. If False,
          will return the currently associated model object "stripped" in-place.
        :return: The stripped model.
        """
        return self.strip_model(self.model, do_copy)  # type: ignore

    @abstractmethod
    def export_model(
        self,
        save_path: str,
        save_format: Optional[str] = None,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        model_args: Optional[Tuple[Any, ...]] = None,
    ) -> None:
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

            * (a, b, {'x': None, 'y': y}) for positional and keyword arguments.
            * (a, b, {}) for positional arguments only.
            * ({'x': None, 'y': y},) for keyword arguments only.
        """

    @property
    @abstractmethod
    def compression_rate(self) -> float:
        """
        Returns a float compression rate value ranging from 0 to 1 (e.g. the sparsity level,
        or the ratio of filters pruned).
        """

    @compression_rate.setter
    @abstractmethod
    def compression_rate(self, compression_rate: float) -> None:
        """
        Set a float compression rate value in the model (e.g. the sparsity
        level or the ratio of filters pruned).

        :param compression_rate: The compressed rate value to be set.
        """

    @abstractmethod
    def disable_scheduler(self) -> None:
        """
        Disables current compression scheduler during training by changing it to a dummy one that does not change
        the compression rate.
        """

    @property
    @abstractmethod
    def maximal_compression_rate(self) -> float:
        """
        Returns the maximal model compression rate supported by the compression controller.
        """


@api()
class CompressionAlgorithmBuilder(ABC):
    """
    Determines which modifications should be made to the original model in order to enable algorithm-specific
    compression during fine-tuning.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        :return: name of the compression algorithm that is being built. Should be unique to identify the builder
        and its state among other builders and their states.
        """

    @abstractmethod
    def apply_to(self, model: TModel) -> TModel:
        """
        Applies algorithm-specific modifications to the model.

        :param model: The original uncompressed model.
        :return: The model with additional modifications necessary to enable
            algorithm-specific compression during fine-tuning.
        """

    @abstractmethod
    def build_controller(self, model: TModel) -> CompressionAlgorithmController:
        """
        Builds an instance of algorithm-specific `nncf.api.compression.CompressionAlgorithmController`
        to handle the additional modules, parameters, and hooks inserted into the model to enable algorithm-specific
        compression.

        :param model: The model with additional modifications necessary to enable
            algorithm-specific compression during fine-tuning.
        :return: The instance of a `CompressionAlgorithmController`-derived class, specific for this algorithm.
        """

    @abstractmethod
    def get_transformation_layout(self, model: TModel) -> TransformationLayout:
        """
        Computes necessary model transformations to enable algorithm-specific
        compression.

        :param model: The original uncompressed model.
        :return: The instance of the `TransformationLayout` class containing
            a list of algorithm-specific modifications.
        """

    @abstractmethod
    def initialize(self, model: TModel) -> None:
        """
        Initialize model parameters before training.

        :param model: The model with additional modifications necessary to enable
            algorithm-specific compression during fine-tuning.
        """

    @abstractmethod
    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Initializes object from the supplied state.

        :param state: The state of the builder, most likely obtained as the result of a `.get_state()` call.
        """

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        Returns a dictionary with Python data structures (dict, list, tuple, str, int, float, True, False, None) that
        represents state of the object.
        """


class CompressionLevel(IntEnum):
    """
    Legacy class, now replaced by CompressionStage.
    Supports backward compatibility of older checkpoints produced with NNCF.
    CompressionLevel is deprecated and will be removed in future releases.
    """

    NONE = 0
    PARTIAL = 1
    FULL = 2

    @classmethod
    def map_legacy_level_to_stage(cls) -> Dict[CompressionLevel, CompressionStage]:
        return {
            CompressionLevel.NONE: CompressionStage.UNCOMPRESSED,
            CompressionLevel.PARTIAL: CompressionStage.PARTIALLY_COMPRESSED,
            CompressionLevel.FULL: CompressionStage.FULLY_COMPRESSED,
        }
