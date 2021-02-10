#
#  Copyright (c) 2019-2020 Intel Corporation
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#       http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

"""
@package docstring
This package defines the API for the NNCF compression methods so that the user could
extend the existing algorithms.
"""
from typing import Any, Dict, Optional, TypeVar

from nncf import NNCFConfig
from nncf.common.graph.model_transformer import ModelTransformer
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.utils.ordered_enum import OrderedEnum

DOMAIN_CUSTOM_OPS_NAME = "org.openvinotoolkit"

ModelType = TypeVar('ModelType')


class CompressionLoss:
    """
    Used to calculate the additional loss to be added to the base loss during the
    training process. It uses the model graph to measure variables and activations
    values of the layers during the loss construction. For example, the $L_0$-based
    sparsity algorithm calculates the number of non-zero weights in convolutional
    and fully-connected layers to construct the loss function.
    """

    def calculate(self) -> Any:
        """
        Calculates the compression loss value.

        :return: The compression loss value.
        """
        return 0

    def statistics(self, quickly_collected_only: bool = False) -> Dict[str, object]:
        """
        Returns a dictionary of printable statistics.

        :param quickly_collected_only: Enables collection of the statistics that
            don't take too much time to compute. Can be helpful for the case when
            need to keep track of statistics on each training batch/step/iteration.
        :return: A dictionary of printable statistics.
        """
        return {}

    def __call__(self, *args, **kwargs) -> Any:
        """
        Invokes the `CompressionLoss` instance.

        :return: The compression loss value.
        """
        return self.calculate(*args, **kwargs)


class CompressionScheduler:
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

    def __init__(self):
        """
        Initializes the internal state of the compression scheduler specified by:
            - `current_step` is the index of the global training step, counted
            from 0 to the end of training. The initial value is -1
            - `current_epoch` is the training epoch index (numbering from zero).
            The initial value is -1.

        The `current_step` and` current_epoch` specify the training step and epoch,
        respectively, for which the compression scheduler has updated the state of
        the compression method, in particular its hyperparameters. It means that
        the compression method is configured and ready to continue training at
        `current_step` and `current_epoch`.

        When `current_step` is -1, it means that the compression scheduler did not
        update the compression method state taking into account the training step,
        there is the same for current_epoch is -1.
        """
        self.current_step = -1
        self.current_epoch = -1

    def step(self, next_step: Optional[int] = None) -> None:
        """
        Should be called at the beginning of each training step to prepare
        the compression method to continue training the model in the 'next_step'.

        :param next_step: The global step index for which the compression scheduler
            will update the state of the compression method.
        """
        if next_step is None:
            next_step = self.current_step + 1
        self.current_step = next_step

    def epoch_step(self, next_epoch: Optional[int] = None) -> None:
        """
        Should be called at the beginning of each training epoch to prepare
        the compression method to continue training the model in the 'next_epoch'.

        :param next_epoch: The epoch index for which the compression scheduler
            will update the state of the compression method.
        """
        if next_epoch is None:
            next_epoch = self.current_epoch + 1
        self.current_epoch = next_epoch

    def load_state(self, step: int, epoch: int) -> None:
        """
        Loads the compression scheduler state, but does not update the state of the
        compression method.

        :param step: The index of the global training step.
        :param epoch: The training epoch index.
        """
        self.current_step = step
        self.current_epoch = epoch

    def get_state(self) -> Dict[str, int]:
        """
        Returns the compression scheduler state.

        :return: The compression scheduler state.
        """
        return {
            'current_step': self.current_step,
            'current_epoch': self.current_epoch
        }


class CompressionLevel(OrderedEnum):
    """
    Specifies the compression level for the model.
    """

    NONE = 0
    PARTIAL = 1
    FULL = 2

    def __add__(self, other: 'CompressionLevel') -> 'CompressionLevel':
        """
        Defines compression level of a composite compression controller, consist of
        two algorithms, where `self` is the compression level of the first algorithm
        and other - compression level of the second one.
            NONE    & NONE    = NONE
            PARTIAL & PARTIAL = PARTIAL
            FULL    & FULL    = FULL
            NONE    & PARTIAL = PARTIAL
            NONE    & FULL    = PARTIAL
            PARTIAL & FULL    = PARTIAL

        :param other: An instance of another compression level.
        :return: The common compression level of the two algorithms.
        """
        if self == other:
            return self
        return CompressionLevel.PARTIAL


class CompressionAlgorithmController:
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
        self._loss = CompressionLoss()
        self._scheduler = CompressionScheduler()

    @property
    def model(self) -> ModelType:
        return self._model

    @property
    def loss(self) -> CompressionLoss:
        return self._loss

    @property
    def scheduler(self) -> CompressionScheduler:
        return self._scheduler

    def compression_level(self) -> CompressionLevel:
        """
        Returns the compression level. Should be used on saving best checkpoints
        to distinguish between uncompressed, partially compressed, and fully
        compressed models.

        :return: The compression level of the target model.
        """
        return CompressionLevel.NONE

    def statistics(self, quickly_collected_only: bool = False) -> Dict[str, object]:
        """
        Returns a dictionary of printable statistics.

        :param quickly_collected_only: Enables collection of the statistics that
            don't take too much time to compute. Can be helpful for the case when
            need to keep track of statistics on each training batch/step/iteration.
        :return: A dictionary of printable statistics.
        """
        return self._loss.statistics(quickly_collected_only)

    def prepare_for_export(self) -> None:
        """
        Prepare the compressed model for deployment.
        """
        self._model = self.strip_model(self._model)

    def export_model(self, save_path: str, *args, **kwargs) -> None:
        """
        Used to export the compressed model for deployment. Makes method-specific
        preparations of the model, (e.g. removing auxiliary layers that were used
        for the model compression), then exports the model in the specified path.

        :param save_path: The path to export model.
        :param args: Advanced export options.
        :param kwargs: Advanced export options
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


class CompressionAlgorithmBuilder:
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

    def apply_to(self, model: ModelType) -> ModelType:
        """
        Applies algorithm-specific modifications to the model.

        :param model: The original uncompressed model.
        :return: The model with additional modifications necessary to enable
            algorithm-specific compression during fine-tuning.
        """
        transformation_layout = self.get_transformation_layout(model)
        return ModelTransformer(model, transformation_layout).transform()

    def build_controller(self, model: ModelType) -> CompressionAlgorithmController:
        """
        Builds `CompressionAlgorithmController` to handle the additional modules,
        parameters, and hooks inserted into the model to enable algorithm-specific
        compression.

        :param model: The model with additional modifications necessary to enable
            algorithm-specific compression during fine-tuning.
        :return: The instance of the `CompressionAlgorithmController`.
        """
        return CompressionAlgorithmController(model)

    def get_transformation_layout(self, model: ModelType) -> TransformationLayout:
        """
        Computes necessary model transformations to enable algorithm-specific
        compression.

        :param model: The original uncompressed model.
        :return: The instance of the `TransformationLayout` class containing
            a list of algorithm-specific modifications.
        """
        return TransformationLayout()
