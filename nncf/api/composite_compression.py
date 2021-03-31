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

from typing import Any, Dict, List, Optional, TypeVar

from nncf import NNCFConfig
from nncf.api.compression import CompressionLevel
from nncf.api.compression import CompressionLoss
from nncf.api.compression import CompressionScheduler
from nncf.api.compression import CompressionAlgorithmBuilder
from nncf.api.compression import CompressionAlgorithmController
from nncf.common.graph.transformations.layout import TransformationLayout

ModelType = TypeVar('ModelType')


class CompositeCompressionLoss(CompressionLoss):
    """
    The `CompositeCompressionLoss` class stores a group of `CompressionLoss`
    instances as a list of children that are treated the same way as a single
    `CompressionLoss` instance.
    """

    def __init__(self):
        super().__init__()
        self._child_losses = []

    @property
    def child_losses(self) -> List[CompressionLoss]:
        return self._child_losses

    def add(self, child_loss: CompressionLoss) -> None:
        """
        Add `CompressionLoss` instance to the list of children.

        :param child_loss: A `CompressionLoss` instance.
        """
        self._child_losses.append(child_loss)

    def calculate(self) -> Any:
        """
        Traverses through all children and calculates the total compression
        loss value.

        :return: The compression loss value.
        """
        result_loss = 0
        for loss in self._child_losses:
            result_loss += loss()
        return result_loss

    def statistics(self, quickly_collected_only: bool = False) -> Dict[str, object]:
        """
        Traverses through all children and returns a sum-up dictionary of
        printable statistics.

        :param quickly_collected_only: Enables collection of the statistics that
            don't take too much time to compute. Can be helpful for the case when
            need to keep track of statistics on each training batch/step/iteration.
        :return: A dictionary of printable statistics.
        """
        stats = {}
        for loss in self._child_losses:
            stats.update(loss.statistics(quickly_collected_only))
        return stats


class CompositeCompressionScheduler(CompressionScheduler):
    """
    The `CompositeCompressionScheduler` class stores a group of `CompressionScheduler`
    instances as a list of children that are treated the same way as a single
    `CompressionScheduler` instance.
    """

    def __init__(self):
        super().__init__()
        self._child_schedulers = []

    @property
    def child_schedulers(self) -> List[CompressionScheduler]:
        return self._child_schedulers

    def add(self, child_scheduler: CompressionScheduler) -> None:
        """
        Add `CompressionScheduler` instance to the list of children.

        :param child_scheduler: A `CompressionScheduler` instance.
        """
        self._child_schedulers.append(child_scheduler)

    def step(self, next_step: Optional[int] = None) -> None:
        """
        Calls step() method for all children.

        :param next_step: The global step index for which the compression scheduler
            will update the state of the compression method.
        """
        super().step(next_step)
        for scheduler in self._child_schedulers:
            scheduler.step(next_step)

    def epoch_step(self, next_epoch: Optional[int] = None) -> None:
        """
        Calls epoch_step() method for all children.

        :param next_epoch: The epoch index for which the compression scheduler
            will update the state of the compression method.
        """
        super().epoch_step(next_epoch)
        for scheduler in self._child_schedulers:
            scheduler.epoch_step(next_epoch)

    def load_state(self, state: List[dict]) -> None:
        """
        Calls `load_state()` method for all children.

        :param state: Output of `get_state()` method.
        """
        for child_scheduler, child_state in zip(self._child_schedulers, state):
            child_scheduler.load_state(child_state)

    def get_state(self) -> List[dict]:
        """
        Returns the composite compression scheduler state. This state contains
        the state of all children.

        :return: The composite compression scheduler state.
        """
        composite_state = []
        for child_scheduler in self._child_schedulers:
            composite_state.append(child_scheduler.get_state())
        return composite_state


class CompositeCompressionAlgorithmController(CompressionAlgorithmController):
    """
    The `CompositeCompressionAlgorithmController` class stores a group of
    `CompressionAlgorithmController` instances as a list of children that are
    treated the same way as a single `CompressionAlgorithmController` instance.
    """

    def __init__(self, target_model: ModelType):
        """
        Initializes the internal state of the composite compression algorithm
        controller.

        :param target_model: The model with additional modifications necessary
            to enable algorithm-specific compression during fine-tuning built
            by the `CompressionAlgorithmBuilder`.
        """
        super().__init__(target_model)
        self._child_ctrls = []
        self._loss = CompositeCompressionLoss()
        self._scheduler = CompositeCompressionScheduler()

    @property
    def child_ctrls(self) -> List[CompressionAlgorithmController]:
        return self._child_ctrls

    def add(self, child_ctrl: CompressionAlgorithmController) -> None:
        """
        Add `CompressionAlgorithmController` instance to the list of children.

        :param child_ctrl: A `CompressionAlgorithmController` instance.
        """
        if child_ctrl.model is not self.model:
            raise RuntimeError('Cannot create a composite controller '
                               'from controllers belonging to different models!')

        self._child_ctrls.append(child_ctrl)
        self._loss.add(child_ctrl.loss)
        self._scheduler.add(child_ctrl.scheduler)

    def compression_level(self) -> CompressionLevel:
        """
        Returns the compression level. Should be used on saving best checkpoints
        to distinguish between uncompressed, partially compressed, and fully
        compressed models.

        :return: The compression level of the target model.
        """
        if not self.child_ctrls:
            return CompressionLevel.NONE
        result = None
        for ctrl in self.child_ctrls:
            current_level = ctrl.compression_level()
            if not result:
                result = current_level
            else:
                result += current_level
        return result

    def statistics(self, quickly_collected_only: bool = False) -> Dict[str, object]:
        """
        Traverses through all children and returns a sum-up dictionary of
        printable statistics.

        :param quickly_collected_only: Enables collection of the statistics that
            don't take too much time to compute. Can be helpful for the case when
            need to keep track of statistics on each training batch/step/iteration.
        :return: A dictionary of printable statistics.
        """
        stats = {}
        for ctrl in self.child_ctrls:
            stats.update(ctrl.statistics(quickly_collected_only))
        return stats

    def prepare_for_export(self) -> None:
        """
        Prepare the compressed model for deployment.
        """
        stripped_model = self._model
        for ctrl in self.child_ctrls:
            stripped_model = ctrl.strip_model(stripped_model)
        self._model = stripped_model


class CompositeCompressionAlgorithmBuilder(CompressionAlgorithmBuilder):
    """
    The `CompositeCompressionAlgorithmBuilder` class stores a group of
    `CompressionAlgorithmBuilder` instances as a list of children that are
    treated the same way as a single `CompressionAlgorithmBuilder` instance.
    """

    def __init__(self, config: NNCFConfig, should_init: bool = True):
        """
        Initializes internal state of the composite compression algorithm builder

        :param config: The dictionary that contains parameters of the compression
            methods.
        :param should_init: If False, trainable parameter initialization will be
            skipped during building.
        """
        super().__init__(config, should_init)
        self._child_builders = []

    @property
    def child_builders(self) -> List[CompressionAlgorithmBuilder]:
        return self._child_builders

    def build_controller(self, model: ModelType) -> CompositeCompressionAlgorithmController:
        """
        Builds `CompositeCompressionAlgorithmController` to handle the additional
        modules, parameters, and hooks inserted into the model to enable
        algorithm-specific compression.

        :param model: The model with additional modifications necessary to enable
         algorithm-specific compression during fine-tuning.
        :return: The instance of the `CompositeCompressionAlgorithmController`.
        """
        composite_ctrl = CompositeCompressionAlgorithmController(model)
        for builder in self.child_builders:
            composite_ctrl.add(builder.build_controller(model))
        return composite_ctrl

    def get_transformation_layout(self, model: ModelType) -> TransformationLayout:
        """
        Computes necessary model transformations to enable algorithm-specific
        compression.

        :param model: The original uncompressed model.
        :return: The instance of the `TransformationLayout` class containing
            a list of algorithm-specific modifications.
        """
        transformations = TransformationLayout()
        for builder in self.child_builders:
            transformations.update(builder.get_transformation_layout(model))
        return transformations
