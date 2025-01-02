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

from typing import Any, Dict, List, Optional, Tuple

import nncf
from nncf import NNCFConfig
from nncf.api.compression import CompressionAlgorithmBuilder
from nncf.api.compression import CompressionAlgorithmController
from nncf.api.compression import CompressionLoss
from nncf.api.compression import CompressionScheduler
from nncf.api.compression import CompressionStage
from nncf.api.compression import TModel
from nncf.common.statistics import NNCFStatistics
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import copy_model
from nncf.common.utils.backend import get_backend


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

    def load_state(self, state: List[Dict[str, Any]]) -> None:
        """
        Loads the composite compression loss state.

        :param state: Output of `get_state()` method.
        """
        for child_loss, child_state in zip(self._child_losses, state):
            child_loss.load_state(child_state)

    def get_state(self) -> List[Dict[str, Any]]:
        """
        Returns the composite compression loss state.

        :return: The composite compression loss state.
        """
        composite_state = []
        for child_loss in self.child_losses:
            composite_state.append(child_loss.get_state())
        return composite_state

    def calculate(self, *args, **kwargs) -> Any:
        """
        Traverses through all children and calculates the total compression
        loss value.

        :return: The compression loss value.
        """

        if len(self._child_losses) == 0:
            raise nncf.InternalError("Cannot calculate the loss value because the number of child loss is 0.")

        result_loss = 0
        for loss in self._child_losses:
            result_loss += loss()
        return result_loss


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

    def load_state(self, state: List[Dict[str, Any]]) -> None:
        """
        Calls `load_state()` method for all children.

        :param state: Output of `get_state()` method.
        """
        for child_scheduler, child_state in zip(self._child_schedulers, state):
            child_scheduler.load_state(child_state)

    def get_state(self) -> List[Dict[str, Any]]:
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

    BUILDER_STATE = "builder_state"
    CONTROLLER_STATE = "ctrl_state"

    def __init__(self, target_model: TModel):
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
        self._builder_state = None
        self._name = None

    @property
    def loss(self) -> CompressionLoss:
        return self._loss

    @property
    def scheduler(self) -> CompressionScheduler:
        return self._scheduler

    @property
    def child_ctrls(self) -> List[CompressionAlgorithmController]:
        return self._child_ctrls

    @property
    def name(self) -> str:
        raise self._name

    def add(self, child_ctrl: CompressionAlgorithmController) -> None:
        """
        Add `CompressionAlgorithmController` instance to the list of children.

        :param child_ctrl: A `CompressionAlgorithmController` instance.
        """
        if child_ctrl.model is not self.model:
            raise nncf.InternalError(
                "Cannot create a composite controller from controllers belonging to different models!"
            )

        self._child_ctrls.append(child_ctrl)
        self._loss.add(child_ctrl.loss)
        self._scheduler.add(child_ctrl.scheduler)

    def compression_stage(self) -> CompressionStage:
        """
        Returns the compression stage. Should be used on saving best checkpoints
        to distinguish between uncompressed, partially compressed, and fully
        compressed models.

        :return: The compression stage of the target model.
        """
        if not self.child_ctrls:
            return CompressionStage.UNCOMPRESSED
        result = None
        for ctrl in self.child_ctrls:
            current_level = ctrl.compression_stage()
            if result is None:
                result = current_level
            else:
                result += current_level
        return result

    def load_state(self, state: Dict[str, Dict[str, Any]]) -> None:
        """
        Loads the composite compression controller state from the map of algorithm name to the dictionary with state
        attributes.

        :param state: map of the algorithm name to the dictionary with the corresponding state attributes.
        """
        for ctrl in self.child_ctrls:
            ctrl.load_state(state)

    def get_state(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns composite compression controller state, which is the map of the algorithm name to the dictionary with
        the corresponding state attributes. This state contains the state of all children.

        :return: The composite compression controller state.
        """
        result = {}
        for ctrl in self.child_ctrls:
            result.update(ctrl.get_state())
        return result

    def statistics(self, quickly_collected_only: bool = False) -> NNCFStatistics:
        """
        Returns a `NNCFStatistics` class instance.

        :param quickly_collected_only: Enables collection of the statistics that
            don't take too much time to compute. Can be helpful for the case when
            need to keep track of statistics on each training batch/step/iteration.
        :return: A `NNCFStatistics` class instance.
        """
        nncf_stats = NNCFStatistics()

        for ctrl in self.child_ctrls:
            ctrl_stats = ctrl.statistics(quickly_collected_only)
            for algorithm_name, stats in ctrl_stats:
                nncf_stats.register(algorithm_name, stats)

        return nncf_stats

    def prepare_for_export(self) -> None:
        """
        Prepare the compressed model for deployment.
        """
        stripped_model = self._model
        for ctrl in self.child_ctrls:
            stripped_model = ctrl.strip_model(stripped_model)
        self._model = stripped_model

    def strip(self, do_copy: bool = True) -> TModel:
        model = self.model
        if do_copy:
            model = copy_model(model)
        for ctrl in self.child_ctrls:
            model = ctrl.strip_model(model, do_copy=False)
        return model

    @property
    def compression_rate(self) -> float:
        raise NotImplementedError

    @compression_rate.setter
    def compression_rate(self, compression_rate: float) -> None:
        raise NotImplementedError

    @property
    def maximal_compression_rate(self) -> float:
        return min(child_ctrl.maximal_compression_rate for child_ctrl in self.child_ctrls)

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
                - (a, b, {'x': None, 'y': y}) for positional and keyword arguments.
                - (a, b, {}) for positional arguments only.
                - ({'x': None, 'y': y},) for keyword arguments only.
        """
        self.prepare_for_export()
        backend = get_backend(self.model)
        if backend is BackendType.TENSORFLOW:
            from nncf.tensorflow.exporter import TFExporter

            exporter = TFExporter(self.model, input_names, output_names, model_args)
        else:
            assert backend is BackendType.TORCH
            from nncf.torch.exporter import PTExporter

            exporter = PTExporter(self.model, input_names, output_names, model_args)
        if save_format is not None:
            exporter.export_model(save_path, save_format)
        else:
            exporter.export_model(save_path)

    def disable_scheduler(self) -> None:
        self._scheduler = CompositeCompressionScheduler()
        for ctrl in self.child_ctrls:
            ctrl.disable_scheduler()
            self._scheduler.add(ctrl.scheduler)

    def get_compression_state(self) -> Dict[str, Any]:
        if self._builder_state is None:
            raise nncf.InternalError("Internal error: builder state is not set for the controller")

        return {self.BUILDER_STATE: self._builder_state, self.CONTROLLER_STATE: self.get_state()}

    def set_builder_state_with_name(self, name: str, builder_state: Dict):
        """
        Sets state of the builder and the corresponding algorithm name. Should be called by the builder to set its
        state and registered algorithm key.

        :param name: algorithm name, the string that was used to register the builder
        :param builder_state: state of the builder
        """
        self._name = name
        self._builder_state = builder_state


class CompositeCompressionAlgorithmBuilder(CompressionAlgorithmBuilder):
    """
    The `CompositeCompressionAlgorithmBuilder` class stores a group of
    `CompressionAlgorithmBuilder` instances as a list of children that are
    treated the same way as a single `CompressionAlgorithmBuilder` instance.
    """

    def __init__(self, config: NNCFConfig, should_init: bool = True):
        """
        Initializes internal state of the composite compression algorithm builder

        :param config: The top-level NNCFConfig object (i.e. parsed from a .json and extended with
            all necessary objects required for compression such as initialization data loaders).
        :param should_init: If False, trainable parameter initialization will be
            skipped during building.
        """
        self._config = config
        self.should_init = should_init
        self._child_builders = []

    def _get_algo_specific_config_section(self) -> Dict:
        return {}

    @property
    def child_builders(self) -> List[CompressionAlgorithmBuilder]:
        return self._child_builders

    def load_state(self, state: Dict[str, Dict]) -> None:
        """
        Loads the compression builder state of children

        :param state: Output of `get_state()` method.
        """
        for builder in self.child_builders:
            builder.load_state(state)

    def get_state(self) -> Dict[str, Dict]:
        """
        Returns the composite compression builder state. This state contains
        the state of all children.

        :return: The composite compression builder state.
        """
        result = {}
        for builder in self.child_builders:
            result.update(builder.get_state())
        return result
