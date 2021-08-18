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

from abc import abstractmethod

from typing import Dict
from typing import Optional, List, Tuple, Any, TypeVar

from nncf import NNCFConfig
from nncf.common.schedulers import StubCompressionScheduler
from nncf.api.compression import CompressionAlgorithmBuilder
from nncf.api.compression import CompressionAlgorithmController
from nncf.common.utils.logger import logger as nncf_logger
from nncf.common.utils.registry import Registry
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import infer_backend_from_model
from nncf.config.extractors import extract_algo_specific_config

ModelType = TypeVar('ModelType')

NO_COMPRESSION_ALGORITHM_NAME = 'NoCompressionAlgorithm'


class BaseControllerStateNames:
    LOSS = 'loss_state'
    SCHEDULER = 'scheduler_state'
    COMPRESSION_STAGE = 'compression_stage'
    COMPRESSION_LEVEL = 'compression_level'


class BaseCompressionAlgorithmController(CompressionAlgorithmController):
    """
    Contains the implementation of the basic functionality of the compression controller.
    """

    BUILDER_STATE = 'builder_state'
    CONTROLLER_STATE = 'ctrl_state'

    def __init__(self, target_model: ModelType):
        """
        Initializes the internal state of the compression algorithm controller.

        :param target_model: The model with additional modifications necessary
            to enable algorithm-specific compression during fine-tuning built
            by the `CompressionAlgorithmBuilder`.
        """
        super().__init__(target_model)
        self._name = None
        self._builder_state = None
        self._state_names = BaseControllerStateNames()

    @property
    def name(self):
        if self._name is None:
            raise RuntimeError('Internal error: name of the controller is not set!')
        return self._name

    @property
    def compression_rate(self) -> float:
        return None

    @compression_rate.setter
    def compression_rate(self) -> float:
        pass

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
        self.prepare_for_export()
        backend = infer_backend_from_model(self.model)
        if backend is BackendType.TENSORFLOW:
            from nncf.tensorflow.exporter import TFExporter
            exporter = TFExporter(self.model, input_names, output_names, model_args)
        else:
            assert backend is BackendType.TORCH
            from nncf.torch.exporter import PTExporter
            exporter = PTExporter(self.model, input_names, output_names, model_args)
        exporter.export_model(save_path, save_format)

    def disable_scheduler(self) -> None:
        self._scheduler = StubCompressionScheduler()

    def set_builder_state_with_name(self, name: str, builder_state: Dict):
        """
        Sets state of the builder and the corresponding algorithm name. Should be called by the builder to set its
        state and registered algorithm key.

        :param name: algorithm name, the string that was used to register the builder
        :param builder_state: state of the builder
        """
        self._name = name
        self._builder_state = builder_state

    def load_state(self, state: Dict[str, Dict[str, Any]]) -> None:
        """
        Loads the compression controller state from the map of algorithm name to the dictionary with state attributes.

        :param state: map of the algorithm name to the dictionary with the corresponding state attributes.
        """
        if self.name in state:
            algo_state = state[self.name]
            if self._state_names.COMPRESSION_STAGE in state:
                if self.compression_stage() != state[self._state_names.COMPRESSION_STAGE]:
                    nncf_logger.warning('Current CompressionStage ({}) of the compression controller does '
                                        'not correspond to the value found in '
                                        'the checkpoint ({})'.format(self.compression_stage(),
                                                                     state[self._state_names.COMPRESSION_STAGE]))
            self.loss.load_state(algo_state[self._state_names.LOSS])
            self.scheduler.load_state(algo_state[self._state_names.SCHEDULER])

    def get_state(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns compression controller state, which is the map of the algorithm name to the dictionary with the
        corresponding state attributes.

        :return: The compression controller state.
        """
        return {
            self.name: {
                self._state_names.LOSS: self.loss.get_state(),
                self._state_names.SCHEDULER: self.scheduler.get_state(),
                self._state_names.COMPRESSION_STAGE: self.compression_stage()
            }
        }

    def get_compression_state(self) -> Dict[str, Any]:
        """
        Returns compression state - builder and controller state.
        This state should be used to resume compression via `compression_state` argument of `create_compressed_model`
        method.

        :return: The compression state.
        """
        if self._builder_state is None:
            raise RuntimeError('Internal error: builder state is not set for the controller')

        return {
            self.BUILDER_STATE: self._builder_state,
            self.CONTROLLER_STATE: self.get_state()
        }


class BaseCompressionAlgorithmBuilder(CompressionAlgorithmBuilder):
    """
    Contains the implementation of the basic functionality of a single compression algorithm builder.
    """

    def __init__(self, config: NNCFConfig, should_init: bool = True):
        """
        Initializes internal state of the compression algorithm builder

        :param config: The top-level NNCFConfig object (i.e. parsed from a .json and extended with
            all necessary objects required for compression such as initialization data loaders).
        :param should_init: If False, trainable parameter initialization will be
            skipped during building.
        """
        super().__init__()
        self.config = config
        self.should_init = should_init
        self._algo_config = self._get_algo_specific_config_section()

        self.ignored_scopes = self.config.get('ignored_scopes')

        if 'ignored_scopes' in self._algo_config:
            algo_ignored_scopes = self._algo_config['ignored_scopes']
            if self.ignored_scopes is not None:
                self.ignored_scopes.extend(algo_ignored_scopes)
            else:
                self.ignored_scopes = algo_ignored_scopes

        self._global_target_scopes = self.config.get('target_scopes')
        self.target_scopes = self._global_target_scopes
        if 'target_scopes' in self._algo_config:
            algo_target_scopes = self._algo_config['target_scopes']
            if self.target_scopes is None:
                self.target_scopes = algo_target_scopes

    def _get_algo_specific_config_section(self) -> Dict:
        return extract_algo_specific_config(self.config, self.name)

    @property
    def name(self) -> str:
        return getattr(self, Registry.REGISTERED_NAME_ATTR, 'NOT_REGISTERED_' + self.__class__.__name__)

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Initializes object from the state.

        :param state: Output of `get_state()` method.
        """
        if self.name in state:
            self._load_state_without_name(state[self.name])

    def get_state(self) -> Dict[str, Any]:
        """
        Returns a dictionary with Python data structures (dict, list, tuple, str, int, float, True, False, None) that
        represents state of the object.

        :return: state of the object
        """
        return {self.name: self._get_state_without_name()}

    @abstractmethod
    def _build_controller(self, model: ModelType) -> BaseCompressionAlgorithmController:
        """
        Simple implementation of building controller without setting builder state and loading controller's one.

        :param model: The model with additional modifications necessary to enable
            algorithm-specific compression during fine-tuning.
        :return: The instance of the `BaseCompressionAlgorithmController`.
        """

    def build_controller(self, model: ModelType) -> BaseCompressionAlgorithmController:
        """
        Builds `BaseCompressionAlgorithmController` to handle the additional modules,
        parameters, and hooks inserted into the model to enable algorithm-specific
        compression.

        :param model: The model with additional modifications necessary to enable
            algorithm-specific compression during fine-tuning.
        :return: The instance of the `BaseCompressionAlgorithmController`.
        """
        ctrl = self._build_controller(model)
        ctrl.set_builder_state_with_name(self.name, self.get_state())
        return ctrl

    @abstractmethod
    def _load_state_without_name(self, state_without_name: Dict[str, Any]):
        """
        Implementation of load state that takes state without builder name.

        :param state_without_name: Output of `_get_state_without_name()` method.
        """

    @abstractmethod
    def _get_state_without_name(self) -> Dict[str, Any]:
        """
        Implementation of get_state that returns state without builder name.

        :return: Returns a dictionary with Python data structures
            (dict, list, tuple, str, int, float, True, False, None) that represents state of the object.
        """
