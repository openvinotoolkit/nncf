"""
 Copyright (c) 2019-2021 Intel Corporation
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
from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import TypeVar

from nncf.common.graph.transformations.commands import TransformationCommand
from nncf.common.utils.logger import logger as nncf_logger
from nncf.common.utils.registry import Registry
from nncf.torch.nncf_network import NNCFNetwork

ElasticSearchSpace = TypeVar('ElasticSearchSpace')
ElasticConfig = TypeVar('ElasticConfig')
ELASTICITY_BUILDERS = Registry('Elasticity builder', add_name_as_attr=True)


class ElasticHandler(ABC):
    @abstractmethod
    def get_active_config(self) -> ElasticConfig:
        pass

    @abstractmethod
    def activate_random_subnet(self):
        pass

    @abstractmethod
    def activate_minimal_subnet(self):
        pass

    @abstractmethod
    def activate_maximal_subnet(self):
        pass

    @abstractmethod
    def activate_supernet(self):
        pass

    @abstractmethod
    def set_config(self, config: ElasticConfig):
        pass

    @abstractmethod
    def activate(self):
        pass

    @abstractmethod
    def deactivate(self):
        pass


class SEHBuilderStateNames:
    ELASTICITY_PARAMS = 'elasticity_params'


class SingleElasticityBuilder:
    _state_names = SEHBuilderStateNames

    def __init__(self,
                 ignored_scopes: Optional[List[str]] = None,
                 target_scopes: Optional[List[str]] = None,
                 elasticity_params: Optional[Dict[str, Any]] = None):
        self._target_scopes = target_scopes
        self._ignored_scopes = ignored_scopes
        self._elasticity_params = {} if elasticity_params is None else elasticity_params

    @abstractmethod
    def build(self, target_model: NNCFNetwork, **kwargs):
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        Returns a dictionary with Python data structures (dict, list, tuple, str, int, float, True, False, None) that
        represents state of the object.

        :return: state of the object
        """
        return {SingleElasticityBuilder._state_names.ELASTICITY_PARAMS: self._elasticity_params}

    @abstractmethod
    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Loads the compression loss state.

        :param state: Output of `get_state()` method.
        """
        elasticity_params_from_state = state[SingleElasticityBuilder._state_names.ELASTICITY_PARAMS]
        if self._elasticity_params and self._elasticity_params != elasticity_params_from_state:
            nncf_logger.warning('Elasticity parameters were provided in two places: on init and on loading '
                                'state. The one from state is taken by ignoring the ones from init.')
        self._elasticity_params = elasticity_params_from_state


class SEHandlerStateNames:
    ACTIVE_CONFIG = 'active_config'
    IS_ACTIVE = 'is_active'


class SingleElasticHandler(ElasticHandler, ABC):
    _state_names = SEHandlerStateNames

    def __init__(self):
        self._is_active = True

    @abstractmethod
    def get_search_space(self) -> ElasticSearchSpace:
        pass

    @abstractmethod
    def get_kwargs_for_flops_counting(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_transformation_commands(self) -> List[TransformationCommand]:
        pass

    @property
    def is_active(self):
        return self._is_active

    def activate(self):
        self._is_active = True

    def deactivate(self):
        self._is_active = False
        self.activate_supernet()

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Loads the compression controller state from the map of algorithm name to the dictionary with state attributes.

        :param state: map of the algorithm name to the dictionary with the corresponding state attributes.
        """
        active_config = state[self._state_names.ACTIVE_CONFIG]
        self.set_config(active_config)
        self._is_active = state[self._state_names.IS_ACTIVE]

    def get_state(self) -> Dict[str, Any]:
        """
        Returns compression controller state, which is the map of the algorithm name to the dictionary with the
        corresponding state attributes.

        :return: The compression controller state.
        """
        active_config = self.get_active_config()
        return {
            self._state_names.ACTIVE_CONFIG: active_config,
            self._state_names.IS_ACTIVE: self.is_active
        }
