"""
 Copyright (c) 2022 Intel Corporation
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
from nncf.torch.nas.bootstrapNAS.elasticity.elasticity_dim import ElasticityDim
from nncf.torch.nncf_network import NNCFNetwork

ELASTICITY_HANDLERS_MAP = Dict[ElasticityDim, 'ElasticityHandler']
ElasticSearchSpace = TypeVar('ElasticSearchSpace')
ElasticityConfig = TypeVar('ElasticityConfig')
ELASTICITY_BUILDERS = Registry('Elasticity builder', add_name_as_attr=True)


class ElasticityHandler(ABC):
    """
    An interface for handling elasticity in the network. The elasticity defines variable values in properties of the
    layers or the network, e.g. variable number of channels in the Conv or variable number of layers in the network.
    By applying elasticity it's possible to derive a smaller models (Subnets) that have some elements in common with
    the original model.
    The interface defines methods for activation Subnets.
    """

    @abstractmethod
    def get_active_config(self) -> ElasticityConfig:
        """
        Forms an elasticity configuration that describes currently activated Subnet

        :return: elasticity configuration
        """

    @abstractmethod
    def get_random_config(self) -> ElasticityConfig:
        """
        Forms an elasticity configuration that describes a Subnet with randomly chosen elastic values

        :return: elasticity configuration
        """

    @abstractmethod
    def get_minimum_config(self) -> ElasticityConfig:
        """
        Forms an elasticity configuration that describes a Subnet with minimum elastic values

        :return: elasticity configuration
        """

    @abstractmethod
    def get_maximum_config(self) -> ElasticityConfig:
        """
        Forms an elasticity configuration that describes a Subnet with maximum elastic values

        :return: elasticity configuration
        """

    @abstractmethod
    def activate_supernet(self) -> None:
        """
        Activates the Supernet - the original network to which elasticity was applied.
        The Supernet is usually a maximum Subnet. But it might be not the case if maximum value of elastic properties
        is limited for efficiency of search. For instance, a supernet may contain 32 convolutional channels,
        but maximum subnet can be restricted to have 24 channels, at maximum.
        """

    @abstractmethod
    def set_config(self, config: ElasticityConfig) -> None:
        """
        Activates a Subnet that corresponds to the given elasticity configuration

        :param config: elasticity configuration
        """

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        Returns a dictionary with Python data structures (dict, list, tuple, str, int, float, True, False, None) that
        represents state of the object.

        :return: state of the object
        """

    @abstractmethod
    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Initializes object from the state.

        :param state: Output of `get_state()` method.
        """

    def activate_random_subnet(self) -> None:
        """
        Activates a Subnet with random values of elastic properties.
        """
        config = self.get_random_config()
        self.set_config(config)

    def activate_minimum_subnet(self) -> None:
        """
        Activates a minimum Subnet that corresponds to the minimum values of elastic properties.
        """
        config = self.get_minimum_config()
        self.set_config(config)

    def activate_maximum_subnet(self) -> None:
        """
        Activates a maximum Subnet that corresponds to the maximum values of elastic properties.
        """
        config = self.get_maximum_config()
        self.set_config(config)


class SEHandlerStateNames:
    ACTIVE_CONFIG = 'active_config'


class SingleElasticityHandler(ElasticityHandler, ABC):
    """
    An interface for handling a single elasticity dimension in the network, e.g. elastic width or depth.
    """
    _state_names = SEHandlerStateNames

    @abstractmethod
    def get_search_space(self) -> ElasticSearchSpace:
        """
        :return: search space that can be produced by iterating over all elastic parameters
        """

    @abstractmethod
    def get_kwargs_for_flops_counting(self) -> Dict[str, Any]:
        """
        Provides arguments for counting flops of the currently activated subnet.
        :return: mapping of parameters to its values
        """

    @abstractmethod
    def get_transformation_commands(self) -> List[TransformationCommand]:
        """
        :return: transformation commands for introducing the elasticity to NNCFNetwork
        """

    @abstractmethod
    def resolve_conflicts_with_other_elasticities(self,
                                                  config: ElasticityConfig,
                                                  elasticity_handlers: ELASTICITY_HANDLERS_MAP) -> ElasticityConfig:
        """
        Resolves a conflict between the given elasticity config and active elasticity configs of the given handlers.
        For example, elastic width configuration may contradict to elastic depth one. When we activate some
        configuration in the Elastic Width Handler, i.e. define number of output channels for some layers, we
        change output shapes of the layers. Consequently, it affects the blocks that can be skipped by Elastic Depth
        Handler, because input and output shapes may not be identical now.

        :param config: elasticity configuration
        :param elasticity_handlers: map of elasticity dimension to elasticity handler
        :return: elasticity configuration without conflicts with other active configs of other elasticity handlers
        """

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Initializes object from the state.

        :param state: Output of `get_state()` method.
        """
        active_config = state[self._state_names.ACTIVE_CONFIG]
        self.set_config(active_config)

    def get_state(self) -> Dict[str, Any]:
        """
        Returns a dictionary with Python data structures (dict, list, tuple, str, int, float, True, False, None) that
        represents state of the object.

        :return: state of the object
        """
        active_config = self.get_active_config()
        return {
            self._state_names.ACTIVE_CONFIG: active_config,
        }


class SEHBuilderStateNames:
    ELASTICITY_PARAMS = 'elasticity_params'


class SingleElasticityBuilder:
    """
    Determines which modifications should be made to the original FP32 model in order to introduce elasticity
    to the model.
    """
    _state_names = SEHBuilderStateNames

    def __init__(self,
                 ignored_scopes: Optional[List[str]] = None,
                 target_scopes: Optional[List[str]] = None,
                 elasticity_params: Optional[Dict[str, Any]] = None):
        self._target_scopes = target_scopes
        self._ignored_scopes = ignored_scopes
        self._elasticity_params = {} if elasticity_params is None else elasticity_params

    @abstractmethod
    def build(self, target_model: NNCFNetwork) -> SingleElasticityHandler:
        """
        Creates modifications to the given NNCFNetwork for introducing elasticity and creates a handler object that
        can manipulate this elasticity.

        :param target_model: a target NNCFNetwork for adding modifications
        :return: a handler object that can manipulate the elasticity.
        """

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
        Initializes object from the state.

        :param state: Output of `get_state()` method.
        """
        elasticity_params_from_state = state[SingleElasticityBuilder._state_names.ELASTICITY_PARAMS]
        if self._elasticity_params and self._elasticity_params != elasticity_params_from_state:
            nncf_logger.warning('Elasticity parameters were provided in two places: on init and on loading '
                                'state. The one from state is taken by ignoring the ones from init.')
        self._elasticity_params = elasticity_params_from_state
