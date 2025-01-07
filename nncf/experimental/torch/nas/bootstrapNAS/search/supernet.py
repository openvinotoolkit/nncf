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

from typing import Any, Callable, Dict, List, Tuple, TypeVar

import torch

from nncf import NNCFConfig
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elasticity_controller import ElasticityController
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.multi_elasticity_handler import SubnetConfig
from nncf.experimental.torch.nas.bootstrapNAS.training.model_creator_helpers import resume_compression_from_state
from nncf.torch.checkpoint_loading import load_state
from nncf.torch.model_creation import create_nncf_network
from nncf.torch.nncf_network import NNCFNetwork

TModel = TypeVar("TModel")
ValFnType = Callable[[NNCFNetwork, Any], Any]


class TrainedSuperNet:
    """
    An interface for handling pre-trained super-networks. This class can be used to quickly implement
    third party solutions for subnetwork search on existing super-networks.
    """

    def __init__(self, elastic_ctrl: ElasticityController, nncf_network: NNCFNetwork):
        """
        Initializes the super-network interface.

        :param elastic_ctrl: Elasticity controller to activate subnetworks
        :param nncf_network: NNCFNetwork that wraps the original PyTorch model.
        """
        self._m_handler = elastic_ctrl.multi_elasticity_handler
        self._elasticity_ctrl = elastic_ctrl
        self._model = nncf_network

    @classmethod
    def from_checkpoint(
        cls,
        model: TModel,
        nncf_config: NNCFConfig,
        supernet_elasticity_path: str,
        supernet_weights_path: str,
    ) -> "TrainedSuperNet":
        """
        Loads existing super-network weights and elasticity information, and creates the SuperNetwork interface.

        :param model: base model that was used to create the super-network.
        :param nncf_config: configuration used to create the super-network.
        :param supernet_elasticity_path: path to file containing state information about the super-network.
        :param supernet_weights_path: trained weights to resume the super-network.
        :return: SuperNetwork with wrapped functionality.
        """
        nncf_network = create_nncf_network(model, nncf_config)
        compression_state = torch.load(supernet_elasticity_path, map_location=torch.device(nncf_config.device))
        model, elasticity_ctrl = resume_compression_from_state(nncf_network, compression_state)
        model_weights = torch.load(supernet_weights_path, map_location=torch.device(nncf_config.device))
        load_state(model, model_weights, is_resume=True)
        elasticity_ctrl.multi_elasticity_handler.activate_maximum_subnet()
        return TrainedSuperNet(elasticity_ctrl, model)

    def get_search_space(self) -> Dict:
        """
        :return: dictionary with possible values for elastic configurations.
        """
        return self._m_handler.get_search_space()

    def get_design_vars_info(self) -> Tuple[int, List[int]]:
        """
        :return: number of possible values in subnet configurations and
        the number of possible values for each elastic property.
        """
        self._m_handler.get_design_vars_info()

    def eval_subnet_with_design_vars(self, design_config: List, eval_fn: ValFnType, **kwargs) -> Any:
        """

        :return: the value produced by the user's function to evaluate the subnetwork.
        """
        self._m_handler.activate_subnet_for_config(self._m_handler.get_config_from_pymoo(design_config))
        return eval_fn(self._model, **kwargs)

    def eval_active_subnet(self, eval_fn: ValFnType, **kwargs) -> Any:
        """
        :param eval_fn: user's function to evaluate the active subnetwork.
        :return: value of the user's function used to evaluate the subnetwork.
        """
        return eval_fn(self._model, **kwargs)

    def eval_subnet(self, config: SubnetConfig, eval_fn: ValFnType, **kwargs) -> Any:
        """
        :param config: subnetwork configuration.
        :param eval_fn: user's function to evaluate the active subnetwork.
        :return: value of the user's function used to evaluate the subnetwork.
        """
        self.activate_config(config)
        return self.eval_active_subnet(eval_fn, **kwargs)

    def activate_config(self, config: SubnetConfig) -> None:
        """
        :param config: subnetwork configuration to activate.
        """
        self._m_handler.activate_subnet_for_config(config)

    def activate_maximal_subnet(self) -> None:
        """
        Activates the maximal subnetwork in the super-network.
        """
        self._m_handler.activate_maximum_subnet()

    def activate_minimal_subnet(self) -> None:
        """
        Activates the minimal subnetwork in the super-network.
        """
        self._m_handler.activate_minimum_subnet()

    def get_active_config(self) -> SubnetConfig:
        """
        :return: the active configuration.
        """
        return self._m_handler.get_active_config()

    def get_macs_for_active_config(self) -> float:
        """
        :return: MACs of active subnet.
        """
        return self._m_handler.count_flops_and_weights_for_active_subnet()[0] / 2e6

    def export_active_subnet_to_onnx(self, filename: str = "subnet") -> None:
        """
        Exports the active subnetwork to ONNX format.

        :param filename: name of the output file.
        """
        self._elasticity_ctrl.export_model(f"{filename}.onnx")

    def get_config_from_pymoo(self, pymoo_config: List) -> SubnetConfig:
        """
        Converts a Pymoo subnetwork configuration into a SubnetConfig.

        :param pymoo_config: subnetwork configuration in Pymoo format.
        :return: subnetwork configuration in SubnetConfig format.
        """
        return self._m_handler.get_config_from_pymoo(pymoo_config)

    def get_active_subnet(self) -> NNCFNetwork:
        """
        :return: the nncf network with the current active configuration.
        """
        return self._model
