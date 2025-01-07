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


"""
@package docstring
This package defines the API for the NNCF compression methods so that the user could
extend the existing algorithms.
"""
from abc import abstractmethod
from typing import Any, Dict, List, Tuple, TypeVar

import torch
from torch import nn

import nncf
from nncf.api.compression import CompressionLoss
from nncf.common.compression import BaseCompressionAlgorithmBuilder
from nncf.common.compression import BaseCompressionAlgorithmController
from nncf.common.graph import NNCFNodeName
from nncf.common.logging import nncf_logger
from nncf.common.scopes import check_scopes_in_graph
from nncf.common.scopes import should_consider_scope
from nncf.config import NNCFConfig
from nncf.torch.graph.transformations.layout import PTTransformationLayout
from nncf.torch.layers import NNCF_MODULES_DICT
from nncf.torch.layers import NNCF_WRAPPED_USER_MODULES_DICT
from nncf.torch.model_transformer import PTModelTransformer
from nncf.torch.nncf_network import NNCFNetwork

TModel = TypeVar("TModel")

DOMAIN_CUSTOM_OPS_NAME = "org.openvinotoolkit"


class PTCompressionLoss(nn.Module, CompressionLoss):
    """
    Used to calculate additional loss to be added to the base loss during the
    training process. It uses the model graph to measure variables and activations
    values of the layers during the loss construction. For example, the $L_0$-based
    sparsity algorithm calculates the number of non-zero weights in convolutional
    and fully-connected layers to construct the loss function.
    """

    def calculate(self) -> torch.Tensor:
        """
        Calculates the compression loss value.

        :return: The compression loss value.
        """
        return torch.zeros([])

    def forward(self) -> torch.Tensor:
        """
        Overriding  forward function of the base nn.Module class

        :return: The compression loss value.
        """
        return self.calculate()

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Loads the compression loss state.

        :param state: Output of `get_state()` method.
        """

    def get_state(self) -> None:
        """
        Returns the compression loss state.

        :return: The compression loss state.
        """


class PTCompressionAlgorithmController(BaseCompressionAlgorithmController):
    """
    Serves as a handle to the additional modules, parameters and hooks inserted
    into the original uncompressed model in order to enable algorithm-specific compression.
    Hosts entities that are to be used during the training process, such as compression scheduler and
    compression loss.
    """

    def distributed(self):
        """
        Should be called when distributed training with multiple training processes
        is going to be used (i.e. after the model is wrapped with DistributedDataParallel).
        Any special preparations for the algorithm to properly support distributed training
        should be made inside this function.
        """

    def prepare_for_export(self) -> None:
        # For Torch models no need to call strip_model
        pass


class PTCompressionAlgorithmBuilder(BaseCompressionAlgorithmBuilder):
    """
    Determines which modifications should be made to the original FP32 model in
    order to enable algorithm-specific compression during fine-tuning. Operates
    on an NNCFNetwork object wrapping a target PyTorch model (torch.nn.Module).
    """

    def __init__(self, config: NNCFConfig, should_init: bool = True):
        """
        Arguments:
          `config` - a dictionary that contains parameters of compression method
          `should_init` - if False, trainable parameter initialization will be skipped during building
        """
        super().__init__(config, should_init)
        self.compressed_nncf_module_names = self._nncf_module_types_to_compress()

    def apply_to(self, model: NNCFNetwork) -> NNCFNetwork:
        transformer = PTModelTransformer(model)
        transformation_layout = self.get_transformation_layout(model)
        transformed_model = transformer.transform(transformation_layout)

        if self.should_init:
            self.initialize(transformed_model)

        return transformed_model

    def get_transformation_layout(self, model: NNCFNetwork) -> PTTransformationLayout:
        """
        Applies algorithm-specific modifications to the model. Hooks to be executed during model
        forward operation may be registered using NNCFNetwork command insertion methods. Additional
        compression modules that are expected to be saved along with the network via torch.save should also be
        registered and added to the model here.
        :param model: An instance of NNCFNetwork for the algorithm to be applied to.
        :return: NNCFNetwork with algorithm-specific modifications applied
        """
        check_scopes_in_graph(
            model.nncf.get_original_graph(), self.ignored_scopes, self.target_scopes, self.validate_scopes
        )

        layout = self._get_transformation_layout(model)
        self._handle_frozen_layers(model)
        return layout

    @abstractmethod
    def _build_controller(self, model: TModel) -> PTCompressionAlgorithmController:
        """
        Simple implementation of building controller without setting builder state and loading controller's one.

        :param model: The model with additional modifications necessary to enable
            algorithm-specific compression during fine-tuning.
        :return: The instance of the `BaseCompressionAlgorithmController`.
        """

    def build_controller(self, model: TModel) -> PTCompressionAlgorithmController:
        """
        Builds `PTCompressionAlgorithmController` to handle the additional modules,
        parameters, and hooks inserted into the model to enable algorithm-specific
        compression. Registers the built controller in the model's NNCFNetworkInterface.

        :param model: The model with additional modifications necessary to enable
            algorithm-specific compression during fine-tuning.
        :return: The instance of the `PTCompressionAlgorithmController`.
        """
        ctrl = self._build_controller(model)
        if not isinstance(ctrl, PTCompressionAlgorithmController):
            raise nncf.InternalError(
                "Internal error: builder must create controller inherited from "
                "`PTCompressionAlgorithmController` class"
            )
        ctrl.set_builder_state_with_name(self.name, self.get_state())
        return ctrl

    def _get_state_without_name(self) -> Dict[str, Any]:
        """
        Implementation of get_state that returns state without builder name.

        :return: Returns a dictionary with Python data structures
            (dict, list, tuple, str, int, float, True, False, None) that represents state of the object.
        """
        return {}

    def _load_state_without_name(self, state_without_name: Dict[str, Any]):
        """
        Implementation of load state that takes state without builder name.

        :param state_without_name: Output of `_get_state_without_name()` method.
        """

    def _get_transformation_layout(self, target_model: NNCFNetwork) -> PTTransformationLayout:
        raise NotImplementedError()

    def _handle_frozen_layers(self, target_model: NNCFNetwork):
        scopes_of_frozen_layers = []
        for weighted_node in target_model.nncf.get_weighted_original_graph_nodes():
            should_be_considered = self._should_consider_scope(weighted_node.node_name)
            if not weighted_node.layer_attributes.weight_requires_grad and should_be_considered:
                scopes_of_frozen_layers.append(weighted_node.node_name)
        scopes_to_print = "\n".join(scopes_of_frozen_layers)
        if len(scopes_of_frozen_layers) > 0:
            is_allowed, reason = self._are_frozen_layers_allowed()
            if is_allowed:
                nncf_logger.warning(
                    f"{reason}, compressing them without tuning weights.\nFrozen layers:\n{scopes_to_print}"
                )
            else:
                raise nncf.InternalError(
                    f"{reason}.\n"
                    f"Please unfreeze them or put into the Ignored Scope.\n"
                    f"Frozen Layers:\n"
                    f"{scopes_to_print}"
                )

    def _should_consider_scope(self, node_name: NNCFNodeName) -> bool:
        return should_consider_scope(node_name, self.ignored_scopes, self.target_scopes)

    def _nncf_module_types_to_compress(self) -> List[str]:
        """
        Return list of NNCF module types which should be compressed by specific algorithm.
        As name of algorithm used the value set by decorator @Registry.register() or default one.
        :return: List of names of modules
        """
        filtered_nncf_module_names_list = []
        for module_cls in list(NNCF_MODULES_DICT) + list(NNCF_WRAPPED_USER_MODULES_DICT.values()):
            if self.name not in module_cls.ignored_algorithms:
                filtered_nncf_module_names_list.append(module_cls.__name__)
        return filtered_nncf_module_names_list

    def _are_frozen_layers_allowed(self) -> Tuple[bool, str]:
        algo_name = self.name.replace("_", " ")
        return False, f"Frozen layers are not allowed for {algo_name}"
