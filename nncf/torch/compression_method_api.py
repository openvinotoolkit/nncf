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

from typing import List, Tuple, TypeVar, Dict

import torch
from torch import nn

from nncf.common.graph import NNCFNodeName
from nncf.config import NNCFConfig
from nncf.torch.graph.transformations.layout import PTTransformationLayout
from nncf.torch.layers import NNCF_MODULES_DICT, NNCF_WRAPPED_USER_MODULES_DICT
from nncf.common.utils.logger import logger as nncf_logger
from nncf.common.compression import BaseCompressionAlgorithmController
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.nncf_network import PTModelTransformer
from nncf.common.utils.helpers import should_consider_scope
from nncf.api.compression import CompressionAlgorithmBuilder
from nncf.api.compression import CompressionLoss
from nncf.api.compression import CompressionLevel


ModelType = TypeVar('ModelType')

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

    def load_state(self, state: Dict[str, object]) -> None:
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

    def load_state(self, state: Dict[str, object]) -> None:
        """
        Loads the compression controller state.

        :param state: Output of `get_state()` method.
        """
        self._check_loaded_compression_stage(state)
        if self.scheduler is not None:
            self.scheduler.load_state(state['scheduler'])

    def _check_loaded_compression_stage(self, state: Dict[str, object]) -> None:
        if 'compression_level' in state:
            if 'compression_stage' not in state:
                compression_level = state['compression_level']
                state['compression_stage'] = CompressionLevel.map_legacy_level_to_stage()[compression_level]
            else:
                nncf_logger.warning('Both CompressionStage and (legacy) CompressionLevel attributes '
                                    'are specified in the checkpoint. Proceeding with the value stored '
                                    'in CompressionStage')
        if 'compression_stage' in state and self.compression_stage() != state['compression_stage']:
            nncf_logger.warning('Current CompressionStage ({}) of the compression controller does '
                                'not correspond to the value found in '
                                'the checkpoint ({})'.format(self.compression_stage(),
                                                             state['compression_stage']))

    def get_state(self) -> Dict[str, object]:
        """
        Returns the compression controller state.

        :return: The compression controller state.
        """
        return {'scheduler': self.scheduler.get_state(),
                'compression_stage': self.compression_stage()}


class PTCompressionAlgorithmBuilder(CompressionAlgorithmBuilder):
    """
    Determines which modifications should be made to the original FP32 model in
    order to enable algorithm-specific compression during fine-tuning. Operates
    on an NNCFNetwork object wrapping a target PyTorch model (torch.nn.Module).
    """

    _registered_name: str = None  # Attribute will set by COMPRESSION_ALGORITHMS registry

    def __init__(self, config: NNCFConfig, should_init: bool = True):
        """
        Arguments:
          `config` - a dictionary that contains parameters of compression method
          `should_init` - if False, trainable parameter initialization will be skipped during building
        """
        super().__init__(config, should_init)
        self.ignored_scopes = None
        self.target_scopes = None
        if not isinstance(self.config, list):
            self.ignored_scopes = self.config.get('ignored_scopes')
            self.target_scopes = self.config.get('target_scopes')
        self.compressed_nncf_module_names = self._nncf_module_types_to_compress()

    def apply_to(self, model: NNCFNetwork) -> NNCFNetwork:
        transformer = PTModelTransformer(model)
        transformation_layout = self.get_transformation_layout(model)
        transformed_model = transformer.transform(transformation_layout)

        if self.should_init:
            self.initialize(transformed_model)

        return transformed_model

    def get_transformation_layout(self, target_model: NNCFNetwork) -> PTTransformationLayout:
        """
        Applies algorithm-specific modifications to the model. Hooks to be executed during model
        forward operation may be registered using NNCFNetwork command insertion methods. Additional
        compression modules that are expected to be saved along with the network via torch.save should also be
        registered and added to the model here.
        :param target_model: An instance of NNCFNetwork for the algorithm to be applied to.
        :return: NNCFNetwork with algorithm-specific modifications applied
        """
        layout = self._get_transformation_layout(target_model)
        self._handle_frozen_layers(target_model)
        return layout

    def _get_transformation_layout(self, target_model: NNCFNetwork) -> PTTransformationLayout:
        raise NotImplementedError()

    def _handle_frozen_layers(self, target_model: NNCFNetwork):
        scopes_of_frozen_layers = []
        for weighted_node in target_model.get_weighted_original_graph_nodes():
            if not weighted_node.layer_attributes.weight_requires_grad:
                if should_consider_scope(weighted_node.node_name, self.ignored_scopes, self.target_scopes):
                    scopes_of_frozen_layers.append(weighted_node.node_name)
        scopes_to_print = '\n'.join(scopes_of_frozen_layers)
        if len(scopes_of_frozen_layers) > 0:
            is_allowed, reason = self._are_frozen_layers_allowed()
            if is_allowed:
                nncf_logger.warning('{}, compressing them without tuning weights.\n'
                               'Frozen layers:\n'
                               '{}'.format(reason, scopes_to_print))
            else:
                raise RuntimeError(f'{reason}.\n'
                                   f'Please unfreeze them or put into the Ignored Scope.\n'
                                   f'Frozen Layers:\n'
                                   f'{scopes_to_print}')

    def _should_consider_scope(self, node_name: NNCFNodeName) -> bool:
        return should_consider_scope(node_name, self.ignored_scopes, self.target_scopes)

    def _nncf_module_types_to_compress(self) -> List[str]:
        """
        Return list of NNCF module types which should be compressed by specific algorithm.
        As name of algorithm used self._registered_name that set by decorator @nncf.registry_module.
        :return: List of names of modules
        """
        filtered_nncf_module_names_list = list()
        for module_cls in list(NNCF_MODULES_DICT) + list(NNCF_WRAPPED_USER_MODULES_DICT.values()):
            if self._registered_name not in module_cls.ignored_algorithms:
                filtered_nncf_module_names_list.append(module_cls.__name__)
        return filtered_nncf_module_names_list

    def _are_frozen_layers_allowed(self) -> Tuple[bool, str]:
        algo_name = self._registered_name.replace('_', ' ')
        return False, f'Frozen layers are not allowed for {algo_name}'
