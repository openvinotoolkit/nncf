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
import numpy
from copy import copy
from functools import partial
from typing import List, Tuple, Optional, TypeVar, Dict

import torch
from torch import nn

from nncf.config import NNCFConfig
from nncf.dynamic_graph.graph_tracer import create_mock_tensor
from nncf.graph.transformations.layout import PTTransformationLayout
from nncf.initialization import DataLoaderBNAdaptationRunner
from nncf.layers import NNCF_MODULES_DICT, NNCF_WRAPPED_USER_MODULES_DICT
from nncf.common.utils.logger import logger as nncf_logger
from nncf.nncf_network import NNCFNetwork
from nncf.nncf_network import PTModelTransformer
from nncf.structures import BNAdaptationInitArgs
from nncf.utils import should_consider_scope
from nncf.api.compression import CompressionAlgorithmBuilder
from nncf.api.compression import CompressionAlgorithmController
from nncf.api.compression import CompressionLoss

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

    def statistics(self, quickly_collected_only: bool = False) -> Dict[str, object]:
        """
        Returns a dictionary of printable statistics.

        :param quickly_collected_only: Enables collection of the statistics that
            don't take too much time to compute. Can be helpful for the case when
            need to keep track of statistics on each training batch/step/iteration.
        :return: A dictionary of printable statistics.
        """
        return {}


class PTCompressionAlgorithmController(CompressionAlgorithmController):
    """Serves as a handle to the additional modules, parameters and hooks inserted
    into the original uncompressed model in order to enable algorithm-specific compression.
    Hosts entities that are to be used during the training process, such as compression scheduler and
    compression loss."""


    def distributed(self):
        """
        Should be called when distributed training with multiple training processes
        is going to be used (i.e. after the model is wrapped with DistributedDataParallel).
        Any special preparations for the algorithm to properly support distributed training
        should be made inside this function.
        """

    def statistics(self, quickly_collected_only=False):
        """
        Returns a dictionary of printable statistics.

        :param quickly_collected_only: Enables collection the statistics that don't take
            too much time to compute. Can be helpful for the case when need to keep track
            statistics on each train batch/step/iteration.
        :return: A dictionary of printable statistics.
        """
        stats = super().statistics(quickly_collected_only)
        if hasattr(self._model, 'statistics'):
            stats.update(self._model.statistics(quickly_collected_only))
        return stats

    def run_batchnorm_adaptation(self, config):
        initializer_params = config.get("initializer", {})
        init_bn_adapt_config = initializer_params.get('batchnorm_adaptation', {})
        num_bn_adaptation_samples = init_bn_adapt_config.get('num_bn_adaptation_samples', 0)
        num_bn_forget_samples = init_bn_adapt_config.get('num_bn_forget_samples', 0)
        try:
            bn_adaptation_args = config.get_extra_struct(BNAdaptationInitArgs)
            has_bn_adapt_init_args = True
        except KeyError:
            has_bn_adapt_init_args = False

        if not init_bn_adapt_config:
            if has_bn_adapt_init_args:
                nncf_logger.warning("Enabling quantization batch norm adaptation with default parameters.")
                num_bn_adaptation_samples = 2000
                num_bn_forget_samples = 1000

        if num_bn_adaptation_samples < 0:
            raise AttributeError('Number of adaptation samples must be >= 0')
        if num_bn_adaptation_samples > 0:
            if not has_bn_adapt_init_args:
                nncf_logger.info(
                    'Could not run batchnorm adaptation '
                    'as the adaptation data loader is not provided as an extra struct. '
                    'Refer to `NNCFConfig.register_extra_structs` and the `BNAdaptationInitArgs` class')
                return
            batch_size = bn_adaptation_args.data_loader.batch_size
            num_bn_forget_steps = numpy.ceil(num_bn_forget_samples / batch_size)
            num_bn_adaptation_steps = numpy.ceil(num_bn_adaptation_samples / batch_size)
            bn_adaptation_runner = DataLoaderBNAdaptationRunner(self._model, bn_adaptation_args.device,
                                                                num_bn_forget_steps)
            bn_adaptation_runner.run(bn_adaptation_args.data_loader, num_bn_adaptation_steps)

    # pylint: disable=keyword-arg-before-vararg
    def export_model(self,
                     save_path: str,
                     input_names: Optional[List[str]] = None,
                     output_names: Optional[List[str]] = None,
                     *args, **kwargs) -> None:
        """
        Used to export the compressed model for inference into the ONNX format.
        Makes method-specific preparations of the model graph,
        (e.g. removing auxiliary layers that were used for the model compression),
        then exports the model and dumps it into the output file.
        Parameters:
            `save_path` - a path to the file for the exported model to be saved into.
            `input_names` - list of input tensors names (optional).
            `output_names` - list of output tensors names (optional).
            *args, **kwargs - if the model's `forward` requires additional parameters
            during export, specify these here.
        """
        self.prepare_for_export()
        model = self._model.eval().cpu()
        input_tensor_list = []
        for info in self._model.input_infos:
            single_batch_info = copy(info)
            input_shape = tuple([1] + list(info.shape)[1:])
            single_batch_info.shape = input_shape
            input_tensor_list.append(create_mock_tensor(single_batch_info, "cpu"))
        original_forward = model.forward
        model.forward = partial(model.forward, *args, **kwargs)
        # pylint:disable=unexpected-keyword-arg
        with torch.no_grad():
            # Should call this, otherwise the operations executed during export will end up in graph
            model.disable_dynamic_graph_building()
            torch.onnx.export(model, tuple(input_tensor_list),
                              save_path, input_names=input_names,
                              output_names=output_names,
                              enable_onnx_checker=False,
                              opset_version=10,
                              training=True)  # Do not fuse Conv+BN in ONNX. May cause dropout nodes to appear in ONNX
            model.enable_dynamic_graph_building()
        model.forward = original_forward


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
        transformation_layout = self.get_transformation_layout(model)
        transformer = PTModelTransformer(model, transformation_layout)
        transformed_model = transformer.transform()
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
        for scope, module in target_model.get_nncf_modules().items():
            if not module.weight.requires_grad:
                if should_consider_scope(str(scope), self.target_scopes, self.ignored_scopes):
                    scopes_of_frozen_layers.append(str(scope))
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


    def _should_consider_scope(self, scope_str: str) -> bool:
        return should_consider_scope(scope_str, self.target_scopes, self.ignored_scopes)

    def _nncf_module_types_to_compress(self) -> List[str]:
        """
        Return list of NNCF module types which should be compressed by specific algorithm.
        As name of algorithm used self._registered_name that set by decorator @nncf.registry_module.
        :return: List of names of modules
        """
        filtered_nncf_module_names_list = list()
        for module in list(NNCF_MODULES_DICT) + list(NNCF_WRAPPED_USER_MODULES_DICT.values()):
            if self._registered_name not in module.ignored_algorithms:
                filtered_nncf_module_names_list.append(module.__name__)
        return filtered_nncf_module_names_list

    def _are_frozen_layers_allowed(self) -> Tuple[bool, str]:
        algo_name = self._registered_name.replace('_', ' ')
        return False, f'Frozen layers are not allowed for {algo_name}'
