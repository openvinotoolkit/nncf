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
This package defines the API for the NNCF compression methods, so that the user could
extend the existing algorithms.
"""
import functools
import numpy
from copy import copy
from enum import Enum
from functools import partial
from typing import List

import torch
from torch import nn

from nncf.config import NNCFConfig
from nncf.dynamic_graph.graph_builder import create_mock_tensor
from nncf.initialization import DataLoaderBNAdaptationRunner
from nncf.layers import NNCF_MODULES_DICT, NNCF_WRAPPED_USER_MODULES_DICT
from nncf.nncf_logger import logger as nncf_logger
from nncf.nncf_network import NNCFNetwork, InsertionCommand
from nncf.structures import BNAdaptationInitArgs
from nncf.utils import should_consider_scope


DOMAIN_CUSTOM_OPS_NAME = "org.openvinotoolkit"


class CompressionLoss(nn.Module):
    """
    Used to calculate additional loss to be added to the base loss during the
    training process. It uses the model graph to measure variables and activations
    values of the layers during the loss construction. For example, the $L_0$-based
    sparsity algorithm calculates the number of non-zero weights in convolutional
    and fully-connected layers to construct the loss function.
    """

    def forward(self):
        """
        Returns the compression loss value.
        """
        return torch.zeros([])

    def statistics(self, quickly_collected_only=False):
        """
        Returns a dictionary of printable statistics.

        Args:
            quickly_collected_only: enables collection the statistics that don't take too much time to compute.
            Can be helpful for the case when need to keep track statistics on each train batch/step/iteration.
        """
        return {}


class CompressionScheduler:
    """
    Implements the logic of compression method control during the training process.
    May change the method hyperparameters in regards to the current training step or
    epoch. For example, the sparsity method can smoothly increase the sparsity rate
    over several epochs.
    """

    def __init__(self):
        self.current_step = -1 # Global step - count of all step during training
        self._steps_in_current_epoch = 0
        self._current_epoch = -1

    def step(self, next_step=None):
        """
        Should be called before each optimizer step during training.
        Arguments:
            `next_step` - specifies the initial "next" step which should be set now
        """
        self.current_step = self.current_step + 1 if next_step is None else next_step
        self._steps_in_current_epoch += 1

    def epoch_step(self, next_epoch=None):
        """
        Should be called before each training epoch.
        Arguments:
            `next` - specifies the initial "next" epoch which should be set now
        """
        if next_epoch is None:
            self._current_epoch += 1
        else:
            self._current_epoch = next_epoch
        self._steps_in_current_epoch = 0

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def state_dict(self):
        default_keys = {'current_step', '_current_epoch'}
        return {key: val for key, val in self.__dict__.items() if key in default_keys}

    def initialize(self):
        pass

    @property
    def current_epoch(self):
        return 0 if self._current_epoch == -1 else self._current_epoch

@functools.total_ordering
class CompressionLevel(Enum):
    NONE = 0
    PARTIAL = 1
    FULL = 2

    # pylint:disable=comparison-with-callable
    def __add__(self, other: 'CompressionLevel') -> 'CompressionLevel':
        """
        Defines compression level of a composite compression controller, consist of two algorithms, where `self` is
        compression level of first algorithm and other - compression level of second one.
            NONE    & NONE    = NONE
            PARTIAL & PARTIAL = PARTIAL
            FULL    & FULL    = FULL
            NONE    & PARTIAL = PARTIAL
            NONE    & FULL    = PARTIAL
            PARTIAL & FULL    = PARTIAL
        Args:
            other: instance of another compression level
        Returns:
            common compression level of two algorithms
        """
        if self.value == other.value:
            return self
        return CompressionLevel.PARTIAL

    def __lt__(self, other: 'CompressionLevel') -> bool:
        return self.value < other.value


class CompressionAlgorithmController:
    """Serves as a handle to the additional modules, parameters and hooks inserted
    into the original uncompressed model in order to enable algorithm-specific compression.
    Hosts entities that are to be used during the training process, such as compression scheduler and
    compression loss."""

    def __init__(self, target_model: NNCFNetwork):
        self._model = target_model
        self._loss = CompressionLoss()
        self._scheduler = CompressionScheduler()

    @property
    def loss(self):
        return self._loss

    @property
    def scheduler(self):
        return self._scheduler

    def distributed(self):
        """
        Should be called when distributed training with multiple training processes
        is going to be used (i.e. after the model is wrapped with DistributedDataParallel).
        Any special preparations for the algorithm to properly support distributed training
        should be made inside this function.
        """

    def compression_level(self) -> CompressionLevel:
        """
        Returns level of compression. Should be used on saving best checkpoints to distinguish between
        uncompressed, partially compressed and fully compressed models.
        """
        raise NotImplementedError()

    def statistics(self, quickly_collected_only=False):
        """
        Returns a dictionary of printable statistics.

        Args:
            quickly_collected_only: enables collection the statistics that don't take too much time to compute.
            Can be helpful for the case when need to keep track statistics on each train batch/step/iteration.
        """
        stats = self._loss.statistics()
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

    def prepare_for_export(self):
        pass

    # pylint: disable=keyword-arg-before-vararg
    def export_model(self, filename, input_names: List[str] = None, output_names: List[str] = None, *args, **kwargs):
        """
        Used to export the compressed model for inference into the ONNX format.
        Makes method-specific preparations of the model graph,
        (e.g. removing auxiliary layers that were used for the model compression),
        then exports the model and dumps it into the output file.
        Parameters:
            `filename` - a path to the file for the exported model to be saved into.
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
            torch.onnx.export(model, tuple(input_tensor_list),
                              filename, input_names=input_names,
                              output_names=output_names,
                              enable_onnx_checker=False,
                              opset_version=10,
                              training=True)  # Do not fuse Conv+BN in ONNX. May cause dropout nodes to appear in ONNX
        model.forward = original_forward


class CompressionAlgorithmBuilder:
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
        self.config = config
        self.should_init = should_init
        if not isinstance(self.config, list):
            self.ignored_scopes = self.config.get('ignored_scopes')
            self.target_scopes = self.config.get('target_scopes')
        self.compressed_nncf_module_names = self._nncf_module_types_to_compress()

    def apply_to(self, target_model: NNCFNetwork) -> NNCFNetwork:
        """
        Applies algorithm-specific modifications to the model. Hooks to be executed during model
        forward operation may be registered using NNCFNetwork command insertion methods. Additional
        compression modules that are expected to be saved along with the network via torch.save should also be
        registered and added to the model here.
        :param target_model: An instance of NNCFNetwork for the algorithm to be applied to.
        :return: NNCFNetwork with algorithm-specific modifications applied
        """
        self._model = target_model  # type: NNCFNetwork
        insertion_commands = self._apply_to(target_model)

        for command in insertion_commands:
            target_model.register_insertion_command(command)

        target_model.register_algorithm(self)
        return target_model

    def _apply_to(self, target_model: NNCFNetwork) -> List[InsertionCommand]:
        return []

    def build_controller(self, target_model: NNCFNetwork) -> CompressionAlgorithmController:
        """
        Should be called once the compressed model target_model is fully constructed (i.e. hooks are applied and
        modules are in place. Returns a CompressionAlgorithmController object containing information
        and references to the compressed model or specific modules thereof required for the corresponding compression
        scheduler operation or compression loss calculation.
        :param target_model: An instance of NNCFNetwork with current algorithm already applied
        :return: A CompressionAlgorithmController object.
        """

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


class StubCompressionScheduler(CompressionScheduler):

    def compression_level(self) -> CompressionLevel:
        return CompressionLevel.FULL
