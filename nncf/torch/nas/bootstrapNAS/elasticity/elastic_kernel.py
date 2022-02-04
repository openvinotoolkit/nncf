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
import random
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter

from nncf.common.graph import NNCFNode
from nncf.common.graph import NNCFNodeName
from nncf.common.graph.layer_attributes import ConvolutionLayerAttributes
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationCommand
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.common.utils.logger import logger as nncf_logger
from nncf.torch.graph.transformations.commands import PTInsertionCommand
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.layers import NNCFConv2d
from nncf.torch.module_operations import UpdateInputs
from nncf.torch.module_operations import UpdatePadding
from nncf.torch.module_operations import UpdateWeight
from nncf.torch.nas.bootstrapNAS.elasticity.base_handler import ELASTICITY_BUILDERS
from nncf.torch.nas.bootstrapNAS.elasticity.base_handler import ELASTICITY_HANDLERS_MAP
from nncf.torch.nas.bootstrapNAS.elasticity.base_handler import SingleElasticityHandler
from nncf.torch.nas.bootstrapNAS.elasticity.base_handler import SingleElasticityBuilder
from nncf.torch.nas.bootstrapNAS.elasticity.elasticity_dim import ElasticityDim
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.utils import is_tracing_state
from nncf.torch.utils import no_jit_trace

kernel_size_type = int
ElasticKernelConfig = List[kernel_size_type]  # list of kernel sizes per layer
ElasticKernelSearchSpace = List[List[kernel_size_type]]


class ElasticKernelOp:
    def __init__(self, *args, max_kernel_size: kernel_size_type, node_name: str, **kwargs):
        super().__init__(*args, **kwargs)
        self._kernel_size_list = []
        self._active_kernel_size = max_kernel_size
        self._max_kernel_size = max_kernel_size
        self._node_name = node_name

    def get_active_kernel_size(self) -> kernel_size_type:
        return self._active_kernel_size

    def set_active_kernel_size(self, kernel_size: kernel_size_type):
        if kernel_size is None or kernel_size > self.max_kernel_size or kernel_size < 1:
            raise AttributeError('Invalid kernel size={} in scope={}.\nIt should be within the range: [1, {}]'.format(
                kernel_size, self.node_name, self.max_kernel_size))

        self._active_kernel_size = kernel_size

    @property
    def kernel_size_list(self):
        return self._kernel_size_list

    @property
    def node_name(self):
        return self._node_name

    @property
    def max_kernel_size(self):
        return self._max_kernel_size


class ElasticKernelConv2DOp(ElasticKernelOp, nn.Module):
    def __init__(self, max_kernel_size, node_name, elastic_kernel_params):
        super().__init__(max_kernel_size=max_kernel_size, node_name=node_name)
        self._max_num_params = elastic_kernel_params.get('max_num_kernels', -1)
        # Create kernel_size_list based on max module kernel size
        self._kernel_size_list = self.generate_kernel_size_list(max_kernel_size)
        self._ks_set = list(set(self.kernel_size_list))
        self._ks_set.sort()

        scale_params = {}
        for i in range(len(self._ks_set) - 1):
            ks_small = self._ks_set[i]
            ks_larger = self._ks_set[i + 1]
            param_name = '%dto%d' % (ks_larger, ks_small)
            # noinspection PyArgumentList
            scale_params['%s_matrix' % param_name] = Parameter(torch.eye(ks_small ** 2))
        for name, param in scale_params.items():
            self.register_parameter(name, param)

    def generate_kernel_size_list(self, max_kernel_size):
        assert max_kernel_size % 2 > 0, 'kernel size should be odd number'
        if max_kernel_size == 1:
            return [1]
        kernel = max_kernel_size
        ks_list = []
        while kernel > 1:
            ks_list.append(kernel)
            kernel -= 2
            if self._max_num_params == len(ks_list):
                break
        return ks_list

    def forward(self, weight):
        kernel_size = self.get_active_kernel_size()
        nncf_logger.debug('Conv2d with active kernel size={} in scope={}'.format(kernel_size, self.node_name))

        result = weight
        if is_tracing_state():
            with no_jit_trace():
                if kernel_size > 1:
                    result = self._get_active_filter(kernel_size, weight).detach()
        else:
            if kernel_size > 1:
                result = self._get_active_filter(kernel_size, weight)
        return result

    def set_active_kernel_size(self, kernel_size):
        nncf_logger.debug('set active elastic_kernel={} in scope={}'.format(kernel_size, self.node_name))
        assert kernel_size % 2 > 0, 'kernel size should be odd number'
        if kernel_size not in self.kernel_size_list and kernel_size != self.max_kernel_size:
            raise ValueError(
                'invalid kernel size to set. Should be a number in {}'.format(self.kernel_size_list))
        super().set_active_kernel_size(kernel_size)

    @staticmethod
    def _sub_filter_start_end(kernel_size, sub_kernel_size):
        center = kernel_size // 2
        dev = sub_kernel_size // 2
        start, end = center - dev, center + dev + 1
        assert end - start == sub_kernel_size
        return start, end

    def _get_active_filter(self, kernel_size, weight):
        max_kernel_size = max(self.kernel_size_list)
        start, end = self._sub_filter_start_end(max_kernel_size, kernel_size)
        filters = weight[:, :, start:end, start:end]
        if kernel_size < max_kernel_size:
            start_filter = weight  # start with max kernel
            for i in range(len(self._ks_set) - 1, 0, -1):
                src_ks = self._ks_set[i]
                if src_ks <= kernel_size:
                    break
                target_ks = self._ks_set[i - 1]
                start, end = self._sub_filter_start_end(src_ks, target_ks)
                _input_filter = start_filter[:, :, start:end, start:end]
                _input_filter = _input_filter.contiguous()
                _input_filter = _input_filter.view(_input_filter.size(0), _input_filter.size(1), -1)
                _input_filter = _input_filter.view(-1, _input_filter.size(2))
                _input_filter = F.linear(
                    _input_filter, self.__getattr__('%dto%d_matrix' % (src_ks, target_ks)),
                )
                _input_filter = _input_filter.view(filters.size(0), filters.size(1), target_ks ** 2)
                _input_filter = _input_filter.view(filters.size(0), filters.size(1), target_ks, target_ks)
                start_filter = _input_filter
            filters = start_filter
        return filters


class ElasticKernelPaddingAdjustment:
    def __init__(self, elastic_k_w_op: ElasticKernelConv2DOp):
        self._elastic_k_w_op = elastic_k_w_op

    def __call__(self, previous_padding) -> int:
        return self._elastic_k_w_op.get_active_kernel_size() // 2


class ElasticKernelInputForExternalPadding:
    def __init__(self, elastic_k_w_op: ElasticKernelConv2DOp, max_kernel_size):
        self._elastic_k_w_op = elastic_k_w_op
        self._max_kernel_size = max_kernel_size

    def __call__(self, x) -> int:
        active_kernel_size = self._elastic_k_w_op.get_active_kernel_size()
        diff = (self._max_kernel_size - active_kernel_size) // 2
        h = x.size(2)
        w = x.size(3)
        new_x = x[:, :, diff:h - diff, diff:w - diff]
        return new_x


class ElasticKernelHandler(SingleElasticityHandler):
    """
    An interface for handling elastic kernel dimension in the network, i.e. define size of kernels in the conv layers.
    """
    def __init__(self,
                 elastic_kernel_ops: List[ElasticKernelOp],
                 transformation_commands: List[TransformationCommand]):
        super().__init__()
        self._elastic_kernel_ops = elastic_kernel_ops
        self._transformation_commands = transformation_commands

    def get_transformation_commands(self) -> List[TransformationCommand]:
        """
        :return: transformation commands for introducing the elasticity to NNCFNetwork
        """
        return self._transformation_commands

    def get_search_space(self) -> ElasticKernelSearchSpace:
        return self._collect_ops_data_by_selection_rule(lambda op: op.kernel_size_list)

    def get_active_config(self) -> ElasticKernelConfig:
        """
        Forms an elasticity configuration that describes currently activated Subnet

        :return: list of kernel sizes per layer
        """
        return self._collect_ops_data_by_selection_rule(lambda op: op.get_active_kernel_size())

    def get_random_config(self) -> ElasticKernelConfig:
        """
        Forms an elasticity configuration that describes a Subnet with randomly chosen elastic kernels

        :return: list of kernel sizes per layer
        """
        config = self._collect_ops_data_by_selection_rule(
            lambda op: op.kernel_size_list[random.randrange(0, len(op.kernel_size_list))]
        )
        return config

    def get_minimum_config(self) -> ElasticKernelConfig:
        """
        Forms an elasticity configuration that describes a Subnet with minimum elastic kernels

        :return: list of kernel sizes per layer
        """
        return self._collect_ops_data_by_selection_rule(lambda op: min(op.kernel_size_list))

    def get_maximum_config(self) -> ElasticKernelConfig:
        """
        Forms an elasticity configuration that describes a Subnet with maximum elastic kernels

        :return: list of kernel sizes per layer
        """
        return self._collect_ops_data_by_selection_rule(lambda op: max(op.kernel_size_list))

    def activate_supernet(self) -> None:
        """
        Activates the Supernet - the original network to which elasticity was applied.
        """
        supernet_config = self._collect_ops_data_by_selection_rule(lambda op: op.max_kernel_size)
        self.set_config(supernet_config)

    def set_config(self, config: ElasticKernelConfig) -> None:
        """
        Activates a Subnet that corresponds to the given elasticity configuration

        :return: list of kernel sizes per layer
        """
        for op, ks in zip(self._elastic_kernel_ops, config):
            op.set_active_kernel_size(ks)

    def get_kwargs_for_flops_counting(self) -> Dict[str, Any]:
        active_kernel_sizes = {}
        for elastic_kernel_op in self._elastic_kernel_ops:
            active_kernel_size = elastic_kernel_op.get_active_kernel_size()
            active_kernel_sizes[elastic_kernel_op.node_name] = (active_kernel_size, active_kernel_size)
        return {'kernel_sizes': active_kernel_sizes}

    def resolve_conflicts_with_other_elasticities(self,
                                                  config: ElasticKernelConfig,
                                                  elasticity_handlers: ELASTICITY_HANDLERS_MAP) -> ElasticKernelConfig:
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
        return config

    def _collect_ops_data_by_selection_rule(self, selection_rule: Callable) -> List[Any]:
        return list(map(selection_rule, self._elastic_kernel_ops))


class EKBuilderStateNames:
    NODE_NAMES_TO_MAKE_ELASTIC = 'node_names_to_make_elastic'


@ELASTICITY_BUILDERS.register(ElasticityDim.KERNEL)
class ElasticKernelBuilder(SingleElasticityBuilder):
    _state_names = EKBuilderStateNames

    def __init__(self, elasticity_params: Optional[Dict[str, Any]] = None,
                 ignored_scopes: Optional[List[str]] = None,
                 target_scopes: Optional[List[str]] = None):
        super().__init__(ignored_scopes, target_scopes, elasticity_params)
        self._node_names_to_make_elastic = []  # type: List[NNCFNodeName]

    def build(self, target_model: NNCFNetwork) -> ElasticKernelHandler:
        """
        Creates modifications to the given NNCFNetwork for introducing elastic kernel and creates a handler object that
        can manipulate this elasticity.

        :param target_model: a target NNCFNetwork for adding modifications
        :return: a handler object that can manipulate the elastic kernel.
        """
        elastic_kernel_ops = []  # type: List[ElasticKernelOp]
        transformation_commands = []

        graph = target_model.get_original_graph()
        device = next(target_model.parameters()).device
        pad_commands = []

        if not self._node_names_to_make_elastic:
            elastic_kernel_types = [NNCFConv2d.op_func_name]
            all_elastic_kernel_nodes = graph.get_nodes_by_types(elastic_kernel_types)  # type: List[NNCFNode]
            self._node_names_to_make_elastic = [node.node_name for node in all_elastic_kernel_nodes]

        for node_name in self._node_names_to_make_elastic:
            nncf_logger.info("Adding Elastic Kernel op for Conv2D in scope: {}".format(node_name))
            node = graph.get_node_by_name(node_name)
            layer_attrs = node.layer_attributes
            assert isinstance(layer_attrs, ConvolutionLayerAttributes), 'Conv2D can have elastic kernel only'
            max_kernel_size = layer_attrs.kernel_size[0]
            elastic_kernel_op = ElasticKernelConv2DOp(max_kernel_size, node_name, self._elasticity_params)
            elastic_kernel_op.to(device)
            update_conv_params_op = UpdateWeight(elastic_kernel_op)
            transformation_commands.append(
                PTInsertionCommand(
                    PTTargetPoint(
                        TargetType.PRE_LAYER_OPERATION,
                        target_node_name=node_name
                    ),
                    update_conv_params_op,
                    TransformationPriority.PRUNING_PRIORITY
                )
            )
            # TODO(nlyalyus): ticket 71613: remove hardcode for EfficientNet
            #   should be pairing of pad and conv operations by graph analysis
            #   get
            #       1) input size before pad
            #       2) input size after pad
            #       3) active kernel size
            #       4) max kernel size
            #       5) built-in conv padding
            #   output
            #       how to change padded input (e.g. center crop)
            if 'NNCFUserConv2dStaticSamePadding' in node.node_name:
                if max_kernel_size >= 3:
                    crop_op = ElasticKernelInputForExternalPadding(elastic_kernel_op, max_kernel_size)
                    op = UpdateInputs(crop_op).to(device)
                    nncf_logger.warning('Padded input will be cropped for {}'.format(node_name))
                    pad_commands.append(
                        PTInsertionCommand(
                            PTTargetPoint(
                                target_type=TargetType.PRE_LAYER_OPERATION,
                                target_node_name=node_name
                            ),
                            op,
                            TransformationPriority.DEFAULT_PRIORITY
                        )
                    )
            else:
                # Padding
                ap = ElasticKernelPaddingAdjustment(elastic_kernel_op)
                pad_op = UpdatePadding(ap).to(device)
                nncf_logger.warning('Padding will be adjusted for {}'.format(node_name))
                pad_commands.append(
                    PTInsertionCommand(
                        PTTargetPoint(
                            target_type=TargetType.PRE_LAYER_OPERATION,
                            target_node_name=node_name
                        ),
                        pad_op,
                        TransformationPriority.DEFAULT_PRIORITY
                    )
                )
            elastic_kernel_ops.append(elastic_kernel_op)
        if pad_commands:
            transformation_commands += pad_commands

        return ElasticKernelHandler(elastic_kernel_ops, transformation_commands)

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Initializes object from the state.

        :param state: Output of `get_state()` method.
        """
        super().load_state(state)
        self._node_names_to_make_elastic = state[self._state_names.NODE_NAMES_TO_MAKE_ELASTIC]

    def get_state(self) -> Dict[str, Any]:
        """
        Returns a dictionary with Python data structures (dict, list, tuple, str, int, float, True, False, None) that
        represents state of the object.

        :return: state of the object
        """
        state = super().get_state()
        state[self._state_names.NODE_NAMES_TO_MAKE_ELASTIC] = self._node_names_to_make_elastic
        return state
