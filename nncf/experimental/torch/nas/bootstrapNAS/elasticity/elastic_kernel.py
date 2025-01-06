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
import random
from typing import Any, Callable, Dict, List, Optional

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
from nncf.common.logging import nncf_logger
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.base_handler import ELASTICITY_BUILDERS
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.base_handler import ELASTICITY_HANDLERS_MAP
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.base_handler import ELASTICITY_PARAMS
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.base_handler import BaseElasticityParams
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.base_handler import SingleElasticityBuilder
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.base_handler import SingleElasticityHandler
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elasticity_dim import ElasticityDim
from nncf.torch.graph.transformations.commands import PTInsertionCommand
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.layers import NNCFConv2d
from nncf.torch.module_operations import UpdateInputs
from nncf.torch.module_operations import UpdatePadding
from nncf.torch.module_operations import UpdateWeight
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.utils import is_tracing_state
from nncf.torch.utils import no_jit_trace

KernelSizeType = int
ElasticKernelConfig = List[KernelSizeType]  # list of kernel sizes per layer
ElasticKernelSearchSpace = List[List[KernelSizeType]]


class EKParamsStateNames:
    MAX_NUM_KERNELS = "max_num_kernels"


@ELASTICITY_PARAMS.register(ElasticityDim.KERNEL)
class ElasticKernelParams(BaseElasticityParams):
    _state_names = EKParamsStateNames

    def __init__(self, max_num_kernels: int = -1):
        """
        Constructor

        :param max_num_kernels: Restricts total number of different elastic kernel values for each layer.
        The default value is -1 means that there's no restrictions.
        """
        self.max_num_kernels = max_num_kernels

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ElasticKernelParams":
        """
        Creates the object from its config.
        """
        kwargs = {
            cls._state_names.MAX_NUM_KERNELS: config.get(cls._state_names.MAX_NUM_KERNELS, -1),
        }
        return cls(**kwargs)

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "ElasticKernelParams":
        """
        Creates the object from its state.

        :param state: Output of `get_state()` method.
        """
        return cls(**state)

    def get_state(self) -> Dict[str, Any]:
        """
        Returns the compression loss state.

        :return: The compression loss state.
        """
        return {
            self._state_names.MAX_NUM_KERNELS: self.max_num_kernels,
        }

    def __eq__(self, other: "ElasticKernelParams") -> bool:
        return self.__dict__ == other.__dict__


class ElasticKernelOp:
    """
    Base class for introducing elastic kernel for the operations. On the forward pass it takes parameters of operations
    and modifies in the way that kernel size is changing to a given value.
    """

    def __init__(self, max_kernel_size: KernelSizeType, node_name: NNCFNodeName):
        """
        Constructor.

        :param max_kernel_size: maximum kernel size value in the original operation.
        :param node_name: string representation of operation address. It's used for more informative messages only.
        """
        super().__init__()
        self._kernel_size_list = []
        self._active_kernel_size = max_kernel_size
        self._max_kernel_size = max_kernel_size
        self._node_name = node_name

    def get_active_kernel_size(self) -> KernelSizeType:
        """
        :return: a target kernel size that operation's parameters should have after forward's call
        """
        return self._active_kernel_size

    def set_active_kernel_size(self, kernel_size: KernelSizeType) -> None:
        """
        Sets current level of elasticity for the operation - kernel size value that parameters should have.
        The actual modification of parameters happens on forward call.
        The value should be less the original kernel size and more than one.

        :param kernel_size: kernel size value
        """
        if kernel_size is None or kernel_size > self.max_kernel_size or kernel_size < 1:
            raise AttributeError(
                "Invalid kernel size={} in scope={}.\nIt should be within the range: [1, {}]".format(
                    kernel_size, self.node_name, self.max_kernel_size
                )
            )

        self._active_kernel_size = kernel_size

    @property
    def kernel_size_list(self) -> List[KernelSizeType]:
        """
        Gets list of all available kernel sizes to select from. Each value corresponds to a single element in the
        search space of operation. The search space of the model is cartesian product of search spaces of operation.

        :return: list of kernel sizes
        """
        return self._kernel_size_list

    @property
    def node_name(self) -> NNCFNodeName:
        """
        :return: node name that corresponds to operation with elastic kernel
        """
        return self._node_name

    @property
    def max_kernel_size(self) -> KernelSizeType:
        """
        :return: kernel size in the original operation
        """
        return self._max_kernel_size


class ElasticKernelConv2DOp(ElasticKernelOp, nn.Module):
    """
    Introduces elastic kernel for the 2D convolution. On the forward pass it takes parameters of operations
    and modifies in the way that kernel size is changing to a given value.
    """

    def __init__(
        self,
        max_kernel_size: KernelSizeType,
        node_name: NNCFNodeName,
        params: ElasticKernelParams,
        original_padding_value: Optional[int] = 0,
    ):
        """
        Constructor.

        :param max_kernel_size: maximum kernel size value in the original operation.
        :param node_name: string representation of operation address. It's used for more informative messages only.
        :param params: parameters to configure elastic kernel for the operation.
        :param original_padding_value: the padding value used in the original model.
        """
        super().__init__(max_kernel_size=max_kernel_size, node_name=node_name)
        self._max_num_params = params.max_num_kernels
        self._original_padding_value = original_padding_value
        # Create kernel_size_list based on max module kernel size
        self._kernel_size_list = self.generate_kernel_size_list(max_kernel_size, original_padding_value)
        self._ks_set = list(set(self.kernel_size_list))
        self._ks_set.sort()

        scale_params = {}
        for i in range(len(self._ks_set) - 1):
            ks_small = self._ks_set[i]
            ks_larger = self._ks_set[i + 1]
            param_name = "%dto%d" % (ks_larger, ks_small)
            # noinspection PyArgumentList
            scale_params["%s_matrix" % param_name] = Parameter(torch.eye(ks_small**2))
        for name, param in scale_params.items():
            self.register_parameter(name, param)

    def generate_kernel_size_list(
        self, max_kernel_size: KernelSizeType, original_padding_value: int
    ) -> List[KernelSizeType]:
        """
        Generates list of available kernel size values.

        :param max_kernel_size: maximum value of kernel size, it's supposed to be odd
        :param original_padding_value: the padding value used in the original model.
        :return: list of kernel size values.
        """
        DEFAULT_KERNEL_SIZE_STEP = 2
        assert max_kernel_size % 2 > 0, "kernel size should be odd number"
        if max_kernel_size == 1:
            return [1]
        kernel = max_kernel_size
        ks_list = []
        while kernel >= max(max_kernel_size - 2 * original_padding_value, 3):
            ks_list.append(kernel)
            kernel -= DEFAULT_KERNEL_SIZE_STEP
            if self._max_num_params == len(ks_list):
                break
        return ks_list

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Modifies weight to have kernel size equals to active kernel size value.

        :param weight: weight tensor to be modified
        :return: modified weight
        """
        kernel_size = self.get_active_kernel_size()
        nncf_logger.debug(f"Conv2d with active kernel size={kernel_size} in scope={self.node_name}")

        result = weight
        if is_tracing_state():
            with no_jit_trace():
                if kernel_size > 1:
                    result = self._get_active_filter(kernel_size, weight).detach()
        else:
            if kernel_size > 1:
                result = self._get_active_filter(kernel_size, weight)
        return result

    def set_active_kernel_size(self, kernel_size: KernelSizeType) -> None:
        """
        Sets current level of elasticity for the operation - kernel size value that parameters should have.
        The actual modification of parameters happens on forward call.
        The value should be less the original kernel size and more than one.

        :param kernel_size: kernel size value
        """
        nncf_logger.debug(f"set active elastic_kernel={kernel_size} in scope={self.node_name}")
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        if kernel_size not in self.kernel_size_list and kernel_size != self.max_kernel_size:
            raise ValueError("invalid kernel size to set. Should be a number in {}".format(self.kernel_size_list))
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
                    _input_filter,
                    self.__getattr__("%dto%d_matrix" % (src_ks, target_ks)),
                )
                _input_filter = _input_filter.view(filters.size(0), filters.size(1), target_ks**2)
                _input_filter = _input_filter.view(filters.size(0), filters.size(1), target_ks, target_ks)
                start_filter = _input_filter
            filters = start_filter
        return filters

    @property
    def original_padding_value(self) -> int:
        return self._original_padding_value


class ElasticKernelPaddingAdjustment:
    """
    Auxiliary operation that adjusts padding of the operation with elastic kernel.
    This adjustment ensures the same output shapes for different kernel sizes and thus frees from the need to adapt the
    rest of the operation in the model for each new value of kernel size.
    """

    def __init__(self, elastic_k_w_op: ElasticKernelConv2DOp):
        self._elastic_k_w_op = elastic_k_w_op

    def __call__(self, _) -> int:
        shift_padding_value = (
            self._elastic_k_w_op.max_kernel_size - self._elastic_k_w_op.get_active_kernel_size()
        ) // 2
        return self._elastic_k_w_op.original_padding_value - shift_padding_value


class ElasticKernelInputForExternalPadding:
    """
    Auxiliary operation that adjusts inputs for operations that implements same padding type
    (e.g. NNCFUserConv2dStaticSamePadding).
    This adjustment ensures the same output shapes for different kernel sizes and thus frees from the need to adapt the
    rest of the operation in the model for each new value of kernel size.
    """

    def __init__(self, elastic_k_w_op: ElasticKernelConv2DOp, max_kernel_size: KernelSizeType):
        self._elastic_k_w_op = elastic_k_w_op
        self._max_kernel_size = max_kernel_size

    def __call__(self, x) -> int:
        active_kernel_size = self._elastic_k_w_op.get_active_kernel_size()
        diff = (self._max_kernel_size - active_kernel_size) // 2
        h = x.size(2)
        w = x.size(3)
        new_x = x[:, :, diff : h - diff, diff : w - diff]
        return new_x


class ElasticKernelHandler(SingleElasticityHandler):
    """
    An interface for handling elastic kernel dimension in the network, i.e. define size of kernels in the conv layers.
    """

    def __init__(self, elastic_kernel_ops: List[ElasticKernelOp], transformation_commands: List[TransformationCommand]):
        super().__init__()
        self._elastic_kernel_ops = elastic_kernel_ops
        self._transformation_commands = transformation_commands

    def get_transformation_commands(self) -> List[TransformationCommand]:
        """
        :return: transformation commands for introducing the elasticity to NNCFNetwork
        """
        return self._transformation_commands

    def get_search_space(self) -> ElasticKernelSearchSpace:
        """
        :return: search space that is produced by iterating over all elastic kernel parameters
        """
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
            lambda op: op.kernel_size_list[random.randrange(0, len(op.kernel_size_list))]  # nosec
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
        self.activate_subnet_for_config(supernet_config)

    def activate_subnet_for_config(self, config: ElasticKernelConfig) -> None:
        """
        Activates a Subnet that corresponds to the given elasticity configuration

        :return: list of kernel sizes per layer
        """
        for op, ks in zip(self._elastic_kernel_ops, config):
            op.set_active_kernel_size(ks)

    def get_active_kernel_sizes_per_node(self) -> Dict[str, Any]:
        """
        :return: mapping of node name to the active kernel sizes in that node
        """
        active_kernel_sizes = {}
        for elastic_kernel_op in self._elastic_kernel_ops:
            active_kernel_size = elastic_kernel_op.get_active_kernel_size()
            active_kernel_sizes[elastic_kernel_op.node_name] = (active_kernel_size, active_kernel_size)
        return active_kernel_sizes

    def resolve_conflicts_with_other_elasticities(
        self, config: ElasticKernelConfig, elasticity_handlers: ELASTICITY_HANDLERS_MAP
    ) -> ElasticKernelConfig:
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
    NODE_NAMES_TO_MAKE_ELASTIC = "node_names_to_make_elastic"


@ELASTICITY_BUILDERS.register(ElasticityDim.KERNEL)
class ElasticKernelBuilder(SingleElasticityBuilder):
    _state_names = EKBuilderStateNames

    def __init__(
        self,
        params: ElasticKernelParams,
        ignored_scopes: Optional[List[str]] = None,
        target_scopes: Optional[List[str]] = None,
    ):
        super().__init__(ignored_scopes, target_scopes)
        self._node_names_to_make_elastic: List[NNCFNodeName] = []
        self._params = params

    def build(self, target_model: NNCFNetwork) -> ElasticKernelHandler:
        """
        Creates modifications to the given NNCFNetwork for introducing elastic kernel and creates a handler object that
        can manipulate this elasticity.

        :param target_model: a target NNCFNetwork for adding modifications
        :return: a handler object that can manipulate the elastic kernel.
        """
        elastic_kernel_ops: List[ElasticKernelOp] = []
        transformation_commands = []

        graph = target_model.nncf.get_original_graph()
        device = next(target_model.parameters()).device
        pad_commands = []

        if not self._node_names_to_make_elastic:
            elastic_kernel_types = [NNCFConv2d.op_func_name]
            all_elastic_kernel_nodes: List[NNCFNode] = graph.get_nodes_by_types(elastic_kernel_types)
            self._node_names_to_make_elastic = [node.node_name for node in all_elastic_kernel_nodes]

        for node_name in self._node_names_to_make_elastic:
            nncf_logger.debug(f"Adding Elastic Kernel op for Conv2D in scope: {node_name}")
            node = graph.get_node_by_name(node_name)
            layer_attrs = node.layer_attributes
            assert isinstance(layer_attrs, ConvolutionLayerAttributes), "Conv2D can have elastic kernel only"
            max_kernel_size = layer_attrs.kernel_size[0]
            original_padding_values = layer_attrs.padding_values[0]
            elastic_kernel_op = ElasticKernelConv2DOp(max_kernel_size, node_name, self._params, original_padding_values)
            elastic_kernel_op.to(device)
            update_conv_params_op = UpdateWeight(elastic_kernel_op)
            transformation_commands.append(
                PTInsertionCommand(
                    PTTargetPoint(TargetType.PRE_LAYER_OPERATION, target_node_name=node_name),
                    update_conv_params_op,
                    TransformationPriority.PRUNING_PRIORITY,
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
            if "NNCFUserConv2dStaticSamePadding" in node.node_name:
                if max_kernel_size >= 3:
                    crop_op = ElasticKernelInputForExternalPadding(elastic_kernel_op, max_kernel_size)
                    op = UpdateInputs(crop_op).to(device)
                    nncf_logger.debug(f"Padded input will be cropped for {node_name}")
                    pad_commands.append(
                        PTInsertionCommand(
                            PTTargetPoint(target_type=TargetType.PRE_LAYER_OPERATION, target_node_name=node_name),
                            op,
                            TransformationPriority.DEFAULT_PRIORITY,
                        )
                    )
            else:
                # Padding
                ap = ElasticKernelPaddingAdjustment(elastic_kernel_op)
                pad_op = UpdatePadding(ap).to(device)
                nncf_logger.debug(f"Padding will be adjusted for {node_name}")
                pad_commands.append(
                    PTInsertionCommand(
                        PTTargetPoint(target_type=TargetType.PRE_LAYER_OPERATION, target_node_name=node_name),
                        pad_op,
                        TransformationPriority.DEFAULT_PRIORITY,
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
        params_from_state = state[SingleElasticityBuilder._state_names.ELASTICITY_PARAMS]
        params = ElasticKernelParams.from_state(params_from_state)
        if self._params and self._params != params:
            nncf_logger.warning(
                "Different elasticity parameters were provided in two places: on init and on loading "
                "state. The one from state is taken by ignoring the ones from init."
            )
        self._params = params
        self._node_names_to_make_elastic = state[self._state_names.NODE_NAMES_TO_MAKE_ELASTIC]

    def get_state(self) -> Dict[str, Any]:
        """
        Returns a dictionary with Python data structures (dict, list, tuple, str, int, float, True, False, None) that
        represents state of the object.

        :return: state of the object
        """
        return {
            SingleElasticityBuilder._state_names.ELASTICITY_PARAMS: self._params.get_state(),
            self._state_names.NODE_NAMES_TO_MAKE_ELASTIC: self._node_names_to_make_elastic,
        }
