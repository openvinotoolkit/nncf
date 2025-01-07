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
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import nn

import nncf
from nncf.common.graph import NNCFNodeName
from nncf.common.graph.layer_attributes import BaseLayerAttributes
from nncf.common.graph.layer_attributes import ConvolutionLayerAttributes
from nncf.common.graph.layer_attributes import GenericWeightedLayerAttributes
from nncf.common.graph.layer_attributes import LinearLayerAttributes
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationCommand
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.common.logging import nncf_logger
from nncf.common.pruning.clusterization import Cluster
from nncf.common.pruning.clusterization import Clusterization
from nncf.common.pruning.mask_propagation import MaskPropagationAlgorithm
from nncf.common.pruning.node_selector import PruningNodeSelector
from nncf.common.pruning.shape_pruning_processor import ShapePruningProcessor
from nncf.common.pruning.structs import PrunedLayerInfoBase
from nncf.common.pruning.utils import get_input_masks
from nncf.common.pruning.utils import get_prunable_layers_in_out_channels
from nncf.common.pruning.utils import is_prunable_depthwise_conv
from nncf.common.scopes import should_consider_scope
from nncf.common.tensor import NNCFTensor
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.base_handler import ELASTICITY_BUILDERS
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.base_handler import ELASTICITY_HANDLERS_MAP
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.base_handler import ELASTICITY_PARAMS
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.base_handler import BaseElasticityParams
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.base_handler import SingleElasticityBuilder
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.base_handler import SingleElasticityHandler
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elasticity_dim import ElasticityDim
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.filter_reorder import FilterReorderingAlgorithm
from nncf.torch.graph.graph import PTNNCFGraph
from nncf.torch.graph.operator_metatypes import PTModuleBatchNormMetatype
from nncf.torch.graph.operator_metatypes import PTModuleConv2dMetatype
from nncf.torch.graph.operator_metatypes import PTModuleDepthwiseConv2dSubtype
from nncf.torch.graph.operator_metatypes import PTModuleLayerNormMetatype
from nncf.torch.graph.operator_metatypes import PTModuleLinearMetatype
from nncf.torch.graph.transformations.commands import PTInsertionCommand
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.layers import NNCFConv2d
from nncf.torch.layers import NNCFLinear
from nncf.torch.module_operations import UpdateBatchNormParams
from nncf.torch.module_operations import UpdateLayerNormParams
from nncf.torch.module_operations import UpdateNumGroups
from nncf.torch.module_operations import UpdateWeight
from nncf.torch.module_operations import UpdateWeightAndOptionalBias
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.pruning.filter_pruning.functions import FILTER_IMPORTANCE_FUNCTIONS
from nncf.torch.pruning.operations import PT_PRUNING_OPERATOR_METATYPES
from nncf.torch.pruning.operations import PTElementwisePruningOp
from nncf.torch.pruning.tensor_processor import PTNNCFPruningTensorProcessor
from nncf.torch.tensor import PTNNCFTensor
from nncf.torch.utils import get_filters_num
from nncf.torch.utils import get_model_device

PruningGroupID = int
WidthType = int
ElasticWidthConfig = Dict[PruningGroupID, WidthType]
WidthList = List[WidthType]
ElasticWidthSearchSpace = Dict[PruningGroupID, WidthList]


def sum_filter(weight_tensor, dim=0):
    """
    Calculates the sum for weight_tensor for the selected dimension.
    """
    weight_tensor = weight_tensor.transpose(0, dim).contiguous()
    return torch.sum(weight_tensor.view(weight_tensor.shape[0], -1), dim=1)


class ElasticWidthOp:
    """
    Base class for introducing elastic width for the operations. On the forward pass it takes parameters of operations
    and trims its input channels (elastic input width) or output channels (elastic output width). This class produces
    2 groups of classes with prefixes ElasticOutputWidth and ElasticInputWidth correspondingly.
    """

    def __init__(self, max_width: int, node_name: str):
        """
        Constructor.

        :param max_width: maximum number of channels in the original layer.
        :param node_name: string representation of operation address. It's used for more informative messages only.
        """
        super().__init__()
        self._active_width = max_width
        self._max_width = max_width
        self._node_name = node_name

    @property
    def max_width(self) -> WidthType:
        """
        :return: maximum number of channels specified on creation of the object.
        """
        return self._max_width

    def get_active_width(self) -> WidthType:
        """
        :return: number of channels to trim on forward call
        """
        return self._active_width

    def set_active_width(self, width: WidthType) -> None:
        """
        Sets current level of elasticity for the operation - number of channels to trim.
        The actual trimming of specified number of channels happens on forward call.
        The value should be less the original width and more than one. Zero number of channels is
        supported through Dynamic Depth feature.

        :param width: number of channels
        """
        if width is None or width > self._max_width or width < 1:
            raise AttributeError(
                "Invalid width={} in scope={}.\nIt should be within the range: [1, {}]".format(
                    width, self._node_name, self._max_width
                )
            )

        self._active_width = width


class EWParamsStateNames:
    MIN_WIDTH = "min_width"
    MAX_NUM_WIDTHS = "max_num_widths"
    WIDTH_STEP = "width_step"
    WIDTH_MULTIPLIERS = "width_multipliers"
    FILTER_IMPORTANCE = "filter_importance"
    EXTERNAL_IMPORTANCE_PATH = "external_importance_path"
    OVERWRITE_GROUPS = "overwrite_groups"
    OVERWRITE_GROUPS_WIDTHS = "overwrite_groups_widths"
    ADD_DYNAMIC_INPUTS = "add_dynamic_inputs"


@ELASTICITY_PARAMS.register(ElasticityDim.WIDTH)
class ElasticWidthParams(BaseElasticityParams):
    _state_names = EWParamsStateNames

    def __init__(
        self,
        min_width: int,
        max_num_widths: int,
        width_step: int,
        width_multipliers: List[float],
        filter_importance: str,
        external_importance_path: Optional[str] = None,
        overwrite_groups: Optional[List[str]] = None,
        overwrite_groups_widths: Optional[List[str]] = None,
        add_dynamic_inputs: Optional[List[str]] = None,
    ):
        """
        Constructor

        :param min_width: Minimal number of output channels that can be activated for each layers with elastic width.
        Default value is 32.
        :param max_num_widths: Restricts total number of different elastic width values for each layer.
        The default value is -1 means that there's no restrictions.
        :param width_step: Defines a step size for a generation of the elastic width search space - the list of all
        possible width values for each layer. The generation starts from the number of output channels in the original
        model and stops when it reaches whether a `min_width` width value or number of generated width values
        equal to `max_num_widths`.
        This parameter is mutually exclusive with `width_multipliers`.
        :param width_multipliers: Defines elastic width search space via a list of multipliers. All possible width
        values are obtained by multiplying the original width value with the values in the given list.
        The obtained values are rounded to the nearest smaller value divisible by alignment constant (e.g. 8).
        This parameter is mutually exclusive with `width_step`.
        :param filter_importance: The type of filter importance metric. Can be one of `L1`, `L2`, `geometric_median`,
        `external`. `L1` by default.
        :param external_importance_path: Path to the custom external weight importance (PyTorch tensor) per node
        that will be used for weight reordering. Valid only when filter_importance is `external`.
        """
        self.min_width = min_width
        self.max_num_widths = max_num_widths
        self.width_step = width_step
        self.width_multipliers = width_multipliers
        assert (
            filter_importance != "external" or external_importance_path is not None
        ), "Missing external weight importance path."
        self.filter_importance = filter_importance
        self.external_importance_path = external_importance_path

        self.overwrite_groups = overwrite_groups
        self.overwrite_groups_widths = overwrite_groups_widths
        self.add_dynamic_inputs = add_dynamic_inputs

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ElasticWidthParams":
        """
        Creates the object from its config.
        """
        kwargs = {
            cls._state_names.MIN_WIDTH: config.get(cls._state_names.MIN_WIDTH, 32),
            cls._state_names.MAX_NUM_WIDTHS: config.get(cls._state_names.MAX_NUM_WIDTHS, -1),
            cls._state_names.WIDTH_STEP: config.get(cls._state_names.WIDTH_STEP, 32),
            cls._state_names.WIDTH_MULTIPLIERS: config.get(cls._state_names.WIDTH_MULTIPLIERS),
            cls._state_names.FILTER_IMPORTANCE: config.get(cls._state_names.FILTER_IMPORTANCE, "L1"),
            cls._state_names.EXTERNAL_IMPORTANCE_PATH: config.get(cls._state_names.EXTERNAL_IMPORTANCE_PATH, None),
            cls._state_names.OVERWRITE_GROUPS: config.get(cls._state_names.OVERWRITE_GROUPS, None),
            cls._state_names.OVERWRITE_GROUPS_WIDTHS: config.get(cls._state_names.OVERWRITE_GROUPS_WIDTHS, None),
            cls._state_names.ADD_DYNAMIC_INPUTS: config.get(cls._state_names.ADD_DYNAMIC_INPUTS, None),
        }
        return cls(**kwargs)

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "ElasticWidthParams":
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
            self._state_names.MIN_WIDTH: self.min_width,
            self._state_names.MAX_NUM_WIDTHS: self.max_num_widths,
            self._state_names.WIDTH_STEP: self.width_step,
            self._state_names.WIDTH_MULTIPLIERS: self.width_multipliers,
            self._state_names.FILTER_IMPORTANCE: self.filter_importance,
            self._state_names.EXTERNAL_IMPORTANCE_PATH: self.external_importance_path,
            self._state_names.OVERWRITE_GROUPS: self.overwrite_groups,
            self._state_names.OVERWRITE_GROUPS_WIDTHS: self.overwrite_groups_widths,
            self._state_names.ADD_DYNAMIC_INPUTS: self.add_dynamic_inputs,
        }

    def __eq__(self, other: "ElasticWidthParams") -> bool:
        return self.__dict__ == other.__dict__

    def __str__(self):
        return (
            f"{self.__class__.__name__}: width_step: {self.width_step} "
            f"min_width: {self.min_width} width_multipliers: {self.width_multipliers} "
            f"max_num_widths: {self.max_num_widths} overwrite_groups: {self.overwrite_groups} "
            f"overwrite_group_widths: {self.overwrite_groups_widths}"
        )


class ElasticOutputWidthOp(ElasticWidthOp):
    """
    Base class for trimming output channels (output width) of the operations.
    """

    def __init__(
        self,
        max_width: int,
        node_name: str,
        params: ElasticWidthParams,
        fixed_width_list: Optional[List[int]] = None,
    ):
        """
        Constructor.

        :param max_width: maximum number of output channels in the original operation.
        :param node_name: string representation of operation address. It's used for more informative messages only.
        :param params: parameters to configure elastic width for the operation.
        """
        super().__init__(max_width=max_width, node_name=node_name)
        if fixed_width_list is None:
            fixed_width_list = []
        if fixed_width_list:
            fixed_width_list.sort(reverse=True)
            if fixed_width_list[0] > max_width:
                raise nncf.InternalError(
                    f"Width list for {node_name} contains invalid values: {fixed_width_list}, {max_width}"
                )
            if fixed_width_list[0] != max_width:
                raise nncf.ValidationError(f"Max width for {node_name} is not aligned with pre-trained model")
            self._width_list = fixed_width_list
        else:
            self._width_list = self._generate_width_list(self._max_width, params)

    @property
    def width_list(self) -> List[int]:
        """
        list of all available widths to select from. Each value corresponds to a single element in the search space of
        operation. The search space of the model is cartesian product of search spaces of operation.
        If all widths starting from 1 to maximum number of channels with step size 1 are available, the search space
        would be prohibitively large to efficiently train and search.

        That's why there are elastic width parameters that constraint number of all available widths.

        :return: list of widths
        """
        return self._width_list

    def set_active_width(self, width: int) -> None:
        """
        Sets current level of elasticity for the operation - number of output channels to trim.
        The actual trimming of specified number of channels happens on forward call.
        The value should be less the original width and more than one. Zero number of channels is
        supported through Dynamic Depth feature.

        :param width: number of output channels
        """
        if width not in self.width_list and width != self.max_width:
            raise ValueError(
                f"Invalid number of output channels to set: {width} in scope={self._node_name}. "
                f"Should be a number in {self.width_list}"
            )
        super().set_active_width(width)

    @staticmethod
    def _generate_width_list(max_width: int, params: ElasticWidthParams) -> List[int]:
        """
        Generates list of available width values.
        There are two mutually exclusive modes: using `width_step` and using `width_multipliers`.
        The mode with `width_step` defines a step size for a generation of the elastic width search space - the list of
        all possible width values for each layer. The generation starts from the number of output channels in the
        original model and stops when it reaches whether a `min_width` width value or number of generated width values
        equal to `max_num_widths`.
        The mode with `width_multipliers` defines elastic width search space via a list of multipliers. All possible
        width values are obtained by multiplying the original width value with the values in the given list.
        The obtained values are rounded to the nearest smaller value divisible by alignment constant (e.g. 8).

        :param max_width: maximum value of width
        :param params: parameters to configure elastic width for the operation.
        :return: list of available width values.
        """
        ALIGNMENT_CONSTANT_FOR_MULTIPLIERS = 8
        width_list = []
        p = params
        if max_width <= p.min_width:
            width_list.append(max_width)
        elif not p.width_multipliers:
            width = max_width
            width_list.append(width)
            width -= p.width_step
            while width >= p.min_width:
                if p.max_num_widths == len(width_list):
                    break
                width_list.append(width)
                width -= p.width_step
        else:
            p.width_multipliers.sort(reverse=True)
            if p.width_multipliers[0] < 1:
                width_list.append(max_width)
            for multiplier in p.width_multipliers:
                if p.max_num_widths == len(width_list):
                    break
                if 0 >= multiplier > 1:
                    raise nncf.InternalError(f"Wrong value for multiplier: {multiplier}")
                w = int(max_width * multiplier)
                w = w - (w % ALIGNMENT_CONSTANT_FOR_MULTIPLIERS)
                w = max(w, p.min_width)
                if w in width_list:
                    continue
                width_list.append(w)
        return width_list


class ElasticWidthInfo(PrunedLayerInfoBase):
    """
    List of attributes describing operation with elastic width
    """

    def __init__(
        self,
        node_name: NNCFNodeName,
        module: nn.Module,
        elastic_op: ElasticOutputWidthOp,
        node_id: int,
        is_depthwise: bool,
    ):
        super().__init__(node_name, node_id, is_depthwise)
        self.module = module
        self.elastic_op = elastic_op

    def __str__(self):
        return (
            f"{self.__class__.__name__}: node_name: {self.node_name} module: {self.module} "
            f"elastic_op: {self.elastic_op} node_id: {self.nncf_node_id} is_depthwise: {self.is_depthwise}"
        )


class ElasticInputWidthLinearOp(ElasticWidthOp, nn.Module):
    """
    Introduces elastic input width for linear layer.
    """

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Trims weight according to active number of input channels

        :param weight: weight tensor to be trimmed
        :return: trimmed weight
        """
        return weight[:, : self._active_width]


class ElasticInputWidthConvOp(ElasticWidthOp, nn.Module):
    """
    Introduces elastic input width for 2D convolution.
    """

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Trims weight according to active number of input channels

        :param weight: weight tensor to be trimmed
        :return: trimmed weight
        """
        return weight[:, : self._active_width, :, :]


class ElasticInputWidthDWConvOp(ElasticWidthOp, nn.Module):
    """
    Introduces elastic input width for depthwise convolution.
    """

    def forward(self, _) -> int:
        """
        :return: number of input channels to be trimmed. In case of depthwise convolution no need to trim weights, just
        need to change number of group accordingly.
        """
        return self._active_width


class ElasticInputWidthBatchNormOp(ElasticWidthOp, nn.Module):
    """
    Introduces elastic input width for batchnorm layer.
    """

    SET_RUNNING_STATISTICS = False

    def forward(self, **bn_params: torch.Tensor) -> List[torch.Tensor]:
        """
        Trims batchnorm parameters according to active number of input channels.

        :param bn_params: map of name and tensor to be trimmed
        :return: trimmed batchnorm parameters
        """
        return [param[: self._active_width] for param in bn_params.values()]


class ElasticInputWidthLayerNormOp(ElasticWidthOp, nn.Module):
    """
    Introduces elastic input width for layernorm layer.
    """

    def forward(self, weight: torch.Tensor, bias: torch.Tensor, normalized_shape: torch.Tensor) -> List[torch.Tensor]:
        """
        Trims layernorm parameters according to active number of input channels.
        :param weight: weight tensor to be trimmed
        :param bias: bias tensor to be trimmed
        :param normalized_shape: normalized_shape to be trimmed
        :return: list of trimmed layernorm parameters
        """
        assert len(normalized_shape) == 1, "Currently only 1-dimensional shape is supported."
        return [
            weight[: self._active_width],
            bias[: self._active_width],
            (self._active_width,),
        ]


class ElasticOutputWidthConv2DOp(ElasticOutputWidthOp, nn.Module):
    """
    Introduces elastic output width for 2D convolution.
    """

    def forward(self, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> List[torch.Tensor]:
        """
        Trims convolution parameters according to active number of output channels.

        :param weight: weight tensor to be trimmed
        :param bias: bias tensor to be trimmed
        :return: list of trimmed convolution parameters
        """
        nncf_logger.debug(f"Conv2d with active width={self._active_width} in scope={self._node_name}")
        num_out_channels = self._active_width
        new_bias = None if bias is None else bias[:num_out_channels]
        new_weights = weight[:num_out_channels, :, :, :]
        return [new_weights, new_bias]


class ElasticOutputWidthLinearOp(ElasticOutputWidthOp, nn.Module):
    """
    Introduces elastic output width for linear layer.
    """

    def forward(self, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> List[torch.Tensor]:
        """
        Trims linear layer's parameters according to active number of output channels.

        :param weight: weight tensor to be trimmed
        :param bias: bias tensor to be trimmed
        :return: list of trimmed linear parameters
        """
        new_bias = None if bias is None else bias[: self._active_width]
        return [weight[: self._active_width, :], new_bias]


class EWHandlerStateNames:
    WIDTH_NUM_PARAMS_INDICATOR = "width_num_params_indicator"


class ElasticWidthHandler(SingleElasticityHandler):
    """
    An interface for handling elastic width dimension in the network, i.e. define number of channels in the layers.
    """

    _width_state_names = EWHandlerStateNames

    def __init__(
        self,
        target_model: NNCFNetwork,
        filter_importance_fn: Callable[[torch.Tensor, int], torch.Tensor],
        external_importance_path: Optional[str],
        weights_normalizer_fn: Optional[Callable[[torch.Tensor], torch.Tensor]],
        node_name_vs_dynamic_input_width_op_map: Dict[NNCFNodeName, ElasticWidthOp],
        pruned_module_groups_info: Clusterization[ElasticWidthInfo](id_fn=lambda x: x.node_name),
        transformation_commands: List[TransformationCommand],
        add_dynamic_inputs: Optional[List[str]] = None,
    ):
        """
        Constructor

        :param target_model: a target NNCFNetwork for adding elasticity.
        :param filter_importance_fn: a callable that implements calculation of the importance of filters along a given
        dimension for a given weight tensor.
        :param weights_normalizer_fn: a callable that implements normalization of weight tensor
        :param node_name_vs_dynamic_input_width_op_map: map of node name to elastic width operation associated with it
        :param pruned_module_groups_info: structure that specifies information about groups of modules being pruned
        using elastic width operations.
        :param transformation_commands: list of transformation commands for introducing elastic width operations.
        """
        super().__init__()
        self._target_model = target_model
        self._node_name_vs_dynamic_input_width_op_map = node_name_vs_dynamic_input_width_op_map
        self._pruned_module_groups_info = pruned_module_groups_info
        self._transformation_commands = transformation_commands
        self._filter_importance_fn = filter_importance_fn
        self._external_importance = None
        if external_importance_path is not None:
            self._external_importance = torch.load(external_importance_path)
            nncf_logger.debug("Loaded custom external weight importance.")
        self._weights_normalizer_fn = weights_normalizer_fn
        self._add_dynamic_inputs = add_dynamic_inputs

        graph = self._target_model.nncf.get_original_graph()
        prunable_types = [NNCFConv2d.op_func_name, NNCFLinear.op_func_name]
        self._shape_pruning_processor = ShapePruningProcessor(
            prunable_types=prunable_types,
            pruning_operations_metatype=PT_PRUNING_OPERATOR_METATYPES,
        )
        self._next_nodes = self._shape_pruning_processor.get_next_nodes(graph, pruned_module_groups_info)
        # Need a copy because it will be used for adding `output_mask`/`input_masks` to nodes that are relevant to
        # Elastic Width only and therefore it should be isolated to not intercept with other algorithms.
        self._propagation_graph = deepcopy(graph)

        self._width_num_params_indicator = -1

    @property
    def width_num_params_indicator(self):
        return self._width_num_params_indicator

    @width_num_params_indicator.setter
    def width_num_params_indicator(self, width_num_params_indicator):
        if width_num_params_indicator == 0 or width_num_params_indicator < -1:
            raise nncf.InternalError(f"Invalid width indicator: {width_num_params_indicator}")
        self._width_num_params_indicator = width_num_params_indicator

    @property
    def propagation_graph(self) -> PTNNCFGraph:
        """
        :return: nncf graph that is used for propagating pruning and reordering masks
        """
        return self._propagation_graph

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Initializes object from the state.
        :param state: Output of `get_state()` method.
        """
        super().load_state(state)
        self.width_num_params_indicator = state[self._width_state_names.WIDTH_NUM_PARAMS_INDICATOR]

    def get_state(self) -> Dict[str, Any]:
        """
        Returns a dictionary with Python data structures (dict, list, tuple, str, int, float, True, False, None) that
        represents state of the object.
        :return: state of the object
        """
        state = super().get_state()
        state[self._width_state_names.WIDTH_NUM_PARAMS_INDICATOR] = self.width_num_params_indicator
        return state

    def get_transformation_commands(self) -> List[TransformationCommand]:
        """
        :return: transformation commands for introducing the elasticity to NNCFNetwork
        """
        return self._transformation_commands

    def get_search_space(self) -> ElasticWidthSearchSpace:
        """
        :return: search space that is produced by iterating over all elastic parameters
        """
        if self._width_num_params_indicator == -1:
            return self._collect_ops_data_by_selection_rule(lambda op: op.width_list)
        return self._collect_ops_data_by_selection_rule(
            lambda op: op.width_list[: min(self._width_num_params_indicator, len(op.width_list))]
        )

    def get_active_config(self) -> ElasticWidthConfig:
        """
        Forms an elasticity configuration that describes currently activated Subnet

        :return: map of pruning group id to width value
        """
        return self._collect_ops_data_by_selection_rule(lambda op: op.get_active_width())

    def get_random_config(self) -> ElasticWidthConfig:
        """
        Forms an elasticity configuration that describes a Subnet with randomly chosen width for each elastic layer

        :return: map of pruning group id to width value
        """
        return self._collect_ops_data_by_selection_rule(
            lambda op: op.width_list[random.randrange(0, self._get_width_list_len(op))]  # nosec
        )

    def get_minimum_config(self) -> ElasticWidthConfig:
        """
        Forms an elasticity configuration that describes a Subnet with minimum width in each elastic layer

        :return: map of pruning group id to width value
        """
        return self._collect_ops_data_by_selection_rule(lambda op: min(self._get_width_list(op)))

    def get_maximum_config(self) -> ElasticWidthConfig:
        """
        Forms an elasticity configuration that describes a Subnet with maximum width in each elastic layer

        :return: map of pruning group id to width value
        """
        return self._collect_ops_data_by_selection_rule(lambda op: max(self._get_width_list(op)))

    def activate_supernet(self) -> None:
        """
        Activates the Supernet - the original network to which elasticity was applied.
        """
        supernet_config = self._collect_ops_data_by_selection_rule(lambda op: op.max_width)
        self.activate_subnet_for_config(supernet_config)

    def activate_subnet_for_config(self, config: ElasticWidthConfig) -> None:
        """
        Activates a Subnet that corresponds to the given elasticity configuration

        :param config: map of pruning group id to width value
        """
        for node in self._propagation_graph.get_all_nodes():
            node.attributes.pop("output_mask", None)

        names_of_processed_nodes = set()
        for cluster_id, width in config.items():
            cluster = self._pruned_module_groups_info.get_cluster_by_id(cluster_id)
            for elastic_width_info in cluster.elements:
                node_id = elastic_width_info.nncf_node_id
                node = self._propagation_graph.get_node_by_id(node_id)
                max_width = elastic_width_info.elastic_op.max_width
                device = get_model_device(self._target_model)
                mask = self._width_to_mask(width, max_width, device)
                node.attributes["output_mask"] = mask
                elastic_width_info.elastic_op.set_active_width(width)
                names_of_processed_nodes.add(node_id)

        algo = MaskPropagationAlgorithm(
            self._propagation_graph,
            PT_PRUNING_OPERATOR_METATYPES,
            PTNNCFPruningTensorProcessor,
        )
        algo.mask_propagation()

        for (
            node_name,
            dynamic_input_width_op,
        ) in self._node_name_vs_dynamic_input_width_op_map.items():
            node = self._propagation_graph.get_node_by_name(node_name)
            input_masks = get_input_masks(node, self._propagation_graph)
            was_set = False
            if input_masks:
                input_mask = input_masks[0]
                input_width = self.mask_to_width(input_mask)
                if input_width:
                    dynamic_input_width_op.set_active_width(input_width)
                    was_set = True

            if not was_set and node_name not in names_of_processed_nodes:
                nncf_logger.debug(f"input width was not set in scope={node.node_name}")

            if self._add_dynamic_inputs and node_name in self._add_dynamic_inputs and not was_set:
                nncf_logger.debug(f"setting input width by user's request for scope={node_name}")
                nodes_to_check = [node]
                while any(elem is None for elem in input_masks):
                    previous_nodes = []
                    for node in nodes_to_check:
                        previous_nodes.append(self._propagation_graph.get_previous_nodes(node))
                    nodes_to_check.clear()
                    previous_nodes = [item for nodes in previous_nodes for item in nodes]
                    if not previous_nodes:
                        break
                    for previous in previous_nodes:
                        if "output_mask" in previous.attributes:
                            if previous.attributes["output_mask"] is not None:
                                input_masks.append(previous.attributes["output_mask"])
                                input_masks = [i for i in input_masks if i]
                            else:
                                nodes_to_check.append(previous)
                        else:
                            nodes_to_check.append(previous)
                if input_masks:
                    input_mask = input_masks[0]
                    input_width = self.mask_to_width(input_mask)
                    if input_width:
                        dynamic_input_width_op.set_active_width(input_width)
                        was_set = True
                if was_set:
                    nncf_logger.debug(f"Success setting up user's request for dynamic input at scope={node_name}")

    def get_active_in_out_width_values(
        self,
    ) -> Tuple[Dict[NNCFNodeName, int], Dict[NNCFNodeName, int]]:
        """
        Collects the active number of input and output channels (width) for each elastic layer in the graph.

        :return Dictionary with the number of input channels to convolution and linear layers:
            {node_name: input_channels_num}
            Dictionary with the number of output channels from convolution and linear layers:
            {node_name: output_channels_num}
        """
        graph = self._target_model.nncf.get_graph()
        in_channels, out_channels = get_prunable_layers_in_out_channels(graph)

        for group in self._pruned_module_groups_info.get_all_clusters():
            assert all(
                out_channels[group.elements[0].node_name] == out_channels[node.node_name] for node in group.elements
            )
            first_elastic_width_info: ElasticWidthInfo = group.elements[0]
            first_elastic_op: ElasticOutputWidthOp = first_elastic_width_info.elastic_op
            new_out_channels_num = first_elastic_op.get_active_width()
            num_of_pruned_elems = first_elastic_op.max_width - new_out_channels_num
            self._shape_pruning_processor.prune_cluster_shapes(
                group, num_of_pruned_elems, self._next_nodes, in_channels, out_channels
            )

        return in_channels, out_channels

    def resolve_conflicts_with_other_elasticities(
        self, config: ElasticWidthConfig, elasticity_handlers: ELASTICITY_HANDLERS_MAP
    ) -> ElasticWidthConfig:
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

    def get_group_id_by_node_name(self, node_name: NNCFNodeName) -> Optional[int]:
        """
        Provides a pruning group number that corresponds to a layer with the given name in graph. It's intended to be
        used for logging and visualization only.

        :param node_name: node name
        :return: group id
        """
        group_id = None
        for cluster in self._pruned_module_groups_info.get_all_clusters():
            for element in cluster.elements:
                if node_name == element.node_name:
                    group_id = cluster.id
        return group_id

    def get_external_importance(self, node_name: NNCFNodeName):
        """
        Return custom weight importance for the current node from external source.

        :param node_name: node name
        :return: importance tensor
        """
        assert should_consider_scope(
            node_name, ignored_scopes=None, target_scopes=self._external_importance.keys()
        ), f"Cannot match {node_name} in external weight importance data structure"
        return self._external_importance[node_name]

    def reorganize_weights(self) -> None:
        """
        Reorder output filters in descending order of their importance.
        """
        for node in self._propagation_graph.get_all_nodes():
            node.attributes.pop("output_mask", None)

        # 1. Calculate filter importance for all groups of prunable layers
        for group in self._pruned_module_groups_info.get_all_clusters():
            filters_num = torch.tensor([get_filters_num(minfo.module) for minfo in group.elements])
            assert torch.all(filters_num == filters_num[0])
            device = group.elements[0].module.weight.device

            cumulative_filters_importance = torch.zeros(filters_num[0]).to(device)
            # 1.1 Calculate cumulative importance for all filters in this group
            for minfo in group.elements:
                weight = minfo.module.weight
                if self._weights_normalizer_fn:
                    weight = self._weights_normalizer_fn(minfo.module.weight)
                if self._external_importance is not None:
                    weight = self.get_external_importance(minfo.node_name).to(device)
                filters_importance = self._filter_importance_fn(weight, minfo.module.target_weight_dim_for_compression)
                cumulative_filters_importance += filters_importance

            _, reorder_indexes = torch.sort(cumulative_filters_importance, dim=0, descending=True)
            device = get_model_device(self._target_model)
            reorder_indexes.to(device)
            # 1.2 Setup reorder indexes as output mask to reorganize filters
            for minfo in group.elements:
                node = self._propagation_graph.get_node_by_id(minfo.nncf_node_id)
                node.attributes["output_mask"] = PTNNCFTensor(reorder_indexes)

        # 2. Propagating masks across the graph
        reorder_algo = FilterReorderingAlgorithm(
            self._target_model,
            self._propagation_graph,
            PT_PRUNING_OPERATOR_METATYPES,
            PTNNCFPruningTensorProcessor,
        )
        reorder_algo.reorder_filters()

    def find_pairs_of_nodes_with_different_width(self, pairs_of_nodes: List[Tuple[str, str]]) -> List[int]:
        """
        Find pairs of nodes that have different output shapes. It's need to resolve conflict between elastic width and
        elastic depth. The conflicting situation happens when layers are trimmed and skipped independently.
        There might be a situation when channels are trimmed for the layers in the beginning of skipped block, and
        not trimmed in the end, thus violating the necessary condition for skipping a block - output shapes on
        block boundaries should be the same.

        :param pairs_of_nodes: list of pairs of node names
        :return: index of nodes pair in the given list that have different output shapes.
        """
        pair_indexes = []
        for idx, (start_node_name, end_node_name) in enumerate(pairs_of_nodes):
            start_node = self._propagation_graph.get_node_by_name(start_node_name)
            start_mask = start_node.attributes["output_mask"]
            end_node = self._propagation_graph.get_node_by_name(end_node_name)
            end_mask = end_node.attributes["output_mask"]

            all_start_output_shapes = self._propagation_graph.get_output_shapes_for_node(start_node_name)
            start_output_shape = list(OrderedDict.fromkeys(all_start_output_shapes))
            all_end_output_shape = self._propagation_graph.get_output_shapes_for_node(end_node_name)
            end_output_shape = list(OrderedDict.fromkeys(all_end_output_shape))

            start_width = ElasticWidthHandler.mask_to_width(start_mask)
            end_width = ElasticWidthHandler.mask_to_width(end_mask)

            if start_width is None and end_width is None and start_output_shape != end_output_shape:
                reason = f"it has a different shapes on boundaries: {start_output_shape} != {end_output_shape}"
            elif start_width is not None and end_width is not None and start_width != end_width:
                reason = f"it has a different width on boundaries: {start_width} != {end_width}"
            elif (start_width is None or end_width is None) and not (start_width is None and end_width is None):
                reason = (
                    f"it has empty width on one of the boundaries. "
                    f"Width: {start_width} vs {end_width}. Shapes: {start_output_shape} vs {end_output_shape}"
                )
            else:
                continue
            nncf_logger.debug(
                f"The block [\n\t{start_node_name},\n\t{end_node_name}\n]\n can`t be skipped, because {reason}"
            )
            pair_indexes.append(idx)
        return pair_indexes

    def _get_width_list_len(self, op: ElasticOutputWidthOp) -> int:
        N = len(op.width_list)
        if 0 < self._width_num_params_indicator < N:
            return self._width_num_params_indicator
        return N

    def _get_width_list(self, op: ElasticOutputWidthOp) -> List[int]:
        width_list_len = self._get_width_list_len(op)
        return op.width_list[:width_list_len]

    def _collect_ops_data_by_selection_rule(self, selection_rule: Callable) -> Dict[PruningGroupID, Any]:
        elastic_width_config = {}
        for cluster in self._pruned_module_groups_info.get_all_clusters():
            all_max_out_channels = {el.elastic_op.max_width for el in cluster.elements}
            if len(all_max_out_channels) != 1:
                raise nncf.InternalError("Invalid grouping of layers with different number of output channels")

            first_elastic_width_info = next(iter(cluster.elements))
            op = first_elastic_width_info.elastic_op
            selected_width = selection_rule(op)
            elastic_width_config[cluster.id] = selected_width
            nncf_logger.debug(f"Select width={cluster.id} for group #{selected_width}")
        return elastic_width_config

    @staticmethod
    def mask_to_width(mask: NNCFTensor) -> Optional[int]:
        """
        Decodes mask to a single integer. We assume that mask was constructed in a way that first N values are equal
        to 1, and the rest values are 0. The N encodes width value.

        :param mask: tensor with 1 and 0 values.
        :return: width value
        """
        result = None
        if mask is not None:
            actual_mask = mask.tensor
            mask_len = sum(actual_mask.size())
            width = int(sum(actual_mask))
            device = actual_mask.device
            ref_mask = ElasticWidthHandler._width_to_mask(width, mask_len, device).tensor
            assert torch.equal(
                ref_mask, actual_mask
            ), f"Invalid mask {actual_mask}: the first {width} values must be ones, the rest - zeros."
            result = width
        return result

    @staticmethod
    def _width_to_mask(active_width: int, max_width: int, device: torch.device) -> PTNNCFTensor:
        """
        Encodes width to tensor filled by 1 and 0. We intentionally construct mask in a way that first N values are
        equal to 1, and the rest values are 0. The N encodes width value. The mask tensors allows to fully reuse mask
        propagation algorithm without any changes, in contrast of implementing a separate width propagation algorithm.

        :param active_width: width value to encode in the mask.
        :param max_width: maximum width value.
        :param device: device that should have a mask tensor.
        :return: encoded mask tensor
        """
        mask = torch.ones(max_width).to(device)
        mask[active_width:].fill_(0)
        return PTNNCFTensor(mask)


class EWBuilderStateNames:
    GROUPED_NODE_NAMES_TO_PRUNE = "grouped_node_names_to_prune"
    OVERWRITE_GROUP_WIDTHS = "overwrite_groups_widths"
    ADD_DYNAMIC_INPUTS = "add_dynamic_inputs"


@ELASTICITY_BUILDERS.register(ElasticityDim.WIDTH)
class ElasticWidthBuilder(SingleElasticityBuilder):
    """
    Determines which modifications should be made to the original FP32 model in order to introduce elastic width
    to the model.
    """

    _state_names = EWBuilderStateNames

    def __init__(
        self,
        params: ElasticWidthParams,
        ignored_scopes: Optional[List[str]] = None,
        target_scopes: Optional[List[str]] = None,
    ):
        super().__init__(params, ignored_scopes, target_scopes)
        self._weights_normalizer = None
        self._overwriting_pruning_groups = params.overwrite_groups is not None
        self._grouped_node_names_to_prune: List[List[NNCFNodeName]] = (
            params.overwrite_groups if params.overwrite_groups is not None else []
        )
        self._overwrite_groups_widths = params.overwrite_groups_widths
        self._add_dynamic_inputs = params.add_dynamic_inputs
        self._params = params

    def build(self, target_model: NNCFNetwork) -> ElasticWidthHandler:
        """
        Creates modifications to the given NNCFNetwork for introducing elastic width and creates a handler object that
        can manipulate this elasticity.

        :param target_model: a target NNCFNetwork for adding modifications
        :return: a handler object that can manipulate the elastic width.
        """
        filter_importance_str = self._params.filter_importance
        filter_importance = FILTER_IMPORTANCE_FUNCTIONS.get(filter_importance_str, sum_filter)
        external_importance_path = self._params.external_importance_path

        graph = target_model.nncf.get_original_graph()
        device = next(target_model.parameters()).device

        if not self._grouped_node_names_to_prune:
            prunable_types = [NNCFConv2d, NNCFLinear]
            prune_operations_types = [pt.op_func_name for pt in prunable_types]
            types_of_grouping_ops = PTElementwisePruningOp.get_all_op_aliases()
            pruning_node_selector = PruningNodeSelector(
                PT_PRUNING_OPERATOR_METATYPES,
                prune_operations_types,
                types_of_grouping_ops,
                ignored_scopes=self._ignored_scopes,
                target_scopes=self._target_scopes,
                prune_first=True,
                prune_downsample_convs=True,
            )
            groups_of_nodes_to_prune = pruning_node_selector.create_pruning_groups(graph)
            for group in groups_of_nodes_to_prune.get_all_clusters():
                grouped_node_names = [node.node_name for node in group.elements]
                self._grouped_node_names_to_prune.append(grouped_node_names)

        transformation_commands = []
        pruned_module_groups_info = Clusterization[ElasticWidthInfo](id_fn=lambda x: x.node_name)
        node_name_vs_dynamic_input_width_op_map = OrderedDict()

        metatype_vs_elastic_op_creator = {
            PTModuleConv2dMetatype: self._create_elastic_conv_width_op,
            PTModuleDepthwiseConv2dSubtype: self._create_elastic_conv_width_op,
            PTModuleLinearMetatype: self._create_elastic_linear_width_op,
        }

        for i, grouped_node_names in enumerate(self._grouped_node_names_to_prune):
            group_minfos = []
            list_of_node_ids = []
            for node_name in grouped_node_names:
                node = graph.get_node_by_name(node_name)
                metatype = node.metatype
                list_of_node_ids.append(node.node_id)
                layer_attrs = node.layer_attributes
                if metatype not in metatype_vs_elastic_op_creator:
                    raise nncf.InternalError(f"Elastic width is not supported for {metatype}")
                elastic_op_creator = metatype_vs_elastic_op_creator[metatype]

                elastic_width_operation = elastic_op_creator(
                    layer_attrs,
                    node_name,
                    self._params,
                    self._overwrite_groups_widths[i] if self._overwriting_pruning_groups else [],
                )
                elastic_width_operation.to(device)
                update_conv_params_op = UpdateWeightAndOptionalBias(elastic_width_operation)
                transformation_commands.append(
                    PTInsertionCommand(
                        PTTargetPoint(TargetType.PRE_LAYER_OPERATION, target_node_name=node_name),
                        update_conv_params_op,
                        TransformationPriority.PRUNING_PRIORITY,
                    )
                )
                pruned_module = target_model.nncf.get_containing_module(node_name)
                assert isinstance(
                    pruned_module, (nn.Conv2d, nn.Linear)
                ), "currently prune only 2D Convolutions and Linear layers"

                group_minfos.append(
                    ElasticWidthInfo(
                        node_name=node_name,
                        module=pruned_module,
                        elastic_op=elastic_width_operation,
                        node_id=node.node_id,
                        is_depthwise=is_prunable_depthwise_conv(node),
                    )
                )

            cluster = Cluster[ElasticWidthInfo](i, group_minfos, list_of_node_ids)
            pruned_module_groups_info.add_cluster(cluster)

        metatype_vs_dynamic_input_op_creator = {
            PTModuleConv2dMetatype: self._create_dynamic_conv_input_op,
            PTModuleDepthwiseConv2dSubtype: self._create_dynamic_dw_conv_input_op,
            PTModuleBatchNormMetatype: self._create_dynamic_bn_input_op,
            PTModuleLayerNormMetatype: self._create_dynamic_ln_input_op,
            PTModuleLinearMetatype: self._create_dynamic_linear_input_op,
        }
        for metatype, op_creator in metatype_vs_dynamic_input_op_creator.items():
            nodes = graph.get_nodes_by_metatypes([metatype])
            for node in nodes:
                node_name = node.node_name
                nncf_logger.debug(f"Adding Dynamic Input Op for {metatype.name} in scope: {node_name}")
                layer_attrs = node.layer_attributes
                update_module_params = op_creator(layer_attrs, node_name).to(device)
                node_name_vs_dynamic_input_width_op_map[node_name] = update_module_params.op
                transformation_commands.append(
                    PTInsertionCommand(
                        PTTargetPoint(TargetType.PRE_LAYER_OPERATION, target_node_name=node_name),
                        update_module_params,
                        priority=TransformationPriority.DEFAULT_PRIORITY,
                    )
                )

        return ElasticWidthHandler(
            target_model,
            filter_importance,
            external_importance_path,
            self._weights_normalizer,
            node_name_vs_dynamic_input_width_op_map,
            pruned_module_groups_info,
            transformation_commands,
            self._add_dynamic_inputs,
        )

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Initializes object from the state.

        :param state: Output of `get_state()` method.
        """
        params_from_state = state[SingleElasticityBuilder._state_names.ELASTICITY_PARAMS]
        params = ElasticWidthParams.from_state(params_from_state)
        if self._params and self._params != params:
            nncf_logger.warning(
                "Different elasticity parameters were provided in two places: on init and on loading "
                "state. The one from state is taken by ignoring the ones from init."
            )
        self._params = params
        self._grouped_node_names_to_prune = state[self._state_names.GROUPED_NODE_NAMES_TO_PRUNE]

        if params_from_state.get(self._state_names.OVERWRITE_GROUP_WIDTHS, None) is not None:
            self._overwrite_groups_widths = params_from_state[self._state_names.OVERWRITE_GROUP_WIDTHS]
            self._overwriting_pruning_groups = True
            if len(self._grouped_node_names_to_prune) != len(self._overwrite_groups_widths):
                raise nncf.InternalError("Mismatch between number of groups for pruning and their corresponding widths")
        if params_from_state.get(self._state_names.ADD_DYNAMIC_INPUTS, None) is not None:
            self._add_dynamic_inputs = params_from_state[self._state_names.ADD_DYNAMIC_INPUTS]

    def get_state(self) -> Dict[str, Any]:
        """
        Returns a dictionary with Python data structures (dict, list, tuple, str, int, float, True, False, None) that
        represents state of the object.

        :return: state of the object
        """
        return {
            SingleElasticityBuilder._state_names.ELASTICITY_PARAMS: self._params.get_state(),
            self._state_names.GROUPED_NODE_NAMES_TO_PRUNE: self._grouped_node_names_to_prune,
        }

    @staticmethod
    def _create_elastic_conv_width_op(
        conv_layer_attrs: BaseLayerAttributes,
        node_name: str,
        params: ElasticWidthParams,
        fixed_width_list: Optional[List[int]] = None,
    ) -> ElasticOutputWidthConv2DOp:
        assert isinstance(conv_layer_attrs, ConvolutionLayerAttributes)
        nncf_logger.debug(f"Adding Dynamic Conv2D Layer in scope: {str(node_name)}")
        if fixed_width_list is None:
            fixed_width_list = []
        return ElasticOutputWidthConv2DOp(
            conv_layer_attrs.out_channels,
            node_name,
            params,
            fixed_width_list=fixed_width_list,
        )

    @staticmethod
    def _create_elastic_linear_width_op(
        linear_layer_attrs: BaseLayerAttributes,
        node_name: str,
        params: ElasticWidthParams,
        fixed_width_list: Optional[List[int]] = None,
    ) -> ElasticOutputWidthLinearOp:
        assert isinstance(linear_layer_attrs, LinearLayerAttributes)
        if fixed_width_list is None:
            fixed_width_list = []
        nncf_logger.debug(f"Adding Dynamic Linear Layer in scope: {str(node_name)}")
        return ElasticOutputWidthLinearOp(
            linear_layer_attrs.out_features,
            node_name,
            params,
            fixed_width_list=fixed_width_list,
        )

    @staticmethod
    def _create_dynamic_conv_input_op(conv_layer_attrs: BaseLayerAttributes, node_name: str) -> UpdateWeight:
        assert isinstance(conv_layer_attrs, ConvolutionLayerAttributes)
        dynamic_conv_input_op = ElasticInputWidthConvOp(max_width=conv_layer_attrs.in_channels, node_name=node_name)
        return UpdateWeight(dynamic_conv_input_op)

    @staticmethod
    def _create_dynamic_dw_conv_input_op(conv_layer_attrs: BaseLayerAttributes, node_name: str) -> UpdateNumGroups:
        assert isinstance(conv_layer_attrs, ConvolutionLayerAttributes)
        dynamic_dw_conv_input_op = ElasticInputWidthDWConvOp(max_width=conv_layer_attrs.groups, node_name=node_name)
        return UpdateNumGroups(dynamic_dw_conv_input_op)

    @staticmethod
    def _create_dynamic_bn_input_op(generic_layer_attrs: BaseLayerAttributes, node_name: str) -> UpdateBatchNormParams:
        assert isinstance(generic_layer_attrs, GenericWeightedLayerAttributes)
        dynamic_bn_input_op = ElasticInputWidthBatchNormOp(
            max_width=generic_layer_attrs.get_num_filters(), node_name=node_name
        )
        return UpdateBatchNormParams(dynamic_bn_input_op)

    @staticmethod
    def _create_dynamic_ln_input_op(generic_layer_attrs: BaseLayerAttributes, node_name: str) -> UpdateLayerNormParams:
        assert isinstance(generic_layer_attrs, GenericWeightedLayerAttributes)
        dynamic_ln_input_op = ElasticInputWidthLayerNormOp(
            max_width=generic_layer_attrs.get_num_filters(), node_name=node_name
        )
        return UpdateLayerNormParams(dynamic_ln_input_op)

    @staticmethod
    def _create_dynamic_linear_input_op(linear_layer_attrs: BaseLayerAttributes, node_name: str) -> UpdateWeight:
        assert isinstance(linear_layer_attrs, LinearLayerAttributes)
        dynamic_linear_input_op = ElasticInputWidthLinearOp(
            max_width=linear_layer_attrs.in_features, node_name=node_name
        )
        return UpdateWeight(dynamic_linear_input_op)
