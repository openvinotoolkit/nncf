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
import inspect
from collections import OrderedDict
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import OrderedDict as OrderedDictType
from typing import Tuple

from nncf.common.pruning.utils import count_flops_and_weights_per_node
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.base_handler import ElasticityConfig
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.base_handler import ElasticityHandler
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.base_handler import SingleElasticityHandler
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elastic_depth import ElasticDepthHandler
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elastic_depth import ElasticDepthSearchSpace
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elastic_kernel import ElasticKernelHandler
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elastic_kernel import ElasticKernelSearchSpace
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elastic_width import ElasticWidthHandler
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elastic_width import ElasticWidthSearchSpace
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elasticity_dim import ElasticityDim
from nncf.torch.graph.operator_metatypes import PTConv1dMetatype
from nncf.torch.graph.operator_metatypes import PTConv2dMetatype
from nncf.torch.graph.operator_metatypes import PTConv3dMetatype
from nncf.torch.graph.operator_metatypes import PTConvTranspose2dMetatype
from nncf.torch.graph.operator_metatypes import PTConvTranspose3dMetatype
from nncf.torch.graph.operator_metatypes import PTDepthwiseConv1dSubtype
from nncf.torch.graph.operator_metatypes import PTDepthwiseConv2dSubtype
from nncf.torch.graph.operator_metatypes import PTDepthwiseConv3dSubtype
from nncf.torch.graph.operator_metatypes import PTLinearMetatype
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.pruning.utils import collect_input_shapes
from nncf.torch.pruning.utils import collect_output_shapes

SubnetConfig = OrderedDictType[ElasticityDim, ElasticityConfig]


class MEHandlerStateNames:
    IS_HANDLER_ENABLED_MAP = 'is_handler_enabled_map'
    STATES_OF_HANDLERS = 'states_of_handlers'


class MultiElasticityHandler(ElasticityHandler):
    """
    An interface for handling multiple elasticity in the network. The elasticity defines variable values in properties
    of the layers or the network, e.g. variable number of channels in the Conv or variable number of layers in the
    network. By applying elasticity it's possible to derive a smaller models (Subnets) that have some elements in
    common with the original model.
    The interface defines methods for activation Subnets.
    """
    _state_names = MEHandlerStateNames

    def __init__(self,
                 handlers: OrderedDictType[ElasticityDim, SingleElasticityHandler],
                 target_model: NNCFNetwork):
        self._handlers = handlers
        self._target_model = target_model
        self._is_handler_enabled_map = {elasticity_dim: True for elasticity_dim in handlers}
        self.activate_supernet()

    @property
    def width_search_space(self) -> ElasticWidthSearchSpace:
        return self.width_handler.get_search_space()

    @property
    def kernel_search_space(self) -> ElasticKernelSearchSpace:
        return self.kernel_handler.get_search_space()

    @property
    def depth_search_space(self) -> ElasticDepthSearchSpace:
        return self.depth_handler.get_search_space()

    @property
    def width_handler(self) -> Optional[ElasticWidthHandler]:
        return self._get_handler_by_elasticity_dim(ElasticityDim.WIDTH)

    @property
    def kernel_handler(self) -> Optional[ElasticKernelHandler]:
        return self._get_handler_by_elasticity_dim(ElasticityDim.KERNEL)

    @property
    def depth_handler(self) -> Optional[ElasticDepthHandler]:
        return self._get_handler_by_elasticity_dim(ElasticityDim.DEPTH)

    def get_available_elasticity_dims(self) -> List[ElasticityDim]:
        """
        :return: list of available elasticity dimension. E.g. it's possible to have a single elasticity like Elastic
        Depth or all possible elasticities (Elastic Depth/Width/Kernel) added to the model.
        """
        return list(self._handlers)

    def get_active_config(self) -> SubnetConfig:
        """
        Forms an elasticity configuration that describes currently activated Subnet across all elasticities.

        :return: elasticity configuration
        """
        return self._collect_handler_data_by_method_name(self._get_current_method_name())

    def get_random_config(self) -> SubnetConfig:
        """
        Forms an elasticity configuration that describes a Subnet with randomly chosen elastic values across all
        elasticities.

        :return: elasticity configuration
        """
        return self._collect_handler_data_by_method_name(self._get_current_method_name())

    def get_minimum_config(self) -> SubnetConfig:
        """
        Forms an elasticity configuration that describes a Subnet with minimum elastic values across all elasticities.

        :return: elasticity configuration
        """
        return self._collect_handler_data_by_method_name(self._get_current_method_name())

    def get_maximum_config(self) -> SubnetConfig:
        """
        Forms an elasticity configuration that describes a Subnet with maximum elastic values across all elasticities.

        :return: elasticity configuration
        """
        return self._collect_handler_data_by_method_name(self._get_current_method_name())

    def activate_supernet(self) -> None:
        """
        Activates the Supernet - the original network to which elasticity was applied.
        """
        self._collect_handler_data_by_method_name(self._get_current_method_name())

    def activate_subnet_for_config(self, config: SubnetConfig) -> None:
        """
        Activates a Subnet that corresponds to the given elasticity configuration

        :param config: elasticity configuration
        """
        active_handlers = {
            dim: self._handlers[dim] for dim in self._handlers if self._is_handler_enabled_map[dim]
        }
        for handler_id, handler in self._handlers.items():
            if handler_id in config:
                sub_config = config[handler_id]
                other_active_handlers = dict(filter(lambda pair: pair[0] != handler_id, active_handlers.items()))
                resolved_config = handler.resolve_conflicts_with_other_elasticities(sub_config, other_active_handlers)
                handler.activate_subnet_for_config(resolved_config)

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Initializes object from the state.

        :param state: Output of `get_state()` method.
        """
        states_of_handlers = state[self._state_names.STATES_OF_HANDLERS]
        is_handler_enabled_map = state[self._state_names.IS_HANDLER_ENABLED_MAP]

        for dim_str, handler_state in states_of_handlers.items():
            dim = ElasticityDim.from_str(dim_str)
            if dim in self._handlers:
                self._handlers[dim].load_state(handler_state)

        for dim_str, is_enabled in is_handler_enabled_map.items():
            dim = ElasticityDim.from_str(dim_str)
            self._is_handler_enabled_map[dim] = is_enabled

    def get_state(self) -> Dict[str, Any]:
        """
        Returns a dictionary with Python data structures (dict, list, tuple, str, int, float, True, False, None) that
        represents state of the object.

        :return: state of the object
        """
        states_of_handlers = {dim.value: handler.get_state() for dim, handler in self._handlers.items()}
        is_handler_enabled_map = {dim.value: is_enabled for dim, is_enabled in self._is_handler_enabled_map.items()}
        return {
            self._state_names.STATES_OF_HANDLERS: states_of_handlers,
            self._state_names.IS_HANDLER_ENABLED_MAP: is_handler_enabled_map
        }

    def enable_all(self) -> None:
        """
        Enables all elasticities for being selected on sampling subnets.
        """
        self._is_handler_enabled_map = {elasticity_dim: True for elasticity_dim in self._is_handler_enabled_map}

    def disable_all(self) -> None:
        """
        Disables all elasticities for being selected on sampling subnets.
        """
        self._is_handler_enabled_map = {elasticity_dim: False for elasticity_dim in self._is_handler_enabled_map}

    def enable_elasticity(self, dim: ElasticityDim):
        """
        Enables elasticity for being selected on sampling subnets by a given type.
        """
        self._is_handler_enabled_map[dim] = True

    def disable_elasticity(self, dim: ElasticityDim):
        """
        Disables elasticity for being selected on sampling subnets by a given type.
        """
        self._is_handler_enabled_map[dim] = False

    def count_flops_and_weights_for_active_subnet(self) -> Tuple[int, int]:
        """
        :return: FLOPs and the number weights and in the model for convolution and fully connected layers.
        """
        GENERAL_CONV_LAYER_METATYPES = [
            PTConv1dMetatype,
            PTDepthwiseConv1dSubtype,
            PTConv2dMetatype,
            PTDepthwiseConv2dSubtype,
            PTConv3dMetatype,
            PTDepthwiseConv3dSubtype,
            PTConvTranspose2dMetatype,
            PTConvTranspose3dMetatype
        ]
        LINEAR_LAYER_METATYPES = [
            PTLinearMetatype
        ]

        graph = self._target_model.get_graph()
        modules_out_shapes = collect_output_shapes(graph)
        modules_in_shapes = collect_input_shapes(graph)

        kernel_sizes = None
        if self.kernel_handler is not None:
            kernel_sizes = self.kernel_handler.get_active_kernel_sizes_per_node()

        names_of_skipped_nodes = None
        if self.depth_handler is not None:
            names_of_skipped_nodes = self.depth_handler.get_names_of_skipped_nodes()

        input_width_values, output_width_values = None, None
        if self.width_handler is not None:
            input_width_values, output_width_values = self.width_handler.get_active_in_out_width_values()

        flops_pers_node, num_weights_per_node = count_flops_and_weights_per_node(
            graph=graph,
            input_shapes=modules_in_shapes,
            output_shapes=modules_out_shapes,
            input_channels=input_width_values,
            output_channels=output_width_values,
            kernel_sizes=kernel_sizes,
            op_addresses_to_skip=names_of_skipped_nodes,
            conv_op_metatypes=GENERAL_CONV_LAYER_METATYPES,
            linear_op_metatypes=LINEAR_LAYER_METATYPES,
        )

        flops = sum(flops_pers_node.values())
        num_weights = sum(num_weights_per_node.values())
        return flops, num_weights

    def _get_handler_by_elasticity_dim(self, dim: ElasticityDim) -> Optional[SingleElasticityHandler]:
        result = None
        if dim in self._handlers:
            result = self._handlers[dim]
        return result

    def _collect_handler_data_by_method_name(self, method_name: str) -> OrderedDictType[ElasticityDim, Any]:
        result = OrderedDict()
        for elasticity_dim, handler in self._handlers.items():
            if self._is_handler_enabled_map[elasticity_dim]:
                handler_method = getattr(handler, method_name)
                data = handler_method()
                result[elasticity_dim] = data
        return result

    @staticmethod
    def _get_current_method_name() -> str:
        return inspect.stack()[1].function
