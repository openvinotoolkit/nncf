"""
 Copyright (c) 2019-2021 Intel Corporation
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

from nncf.common.pruning.utils import count_flops_and_weights_per_node
from nncf.torch.nas.bootstrapNAS.elasticity.base_handler import ElasticConfig
from nncf.torch.nas.bootstrapNAS.elasticity.base_handler import ElasticHandler
from nncf.torch.nas.bootstrapNAS.elasticity.base_handler import SingleElasticHandler
from nncf.torch.nas.bootstrapNAS.elasticity.elastic_depth import ElasticDepthHandler
from nncf.torch.nas.bootstrapNAS.elasticity.elastic_depth import ElasticDepthSearchSpace
from nncf.torch.nas.bootstrapNAS.elasticity.elastic_kernel import ElasticKernelHandler
from nncf.torch.nas.bootstrapNAS.elasticity.elastic_kernel import ElasticKernelSearchSpace
from nncf.torch.nas.bootstrapNAS.elasticity.elastic_width import ElasticWidthHandler
from nncf.torch.nas.bootstrapNAS.elasticity.elastic_width import ElasticWidthSearchSpace
from nncf.torch.nas.bootstrapNAS.elasticity.elasticity_dim import ElasticityDim

SubnetConfig = OrderedDictType[ElasticityDim, ElasticConfig]


class MultiElasticityHandler(ElasticHandler):
    def __init__(self, handlers: OrderedDictType[ElasticityDim, SingleElasticHandler]):
        self._handlers = handlers

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

    def get_enabled_elasticity_dims(self) -> List[ElasticityDim]:
        return list(self._handlers)

    def get_active_config(self) -> SubnetConfig:
        return self._collect_handler_data_by_method_name(self._get_current_method_name())

    def activate_random_subnet(self) -> SubnetConfig:
        return self._collect_handler_data_by_method_name(self._get_current_method_name())

    def activate_minimal_subnet(self):
        return self._collect_handler_data_by_method_name(self._get_current_method_name())

    def activate_maximal_subnet(self) -> SubnetConfig:
        return self._collect_handler_data_by_method_name(self._get_current_method_name())

    def activate_supernet(self):
        self._collect_handler_data_by_method_name(self._get_current_method_name())

    def activate(self):
        self._collect_handler_data_by_method_name(self._get_current_method_name(), ignore_is_active=True)

    def deactivate(self):
        self._collect_handler_data_by_method_name(self._get_current_method_name())

    def activate_elasticity(self, dim: ElasticityDim):
        self._get_handler_by_elasticity_dim(dim).activate()

    def deactivate_elasticity(self, dim: ElasticityDim):
        self._get_handler_by_elasticity_dim(dim).deactivate()

    def set_config(self, subnet_config: SubnetConfig):
        for handler_id, handler in self._handlers.items():
            if handler_id in subnet_config:
                config = subnet_config[handler_id]
                handler.set_config(config)

    def count_flops_and_weights_for_active_subnet(self):
        kwargs = {}
        for handler in self._handlers.values():
            kwargs.update(handler.get_kwargs_for_flops_counting())

        flops_pers_node, num_weights_per_node = count_flops_and_weights_per_node(**kwargs)

        flops = sum(flops_pers_node.values())
        num_weights = sum(num_weights_per_node.values())
        return flops, num_weights

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Loads the compression controller state from the map of algorithm name to the dictionary with state attributes.

        :param state: map of the algorithm name to the dictionary with the corresponding state attributes.
        """
        for dim_str, handler_state in state.items():
            dim = ElasticityDim.from_str(dim_str)
            if dim in self._handlers:
                self._handlers[dim].load_state(handler_state)

    def get_state(self) -> Dict[str, Any]:
        """
        Returns compression controller state, which is the map of the algorithm name to the dictionary with the
        corresponding state attributes.

        :return: The compression controller state.
        """
        return {dim.value: handler.get_state() for dim, handler in self._handlers.items()}

    def _get_handler_by_elasticity_dim(self, dim: ElasticityDim) -> Optional[SingleElasticHandler]:
        if dim in self._handlers:
            return self._handlers[dim]

    def _collect_handler_data_by_method_name(self, method_name,
                                             ignore_is_active=False) -> OrderedDictType[ElasticityDim, Any]:
        result = OrderedDict()
        for elasticity_dim, handler in self._handlers.items():
            if handler.is_active or ignore_is_active:
                handler_method = getattr(handler, method_name)
                data = handler_method()
                result[elasticity_dim] = data
        return result

    @staticmethod
    def _get_current_method_name() -> str:
        return inspect.stack()[1].function
