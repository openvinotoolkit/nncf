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
from collections import OrderedDict
from typing import Any, Dict, List

from nncf import NNCFConfig
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.base_handler import SingleElasticityBuilder
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.base_handler import create_elasticity_builder_from_config
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elasticity_controller import ElasticityController
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elasticity_dim import ElasticityDim
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.multi_elasticity_handler import MultiElasticityHandler
from nncf.torch.algo_selector import PT_COMPRESSION_ALGORITHMS
from nncf.torch.compression_method_api import PTCompressionAlgorithmBuilder
from nncf.torch.graph.transformations.layout import PTTransformationLayout
from nncf.torch.nncf_network import NNCFNetwork


class EBuilderStateNames:
    AVAILABLE_ELASTICITY_DIMS = "available_elasticity_dims"
    BUILDER_STATES = "builder_states"


@PT_COMPRESSION_ALGORITHMS.register("elasticity")
class ElasticityBuilder(PTCompressionAlgorithmBuilder):
    """
    Determines which modifications should be made to the original FP32 model in order to introduce elasticity
    to the model.
    """

    _state_names = EBuilderStateNames

    # NOTE: This is the order of activation elasticity dimensions when multiple of them are enabled.
    # Don't confuse with the order of adding elasticity dimension on training stages (progressiveness of
    # elasticity). For vanilla progressive shrinking the stages order is the following:
    #   1st stage: kernel
    #   2nd stage: kernel + depth
    #   3rd stage: kernel + depth + width
    # The execution order is orthogonal to this.
    # Though the order of kernel/width operations shouldn't lead to a different result mathematically,
    # there may be a minor floating-point error in the 6th sign. To make behavior stable, elastic kernel is always
    # applied after width.
    # Depth goes after width, because currently only depth handler knows about shapes after setting config by width
    # handler and don't skip some blocks if shapes on the block boundaries are not the same. Potentially, we could
    # support an alternative strategy, when width handler knows about skipped blocks and ,
    # but currently, it's not supported.
    ALL_DIMS_IN_EXECUTION_ORDER = [ElasticityDim.WIDTH, ElasticityDim.KERNEL, ElasticityDim.DEPTH]

    def __init__(self, nncf_config: NNCFConfig, should_init: bool = True):
        super().__init__(nncf_config, should_init)
        self._multi_elasticity_handler = None
        # TODO(nlyalyus): ignored/target scope is not supported (ticket 68052)
        self._ignored_scopes = self.config.get("ignored_scopes", None)
        self._target_scopes = self.config.get("target_scopes", None)
        self._multi_elasticity_handler_state = None

        all_elasticity_dims = {e.value for e in ElasticityDim}
        available_elasticity_dims_str = self._algo_config.get("available_elasticity_dims", all_elasticity_dims)
        self._available_elasticity_dims = list(map(ElasticityDim, available_elasticity_dims_str))
        self._elasticity_builders: Dict[ElasticityDim, SingleElasticityBuilder] = OrderedDict()
        self._builder_states = None

    def initialize(self, model: NNCFNetwork) -> None:
        """
        Initialize model parameters before training

        :param model: The model with additional modifications necessary to enable
            algorithm-specific compression during fine-tuning.
        """

    def get_available_elasticity_dims(self) -> List[ElasticityDim]:
        """
        :return: list of available elasticity dimensions
        """
        return self._available_elasticity_dims

    def _get_algo_specific_config_section(self) -> Dict:
        return self.config.get("bootstrapNAS", {}).get("training", {}).get("elasticity", {})

    def _build_controller(self, model: NNCFNetwork) -> "ElasticityController":
        """
        Simple implementation of building controller without setting builder state and loading controller's one.

        :param model: The model with additional modifications necessary to enable
            algorithm-specific compression during fine-tuning.
        :return: The instance of the `ElasticityController`.
        """
        return ElasticityController(model, self._algo_config, self._multi_elasticity_handler)

    def _get_transformation_layout(self, target_model: NNCFNetwork) -> PTTransformationLayout:
        sorted_elasticity_dims = list(
            filter(lambda x: x in self._available_elasticity_dims, self.ALL_DIMS_IN_EXECUTION_ORDER)
        )
        ignored_scopes = self._ignored_scopes
        target_scopes = self._target_scopes

        for elasticity_dim in sorted_elasticity_dims:
            elasticity_config = self._algo_config.get(elasticity_dim.value, {})
            elasticity_builder = create_elasticity_builder_from_config(
                elasticity_config, elasticity_dim, ignored_scopes, target_scopes
            )
            self._elasticity_builders[elasticity_dim] = elasticity_builder

        if self._builder_states is not None:
            for dim_str, builder_state in self._builder_states.items():
                dim = ElasticityDim(dim_str)
                if dim in self._elasticity_builders:
                    self._elasticity_builders[dim].load_state(builder_state)

        elasticity_handlers = OrderedDict()
        for dim, builder in self._elasticity_builders.items():
            handler = builder.build(target_model)
            elasticity_handlers[dim] = handler
        self._multi_elasticity_handler = MultiElasticityHandler(elasticity_handlers, target_model)

        layout = PTTransformationLayout()
        for handler in elasticity_handlers.values():
            commands = handler.get_transformation_commands()
            for command in commands:
                layout.register(command)
        return layout

    def _get_state_without_name(self) -> Dict[str, Any]:
        """
        Implementation of get_state that returns state without builder name.

        :return: Returns a dictionary with Python data structures
            (dict, list, tuple, str, int, float, True, False, None) that represents state of the object.
        """
        builder_states = {dim.value: builder.get_state() for dim, builder in self._elasticity_builders.items()}
        available_elasticity_dims_state = list(map(lambda x: x.value, self.get_available_elasticity_dims()))
        return {
            self._state_names.BUILDER_STATES: builder_states,
            self._state_names.AVAILABLE_ELASTICITY_DIMS: available_elasticity_dims_state,
        }

    def _load_state_without_name(self, state_without_name: Dict[str, Any]):
        """
        Implementation of load state that takes state without builder name.

        :param state_without_name: Output of `_get_state_without_name()` method.
        """
        self._builder_states = state_without_name[self._state_names.BUILDER_STATES]
        available_elasticity_dims_state = state_without_name[self._state_names.AVAILABLE_ELASTICITY_DIMS]

        # No conflict resolving with the related config options, parameters are overridden by compression state
        self._available_elasticity_dims = list(map(ElasticityDim, available_elasticity_dims_state))
