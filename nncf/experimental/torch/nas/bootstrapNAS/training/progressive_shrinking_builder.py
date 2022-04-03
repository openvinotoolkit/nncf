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
from typing import Any
from typing import Dict
from typing import List

from nncf import NNCFConfig
from nncf.common.initialization.batchnorm_adaptation import BatchnormAdaptationAlgorithm
from nncf.config.extractors import get_bn_adapt_algo_kwargs
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elasticity_builder import ElasticityBuilder
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elasticity_dim import ElasticityDim
from nncf.experimental.torch.nas.bootstrapNAS.training.progressive_shrinking_controller import \
    ProgressiveShrinkingController
from nncf.experimental.torch.nas.bootstrapNAS.training.scheduler import NASSchedulerParams
from nncf.torch.algo_selector import PT_COMPRESSION_ALGORITHMS
from nncf.torch.compression_method_api import PTCompressionAlgorithmBuilder
from nncf.torch.graph.transformations.layout import PTTransformationLayout
from nncf.torch.nncf_network import NNCFNetwork


class PSBuilderStateNames:
    ELASTICITY_BUILDER_STATE = 'elasticity_builder_state'
    PROGRESSIVITY_OF_ELASTICITY = 'progressivity_of_elasticity'


@PT_COMPRESSION_ALGORITHMS.register('progressive_shrinking')
class ProgressiveShrinkingBuilder(PTCompressionAlgorithmBuilder):
    """
    Determines which modifications should be made to the original FP32 model in
    order to train a supernet using Progressive Shrinking procedure from OFA (https://arxiv.org/abs/1908.09791).
    Operates on an NNCFNetwork object wrapping a target PyTorch model (torch.nn.Module).
    """

    DEFAULT_PROGRESSIVITY = [ElasticityDim.KERNEL, ElasticityDim.DEPTH, ElasticityDim.WIDTH]
    _state_names = PSBuilderStateNames

    def __init__(self, nncf_config: NNCFConfig, should_init: bool = True):
        super().__init__(nncf_config, should_init)
        bn_adapt_params = self._algo_config.get('batchnorm_adaptation', {})
        bn_adapt_algo_kwargs = get_bn_adapt_algo_kwargs(nncf_config, bn_adapt_params)
        self._bn_adaptation = BatchnormAdaptationAlgorithm(**bn_adapt_algo_kwargs)

        default_progressivity = map(lambda x: x.value, self.DEFAULT_PROGRESSIVITY)
        progressivity_of_elasticity = self._algo_config.get('progressivity_of_elasticity', default_progressivity)
        self._progressivity_of_elasticity = list(map(ElasticityDim.from_str, progressivity_of_elasticity))
        self._elasticity_builder = ElasticityBuilder(self.config, self.should_init)

    @staticmethod
    def check_elasticity_dims_consistency(available_elasticity_dims: List[ElasticityDim],
                                          progressivity_of_elasticity: List[ElasticityDim]) -> None:
        """
        Verifies that progressivity of elasticity is specified for all available elasticity dimensions.

        :param available_elasticity_dims: list of available elasticity dimension
        :param progressivity_of_elasticity: specifies in which order elasticity should be added
        """
        for dim in available_elasticity_dims:
            if dim not in progressivity_of_elasticity:
                raise ValueError(f'Invalid elasticity dimension {dim} specified as available in `elasticity` section.'
                                 f' This dimension is not part of the progressivity_of_elasticity='
                                 f'{progressivity_of_elasticity} which defines order of adding elasticity dimension'
                                 f' by going from one training stage to another.')

    def initialize(self, model: NNCFNetwork) -> None:
        """
        Initialize model parameters before training

        :param model: The model with additional modifications necessary to enable
            algorithm-specific compression during fine-tuning.
        """

    def _get_algo_specific_config_section(self) -> Dict:
        return self.config.get('bootstrapNAS', {}).get('training', {})

    def _build_controller(self, model: NNCFNetwork) -> 'ProgressiveShrinkingController':
        elasticity_ctrl = self._elasticity_builder.build_controller(model)
        schedule_params = NASSchedulerParams.from_config(self._algo_config.get('schedule', {}))
        return ProgressiveShrinkingController(
            model, elasticity_ctrl, self._bn_adaptation, self._progressivity_of_elasticity, schedule_params)

    def _get_transformation_layout(self, target_model: NNCFNetwork) -> PTTransformationLayout:
        available_elasticity_dims = self._elasticity_builder.get_available_elasticity_dims()
        self.check_elasticity_dims_consistency(available_elasticity_dims, self._progressivity_of_elasticity)
        return self._elasticity_builder.get_transformation_layout(target_model)

    def _get_state_without_name(self) -> Dict[str, Any]:
        """
        Implementation of get_state that returns state without builder name.

        :return: Returns a dictionary with Python data structures
            (dict, list, tuple, str, int, float, True, False, None) that represents state of the object.
        """
        return {
            self._state_names.ELASTICITY_BUILDER_STATE: self._elasticity_builder.get_state(),
            self._state_names.PROGRESSIVITY_OF_ELASTICITY: [d.value for d in self._progressivity_of_elasticity]
        }

    def _load_state_without_name(self, state_without_name: Dict[str, Any]):
        """
        Implementation of load state that takes state without builder name.

        :param state_without_name: Output of `_get_state_without_name()` method.
        """
        elasticity_builder_state = state_without_name[self._state_names.ELASTICITY_BUILDER_STATE]
        self._elasticity_builder.load_state(elasticity_builder_state)
        progressivity_of_elasticity = state_without_name[self._state_names.PROGRESSIVITY_OF_ELASTICITY]
        # No conflict resolving with the related config options, parameters are overridden by compression state
        self._progressivity_of_elasticity = [ElasticityDim.from_str(dim) for dim in progressivity_of_elasticity]
