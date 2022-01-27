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
from typing import Any
from typing import Dict
from typing import List

from nncf.api.compression import CompressionLoss
from nncf.api.compression import CompressionScheduler
from nncf.api.compression import CompressionStage
from nncf.common.initialization.batchnorm_adaptation import BatchnormAdaptationAlgorithm
from nncf.common.statistics import NNCFStatistics
from nncf.common.utils.logger import logger as nncf_logger
from nncf.torch.algo_selector import ZeroCompressionLoss
from nncf.torch.nas.bootstrapNAS.elasticity.elasticity_controller import ElasticityController
from nncf.torch.nas.bootstrapNAS.elasticity.elasticity_dim import ElasticityDim
from nncf.torch.nas.bootstrapNAS.training.base_training import BNASTrainingController
from nncf.torch.nas.bootstrapNAS.training.scheduler import BootstrapNASScheduler
from nncf.torch.nas.bootstrapNAS.training.stage_descriptor import StageDescriptor
from nncf.torch.nncf_network import NNCFNetwork


class PSControllerStateNames:
    ELASTICITY_CONTROLLER_STATE = 'elasticity_controller_compression_state'


class ProgressiveShrinkingController(BNASTrainingController):
    _ps_state_names = PSControllerStateNames

    def __init__(self, target_model: NNCFNetwork,
                 elasticity_ctrl: ElasticityController,
                 bn_adaptation: BatchnormAdaptationAlgorithm,
                 progressivity_of_elasticity: List[ElasticityDim],
                 schedule_params: Dict[str, Any]):
        super().__init__(target_model)
        self._elasticity_ctrl = elasticity_ctrl
        self._bn_adaptation = bn_adaptation
        self._progressivity_of_elasticity = progressivity_of_elasticity
        self._target_model = target_model
        width_handler = self.multi_elasticity_handler.width_handler
        if width_handler is not None:
            width_handler.width_num_params_indicator = 1
        self._loss = ZeroCompressionLoss(next(target_model.parameters()).device)
        self._enabled_elasticity_dims = self.multi_elasticity_handler.get_enabled_elasticity_dims()
        self._scheduler = BootstrapNASScheduler(self, schedule_params, self._enabled_elasticity_dims,
                                                self._progressivity_of_elasticity)

    @property
    def multi_elasticity_handler(self):
        return self._elasticity_ctrl.multi_elasticity_handler

    @property
    def elasticity_controller(self):
        return self._elasticity_ctrl

    @property
    def loss(self) -> CompressionLoss:
        return self._loss

    @property
    def scheduler(self) -> CompressionScheduler:
        return self._scheduler

    def step(self):
        self.multi_elasticity_handler.activate_random_subnet()
        nncf_logger.debug(
            'Active config: {}'.format(self.multi_elasticity_handler.get_active_config()))

    def prepare_for_validation(self):
        self._run_batchnorm_adaptation(self._target_model)

    def get_total_num_epochs(self) -> int:
        return self._scheduler.get_total_training_epochs()

    def set_stage(self, stage_desc: StageDescriptor):
        for elasticity_dim in self._enabled_elasticity_dims:
            if elasticity_dim in stage_desc.train_dims:
                self.multi_elasticity_handler.activate_elasticity(elasticity_dim)
            else:
                self.multi_elasticity_handler.deactivate_elasticity(elasticity_dim)

        width_handler = self.multi_elasticity_handler.width_handler
        depth_handler = self.multi_elasticity_handler.depth_handler
        if width_handler is not None:
            if stage_desc.reorg_weights:
                width_handler.reorganize_weights()
            width_indicator = stage_desc.width_indicator
            if width_indicator:
                width_handler.width_num_params_indicator = width_indicator

        depth_indicator = stage_desc.depth_indicator
        if depth_handler and depth_indicator:
            depth_handler.depth_indicator = depth_indicator

        if stage_desc.bn_adapt:
            self._run_batchnorm_adaptation(self._target_model)

    def statistics(self, quickly_collected_only: bool = False) -> NNCFStatistics:
        return NNCFStatistics()

    def compression_stage(self) -> CompressionStage:
        if self._scheduler.is_final_stage():
            return CompressionStage.FULLY_COMPRESSED
        return CompressionStage.PARTIALLY_COMPRESSED

    def load_state(self, state: Dict[str, Dict[str, Any]]) -> None:
        """
        Loads the compression controller state from the map of algorithm name to the dictionary with state attributes.

        :param state: map of the algorithm name to the dictionary with the corresponding state attributes.
        """
        super().load_state(state)
        elasticity_ctrl_state = state[self._ps_state_names.ELASTICITY_CONTROLLER_STATE]
        self._elasticity_ctrl.load_state(elasticity_ctrl_state)

    def get_state(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns compression controller state, which is the map of the algorithm name to the dictionary with the
        corresponding state attributes.

        :return: The compression controller state.
        """
        state = super().get_state()
        state[self._ps_state_names.ELASTICITY_CONTROLLER_STATE] = self._elasticity_ctrl.get_state()
        return state

    def _run_batchnorm_adaptation(self, model):
        if self._bn_adaptation is None:
            raise RuntimeError("Missing initialization of Batchnorm Adaptation algorithm")
        self._bn_adaptation.run(model)
