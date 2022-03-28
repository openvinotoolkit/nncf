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

from nncf.api.compression import CompressionLoss
from nncf.api.compression import CompressionScheduler
from nncf.api.compression import CompressionStage
from nncf.common.initialization.batchnorm_adaptation import BatchnormAdaptationAlgorithm
from nncf.common.statistics import NNCFStatistics
from nncf.common.utils.logger import logger as nncf_logger
from nncf.experimental.torch.nas.bootstrapNAS.training.scheduler import NASSchedulerParams
from nncf.torch.algo_selector import ZeroCompressionLoss
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elasticity_controller import ElasticityController
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elasticity_dim import ElasticityDim
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.multi_elasticity_handler import MultiElasticityHandler
from nncf.experimental.torch.nas.bootstrapNAS.training.base_training import BNASTrainingController
from nncf.experimental.torch.nas.bootstrapNAS.training.scheduler import BootstrapNASScheduler
from nncf.experimental.torch.nas.bootstrapNAS.training.stage_descriptor import StageDescriptor
from nncf.torch.nncf_network import NNCFNetwork


class PSControllerStateNames:
    ELASTICITY_CONTROLLER_STATE = 'elasticity_controller_compression_state'


class ProgressiveShrinkingController(BNASTrainingController):
    """
    Serves as a handle to the additional modules, parameters and hooks inserted
    into the original uncompressed model in order to train a supernet using Progressive Shrinking procedure
    from OFA (https://arxiv.org/abs/1908.09791).
    Hosts entities that are to be used during the training process, such as compression scheduler and
    compression loss.
    """

    _ps_state_names = PSControllerStateNames

    def __init__(self, target_model: NNCFNetwork,
                 elasticity_ctrl: ElasticityController,
                 bn_adaptation: BatchnormAdaptationAlgorithm,
                 progressivity_of_elasticity: List[ElasticityDim],
                 schedule_params: NASSchedulerParams):
        super().__init__(target_model)
        self._elasticity_ctrl = elasticity_ctrl
        self._bn_adaptation = bn_adaptation
        self._progressivity_of_elasticity = progressivity_of_elasticity
        self._target_model = target_model
        self._loss = ZeroCompressionLoss(next(target_model.parameters()).device)
        self._available_elasticity_dims = self.multi_elasticity_handler.get_available_elasticity_dims()
        self._scheduler = BootstrapNASScheduler(self, schedule_params, self._available_elasticity_dims,
                                                self._progressivity_of_elasticity)

    @property
    def multi_elasticity_handler(self) -> MultiElasticityHandler:
        """
        Gets access to multi elasticity handler to perform some actions with supernet or subnets.

        :return: multi elasticity handler
        """
        return self._elasticity_ctrl.multi_elasticity_handler

    @property
    def elasticity_controller(self) -> ElasticityController:
        """
        Gets access to elasticity controller. Usually it's needed for saving its state for further resuming in the
        search part.

        :return: elasticity controller
        """
        return self._elasticity_ctrl

    @property
    def loss(self) -> CompressionLoss:
        """
        :return: The instance of the `CompressionLoss`.
        """
        return self._loss

    @property
    def scheduler(self) -> CompressionScheduler:
        """
        :return: The instance of the `CompressionScheduler`.
        """
        return self._scheduler

    def step(self) -> None:
        """
        Should be called at the beginning of each training step for activation some Subnet(s).
        """
        self.multi_elasticity_handler.activate_random_subnet()
        nncf_logger.debug(
            'Active config: {}'.format(self.multi_elasticity_handler.get_active_config()))

    def prepare_for_validation(self) -> None:
        """
        Performs some action on active subnet or supernet before validation. For instance, it can be the batchnorm
        adaptation to achieve the best accuracy on validation.
        """
        self._run_batchnorm_adaptation(self._target_model)

    def get_total_num_epochs(self) -> int:
        """
        Returns total number of epochs required for the supernet training.

        :return: number of epochs
        """
        return self._scheduler.get_total_training_epochs()

    def set_stage(self, stage_desc: StageDescriptor) -> None:
        """
        Set a new training stage with parameters from a given stage descriptor

        :param stage_desc: describes parameters of the training stage that should be enabled
        """
        for elasticity_dim in self._available_elasticity_dims:
            if elasticity_dim in stage_desc.train_dims:
                self.multi_elasticity_handler.enable_elasticity(elasticity_dim)
            else:
                self.multi_elasticity_handler.disable_elasticity(elasticity_dim)

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
        """
        Returns a `Statistics` class instance that contains compression algorithm statistics.

        :param quickly_collected_only: Enables collection of the statistics that
            don't take too much time to compute. Can be helpful for the case when
            need to keep track of statistics on each training batch/step/iteration.
        :return: A `Statistics` class instance that contains compression algorithm statistics.
        """
        return NNCFStatistics()

    def compression_stage(self) -> CompressionStage:
        """
        Returns the compression stage. Should be used on saving best checkpoints
        to distinguish between uncompressed, partially compressed, and fully
        compressed models.

        :return: The compression stage of the target model.
        """
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
