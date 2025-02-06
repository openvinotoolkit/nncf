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
from typing import Any, Callable, Dict, List, NoReturn

from nncf.api.compression import CompressionLoss
from nncf.api.compression import CompressionScheduler
from nncf.api.compression import CompressionStage
from nncf.common.initialization.batchnorm_adaptation import BatchnormAdaptationAlgorithm
from nncf.common.logging import nncf_logger
from nncf.common.statistics import NNCFStatistics
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elasticity_controller import ElasticityController
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elasticity_dim import ElasticityDim
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.multi_elasticity_handler import MultiElasticityHandler
from nncf.experimental.torch.nas.bootstrapNAS.training.base_training import BNASTrainingController
from nncf.experimental.torch.nas.bootstrapNAS.training.lr_scheduler import GlobalLRScheduler
from nncf.experimental.torch.nas.bootstrapNAS.training.lr_scheduler import StageLRScheduler
from nncf.experimental.torch.nas.bootstrapNAS.training.scheduler import BootstrapNASScheduler
from nncf.experimental.torch.nas.bootstrapNAS.training.scheduler import NASSchedulerParams
from nncf.experimental.torch.nas.bootstrapNAS.training.stage_descriptor import StageDescriptor
from nncf.torch.nncf_network import NNCFNetwork


class PSControllerStateNames:
    ELASTICITY_CONTROLLER_STATE = "elasticity_controller_compression_state"
    LR_GLOBAL_SCHEDULE_STATE = "learning_rate_global_schedule_state"


class ProgressiveShrinkingController(BNASTrainingController):
    """
    Serves as a handle to the additional modules, parameters and hooks inserted
    into the original uncompressed model in order to train a supernet using Progressive Shrinking procedure
    from OFA (https://arxiv.org/abs/1908.09791).
    Hosts entities that are to be used during the training process, such as compression scheduler and
    compression loss.
    """

    _ps_state_names = PSControllerStateNames

    def __init__(
        self,
        target_model: NNCFNetwork,
        elasticity_ctrl: ElasticityController,
        bn_adaptation: BatchnormAdaptationAlgorithm,
        progressivity_of_elasticity: List[ElasticityDim],
        schedule_params: NASSchedulerParams,
        lr_schedule_config: Dict[str, Any],
        compression_loss_func: Callable,
    ):
        super().__init__(target_model)
        self._elasticity_ctrl = elasticity_ctrl
        self._bn_adaptation = bn_adaptation
        self._progressivity_of_elasticity = progressivity_of_elasticity
        self._target_model = target_model
        self._loss = compression_loss_func
        self._available_elasticity_dims = self.multi_elasticity_handler.get_available_elasticity_dims()
        self._lr_schedule_config = lr_schedule_config
        self._scheduler = BootstrapNASScheduler(
            self, schedule_params, self._available_elasticity_dims, self._progressivity_of_elasticity
        )
        self._sample_rate = 1

    def set_training_lr_scheduler_args(self, optimizer, train_iters):
        params = self._lr_schedule_config.get("params", {})
        num_epochs = params.get("num_epochs", None)
        base_lr = params.get("base_lr", None)

        if base_lr is not None:
            nncf_logger.info("Global LR scheduler in use")
            # Global lr scheduler
            if num_epochs is None:
                params["num_epochs"] = self.get_total_num_epochs()
            lr_scheduler = GlobalLRScheduler(optimizer, train_iters, **params)
        else:
            nncf_logger.info("Stage LR scheduler in use")
            lr_scheduler = StageLRScheduler(optimizer, train_iters)
        self._scheduler.lr_scheduler = lr_scheduler

    @property
    def lr_schedule_config(self) -> str:
        """
        Gets access to learning rate scheduler configuration.

        :return: learning rate scheduler
        """
        return self._lr_schedule_config

    @lr_schedule_config.setter
    def lr_schedule_config(self, val: Dict[str, Any]) -> NoReturn:
        self._lr_schedule_config = val

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
        if self._scheduler.current_step % self._sample_rate == 0:
            self.multi_elasticity_handler.activate_random_subnet()
            nncf_logger.debug(f"Active config: {self.multi_elasticity_handler.get_active_config()}")

    def prepare_for_validation(self) -> None:
        """
        Performs some action on active subnet or supernet before validation. For instance, it can be the batchnorm
        adaptation to achieve the best accuracy on validation.
        """
        if self._bn_adaptation:
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

        self._sample_rate = stage_desc.sample_rate

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
        self._lr_schedule_config = state[self._ps_state_names.LR_GLOBAL_SCHEDULE_STATE]
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
        state[self._ps_state_names.LR_GLOBAL_SCHEDULE_STATE] = self._lr_schedule_config
        return state

    def _run_batchnorm_adaptation(self, model):
        if self._bn_adaptation is None:
            nncf_logger.warning("Batchnorm adaptation requested but it hasn't been enabled for training.")
        self._bn_adaptation.run(model)
