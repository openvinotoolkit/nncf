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
from typing import Any, Dict

from nncf.api.compression import CompressionLoss
from nncf.api.compression import CompressionScheduler
from nncf.api.compression import CompressionStage
from nncf.common.schedulers import BaseCompressionScheduler
from nncf.common.statistics import NNCFStatistics
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.multi_elasticity_handler import MultiElasticityHandler
from nncf.torch.algo_selector import ZeroCompressionLoss
from nncf.torch.compression_method_api import PTCompressionAlgorithmController
from nncf.torch.nncf_network import NNCFNetwork


class EControllerStateNames:
    MULTI_ELASTICITY_HANDLER_STATE = "multi_elasticity_handler_state"


class ElasticityController(PTCompressionAlgorithmController):
    """
    Serves as a handle to the additional modules, parameters and hooks inserted
    into the original uncompressed model in order to control elasticity in the model.
    """

    _ec_state_names = EControllerStateNames

    def __init__(self, target_model: NNCFNetwork, algo_config: Dict, multi_elasticity_handler: MultiElasticityHandler):
        super().__init__(target_model)
        self.target_model = target_model
        self._algo_config = algo_config
        self._loss = ZeroCompressionLoss(next(target_model.parameters()).device)
        self._scheduler = BaseCompressionScheduler()

        self.multi_elasticity_handler = multi_elasticity_handler
        # Handlers deactivated at init
        self.multi_elasticity_handler.disable_all()

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
        return CompressionStage.UNCOMPRESSED

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Loads the compression controller state from the map of algorithm name to the dictionary with state attributes.

        :param state: map of the algorithm name to the dictionary with the corresponding state attributes.
        """
        super().load_state(state)
        self.multi_elasticity_handler.load_state(state[self._ec_state_names.MULTI_ELASTICITY_HANDLER_STATE])

    def get_state(self) -> Dict[str, Any]:
        """
        Returns compression controller state, which is the map of the algorithm name to the dictionary with the
        corresponding state attributes.

        :return: The compression controller state.
        """
        state = super().get_state()
        state[self._ec_state_names.MULTI_ELASTICITY_HANDLER_STATE] = self.multi_elasticity_handler.get_state()
        return state
