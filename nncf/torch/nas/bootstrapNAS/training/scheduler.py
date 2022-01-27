"""
 Copyright (c) 2021 Intel Corporation
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
from typing import Optional
from typing import Tuple

from nncf.common.schedulers import BaseCompressionScheduler
from nncf.torch.nas.bootstrapNAS.elasticity.elasticity_dim import ElasticityDim
from nncf.torch.nas.bootstrapNAS.training.stage_descriptor import StageDescriptor


class BNASSchedulerStateNames:
    LIST_STAGE_DESCRIPTIONS = 'list_stage_descriptions'


class BootstrapNASScheduler(BaseCompressionScheduler):
    _state_names = BNASSchedulerStateNames

    def __init__(self, training_ctrl: 'ProgressiveShrinkingController',
                 params: Dict[str, List[Dict]],
                 enabled_elasticity_dims: List[ElasticityDim],
                 progressivity_of_elasticity: List[ElasticityDim]):
        super().__init__()
        self._training_ctrl = training_ctrl
        self._params = params if params else self._get_default_params()
        self._enabled_elasticity_dims = enabled_elasticity_dims
        self._progressivity_of_elasticity = progressivity_of_elasticity

        list_stage_descriptions = self._params.get('list_stage_descriptions', [])
        self.current_stage_idx = -1
        # Property setter with validation is not used intentionally for the resume case. When the actual list stage
        #  descriptors are loaded after creation of the scheduler. Scheduler is resumed without config = with empty
        #  params = default stage descriptors, that could lead to inconsistency with progressivity and enabled dims.
        #  The validation will happen in the first usage of list_stage_descriptors property.
        self._list_stage_descriptors = [StageDescriptor.from_state(d) for d in list_stage_descriptions]
        self._is_elasticity_dims_validated = False

    @property
    def list_stage_descriptors(self) -> List[StageDescriptor]:
        if not self._is_elasticity_dims_validated:
            self._validate_elasticity_dims(self._enabled_elasticity_dims, self._progressivity_of_elasticity)
        self._is_elasticity_dims_validated = True
        return self._list_stage_descriptors

    @list_stage_descriptors.setter
    def list_stage_descriptors(self, stage_descriptors: List[StageDescriptor]):
        self._list_stage_descriptors = stage_descriptors
        self._validate_elasticity_dims(self._enabled_elasticity_dims, self._progressivity_of_elasticity)
        self._is_elasticity_dims_validated = True

    def is_final_stage(self) -> bool:
        return self.current_stage_idx == len(self.list_stage_descriptors) - 1

    def step(self, next_step: Optional[int] = None) -> None:
        self._training_ctrl.step()

    def epoch_step(self, next_epoch: Optional[int] = None) -> None:
        super().epoch_step(next_epoch)
        stage_desc, stage_desc_idx = self.get_train_dims_for_epoch()
        if stage_desc is not None:
            if stage_desc_idx != self.current_stage_idx:
                self._training_ctrl.set_stage(stage_desc)
                self.current_stage_idx = stage_desc_idx

    def get_train_dims_for_epoch(self) -> Tuple[Optional[StageDescriptor], int]:
        partial_epochs = 0
        stage_desc_idx = 0
        for stage_desc in self.list_stage_descriptors:
            partial_epochs += stage_desc.epochs
            if self.current_epoch < partial_epochs:
                return stage_desc, stage_desc_idx
            stage_desc_idx += 1
        return None, -1

    def get_total_training_epochs(self):
        total_epochs = 0
        for stage_desc in self.list_stage_descriptors:
            total_epochs += stage_desc.epochs
        return total_epochs

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Loads the compression scheduler state, but does not update the state of the
        compression method.

        :param state: Output of `get_state()` method.
        """
        super().load_state(state)
        list_stage_descriptors = state[self._state_names.LIST_STAGE_DESCRIPTIONS]
        # TODO(nlyalyus): no conflict resolving between value in state and in config. It's always overridden by state
        self.list_stage_descriptors = list(map(lambda x: StageDescriptor.from_state(x), list_stage_descriptors))

    def get_state(self) -> Dict[str, Any]:
        """
        Returns the compression scheduler state.

        :return: The compression scheduler state.
        """
        state = super().get_state()
        state[self._state_names.LIST_STAGE_DESCRIPTIONS] = [desc.get_state() for desc in self.list_stage_descriptors]
        return state

    def _validate_elasticity_dims(self, enabled_elasticity_dims, progressivity_of_elasticity):
        last_stage = -1
        for desc in self._list_stage_descriptors:
            high_priority_dim_idx = -1
            stages_covered = []
            for train_dim in desc.train_dims:
                if train_dim not in enabled_elasticity_dims:
                    raise ValueError(
                        f"Invalid training elasticity dimension {train_dim} in the scheduler.\n"
                        f"The elasticity for this dimension is not enabled.\n"
                        f"It can be enabled by specifying `enabled_elasticity_dims` param in the `elasticity` "
                        f"section of config.\n"
                        f"List of currently enabled dimensions: {[dim.value for dim in enabled_elasticity_dims]}")
                dim_idx = progressivity_of_elasticity.index(train_dim)
                if dim_idx not in stages_covered:
                    stages_covered.append(dim_idx)
                if dim_idx > high_priority_dim_idx:
                    high_priority_dim_idx = dim_idx
            if high_priority_dim_idx < last_stage:
                raise ValueError(
                    f"stage {progressivity_of_elasticity[high_priority_dim_idx]} violates progressivity of elasticity")
            for i in range(0, high_priority_dim_idx):
                if i not in stages_covered and progressivity_of_elasticity[i] in enabled_elasticity_dims:
                    raise ValueError(
                        f"Missed to call {progressivity_of_elasticity[i]} in {desc.train_dims} which violates progressivity of elasticity {progressivity_of_elasticity}")
            last_stage = high_priority_dim_idx

    @staticmethod
    def _get_default_params() -> Dict[str, List[Dict]]:
        # TODO(nlyalyus): Perform some studies to determine default params (ticket 76938)
        return {
            "list_stage_descriptions": [
                {"train_dims": ["kernel"], "epochs": 1},
                {"train_dims": ["kernel", "depth"], "epochs": 1},
                {"train_dims": ["kernel", "depth"], "epochs": 1},
                {"train_dims": ["kernel", "depth", "width"], "epochs": 1},
                {"train_dims": ["kernel", "depth", "width"], "epochs": 1, "reorg_weights": True, "bn_adapt": True}
            ]
        }
