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
from typing import Any, Dict, List, Optional, Tuple

from nncf.common.logging import nncf_logger
from nncf.common.schedulers import BaseCompressionScheduler
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elasticity_dim import ElasticityDim
from nncf.experimental.torch.nas.bootstrapNAS.training.base_training import BNASTrainingAlgorithm
from nncf.experimental.torch.nas.bootstrapNAS.training.lr_scheduler import BaseLRScheduler
from nncf.experimental.torch.nas.bootstrapNAS.training.stage_descriptor import DEFAULT_STAGE_LR_RATE
from nncf.experimental.torch.nas.bootstrapNAS.training.stage_descriptor import StageDescriptor


class NSParamsStateNames:
    LIST_STAGE_DESCRIPTIONS = "list_stage_descriptions"


class NASSchedulerParams:
    _state_names = NSParamsStateNames

    def __init__(self, list_stage_descriptions: Optional[List[StageDescriptor]] = None):
        """
        Constructor

        :param list_stage_descriptions: List of parameters per each supernet training stage.
        """
        if list_stage_descriptions is None:
            list_stage_descriptions = [
                StageDescriptor(train_dims=[ElasticityDim.KERNEL], epochs=1),
                StageDescriptor(train_dims=[ElasticityDim.KERNEL, ElasticityDim.DEPTH], epochs=1),
                StageDescriptor(train_dims=[ElasticityDim.KERNEL, ElasticityDim.DEPTH], epochs=1, depth_indicator=2),
                StageDescriptor(train_dims=[ElasticityDim.KERNEL, ElasticityDim.DEPTH, ElasticityDim.WIDTH], epochs=1),
                StageDescriptor(
                    train_dims=[ElasticityDim.KERNEL, ElasticityDim.DEPTH, ElasticityDim.WIDTH],
                    epochs=1,
                    width_indicator=2,
                    reorg_weights=True,
                    bn_adapt=True,
                ),
            ]
        self.list_stage_descriptions = list_stage_descriptions

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "NASSchedulerParams":
        """
        Creates the object from its config.
        """
        descs_config = config.get(cls._state_names.LIST_STAGE_DESCRIPTIONS)
        descs = None
        if descs_config is not None:
            descs = [StageDescriptor.from_config(stage_desc_config) for stage_desc_config in descs_config]
        return cls(descs)

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "NASSchedulerParams":
        """
        Creates the object from its state.

        :param state: Output of `get_state()` method.
        """
        list_stage_descriptions_state = state[cls._state_names.LIST_STAGE_DESCRIPTIONS]
        list_stage_descriptions = [StageDescriptor.from_state(state) for state in list_stage_descriptions_state]
        return cls(list_stage_descriptions)

    def get_state(self) -> Dict[str, Any]:
        """
        Returns the compression loss state.

        :return: The compression loss state.
        """
        return {
            self._state_names.LIST_STAGE_DESCRIPTIONS: [desc.get_state() for desc in self.list_stage_descriptions],
        }

    def __eq__(self, other: "NASSchedulerParams") -> bool:
        return self.__dict__ == other.__dict__


class BNASSchedulerStateNames:
    LIST_STAGE_DESCRIPTIONS = "list_stage_descriptions"


class BootstrapNASScheduler(BaseCompressionScheduler):
    """
    The cornerstone of supernet training within a NAS algorithm. The `step()` and `epoch_step()` methods of the
    compression scheduler must be called in the beginning of each training step and epoch, respectively.
    These methods trigger a subnet activations, elasticity configuration during the training.
    """

    _state_names = BNASSchedulerStateNames

    def __init__(
        self,
        training_ctrl: BNASTrainingAlgorithm,
        params: NASSchedulerParams,
        available_elasticity_dims: List[ElasticityDim],
        progressivity_of_elasticity: List[ElasticityDim],
    ):
        super().__init__()
        self._training_ctrl = training_ctrl
        self._params = params
        self._available_elasticity_dims = available_elasticity_dims
        self._progressivity_of_elasticity = progressivity_of_elasticity
        self.current_stage_idx = -1
        # Property setter with validation is not used intentionally for the resume case. When the actual list stage
        #  descriptors are loaded after creation of the scheduler. Scheduler is resumed without config = with empty
        #  params = default stage descriptors, that could lead to inconsistency with progressivity and enabled dims.
        #  The validation will happen in the first usage of list_stage_descriptors property.
        self._list_stage_descriptors = self._params.list_stage_descriptions
        self._is_elasticity_dims_validated = False
        self._lr_scheduler = None

    @property
    def lr_scheduler(self):
        return self._lr_scheduler

    @lr_scheduler.setter
    def lr_scheduler(self, lr_scheduler: BaseLRScheduler) -> None:
        self._lr_scheduler = lr_scheduler

    @property
    def current_step(self) -> int:
        return self._lr_scheduler.current_step

    @property
    def list_stage_descriptors(self) -> List[StageDescriptor]:
        """
        :return: a list of stage descriptors (parameters of the training stage).
        """
        if not self._is_elasticity_dims_validated:
            self._validate_elasticity_dims(self._available_elasticity_dims, self._progressivity_of_elasticity)
        self._is_elasticity_dims_validated = True
        self._validate_lr()
        return self._list_stage_descriptors

    @list_stage_descriptors.setter
    def list_stage_descriptors(self, stage_descriptors: List[StageDescriptor]) -> None:
        """
        Sets a given stage descriptors to the schedule. Can be used on loading state from a checkpoint.

        :param stage_descriptors: list of stage descriptors
        """
        self._list_stage_descriptors = stage_descriptors
        self._validate_elasticity_dims(self._available_elasticity_dims, self._progressivity_of_elasticity)
        self._is_elasticity_dims_validated = True
        self._validate_lr()

    def step(self, next_step: Optional[int] = None) -> None:
        """
        Should be called at the beginning of each training step to prepare
        the compression method to continue training the model in the `next_step`.

        :param next_step: The global step index for which the compression scheduler
            will update the state of the compression method.
        """
        self._training_ctrl.step()
        self._lr_scheduler.step(next_step)

    def epoch_step(self, next_epoch: Optional[int] = None) -> None:
        """
        Should be called at the beginning of each training epoch to prepare
        the compression method to continue training the model in the `next_epoch`.

        :param next_epoch: The epoch index for which the compression scheduler
            will update the state of the compression method.
        """
        super().epoch_step(next_epoch)
        self._lr_scheduler.epoch_step(next_epoch)
        stage_desc, stage_desc_idx = self.get_current_stage_desc()
        if stage_desc is not None and stage_desc_idx != self.current_stage_idx:
            self._lr_scheduler.stage_step(stage_desc)
            self._training_ctrl.set_stage(stage_desc)
            self.current_stage_idx = stage_desc_idx

    def is_final_stage(self) -> bool:
        """
        :return: True, if final stage has been reached, False - otherwise
        """
        return self.current_stage_idx == len(self.list_stage_descriptors) - 1

    def get_current_stage_desc(self) -> Tuple[Optional[StageDescriptor], int]:
        """
        :return: current stage descriptor and its index in the list of all descriptors
        """
        partial_epochs = 0
        for stage_desc_idx, stage_desc in enumerate(self.list_stage_descriptors):
            partial_epochs += stage_desc.epochs
            if self.current_epoch < partial_epochs:
                return stage_desc, stage_desc_idx
        return None, -1

    def get_total_training_epochs(self) -> int:
        """
        Returns total number of epochs required for the supernet training.

        :return: number of epochs
        """
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
        # No conflict resolving with the related config options, parameters are overridden by compression state
        self.list_stage_descriptors = list(map(StageDescriptor.from_state, list_stage_descriptors))

    def get_state(self) -> Dict[str, Any]:
        """
        Returns the compression scheduler state.

        :return: The compression scheduler state.
        """
        state = super().get_state()
        state[self._state_names.LIST_STAGE_DESCRIPTIONS] = [desc.get_state() for desc in self.list_stage_descriptors]
        return state

    def _validate_elasticity_dims(
        self, available_elasticity_dims: List[ElasticityDim], progressivity_of_elasticity: List[ElasticityDim]
    ) -> None:
        last_stage = -1
        first_stage = len(progressivity_of_elasticity)
        for desc in self._list_stage_descriptors:
            high_priority_dim_idx = -1
            low_priority_dim_idx = len(progressivity_of_elasticity)
            stages_covered = []
            for train_dim in desc.train_dims:
                if train_dim not in available_elasticity_dims:
                    raise ValueError(
                        f"Invalid training elasticity dimension {train_dim} in the scheduler.\n"
                        f"The elasticity for this dimension is not enabled.\n"
                        f"It can be enabled by specifying `available_elasticity_dims` param in the `elasticity` "
                        f"section of config.\n"
                        f"List of currently available dimensions: {[dim.value for dim in available_elasticity_dims]}"
                    )
                dim_idx = progressivity_of_elasticity.index(train_dim)
                if dim_idx not in stages_covered:
                    stages_covered.append(dim_idx)
                if dim_idx > high_priority_dim_idx:
                    high_priority_dim_idx = dim_idx
                if dim_idx < low_priority_dim_idx:
                    low_priority_dim_idx = dim_idx
            if high_priority_dim_idx < last_stage or low_priority_dim_idx > first_stage:
                raise ValueError(
                    f"stage {progressivity_of_elasticity[high_priority_dim_idx]} violates progressivity of elasticity"
                )
            for i in range(low_priority_dim_idx, high_priority_dim_idx):
                if i not in stages_covered and progressivity_of_elasticity[i] in available_elasticity_dims:
                    raise ValueError(
                        f"Missed to call {progressivity_of_elasticity[i]} in {desc.train_dims} which violates "
                        f"progressivity of elasticity {progressivity_of_elasticity}"
                    )
            last_stage = high_priority_dim_idx
            first_stage = low_priority_dim_idx

    def _validate_lr(self):
        for desc in self._list_stage_descriptors:
            # Check if global learning rate has been set
            if desc.init_lr is not None and bool(self._training_ctrl.lr_schedule_config):
                raise ValueError(
                    f"Global learning rate scheduler is in use. Cannot set stage learning rate: {desc.init_lr}"
                )
            # Check if stage learning rate has been set
            if desc.init_lr is None and not bool(self._training_ctrl.lr_schedule_config):
                nncf_logger.warning(
                    "Stage learning rate in use but init_lr value for stage wasn't set. Using default value of 3.5e-6"
                )
                desc.init_lr = DEFAULT_STAGE_LR_RATE

            if desc.init_lr is not None and desc.epochs_lr is None:
                nncf_logger.warning(
                    f"Stage learning rate in use but epochs_lr value for stage wasn't set. "
                    f"Using number of epochs for stage {desc.epochs}"
                )
                desc.epochs_lr = desc.epochs
