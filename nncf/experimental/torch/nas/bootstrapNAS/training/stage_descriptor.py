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
from typing import Any, Dict, List

from nncf.common.logging import nncf_logger
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elasticity_dim import ElasticityDim

DEFAULT_STAGE_LR_RATE = 3.5e-06


class SDescriptorParamNames:
    TRAIN_DIMS = "train_dims"
    EPOCHS = "epochs"
    REORG_WEIGHTS = "reorg_weights"
    WIDTH_INDICATOR = "width_indicator"
    DEPTH_INDICATOR = "depth_indicator"
    BN_ADAPT = "bn_adapt"
    INIT_LR = "init_lr"
    EPOCHS_LR = "epochs_lr"
    SAMPLE_RATE = "sample_rate"


class StageDescriptor:
    """
    Describes parameters of the training stage. The stage defines active elastic dimension and its parameters.
    """

    _state_names = SDescriptorParamNames

    def __init__(
        self,
        train_dims: List[ElasticityDim],
        epochs: int = 1,
        reorg_weights: bool = False,
        bn_adapt: bool = False,
        depth_indicator: int = 1,
        width_indicator: int = 1,
        init_lr: float = None,
        epochs_lr: int = None,
        sample_rate: int = 1,
    ):
        self.train_dims = train_dims
        self.epochs = epochs
        self.depth_indicator = depth_indicator
        self.width_indicator = width_indicator
        self.reorg_weights = reorg_weights
        self.bn_adapt = bn_adapt
        self.init_lr = init_lr
        self.epochs_lr = epochs_lr
        self.sample_rate = sample_rate
        if sample_rate <= 0:
            nncf_logger.warning(f"Only positive integers are allowed for sample rate, but sample_rate={sample_rate}.")
            nncf_logger.warning("Setting sample rate to default 1")
            self.sample_rate = 1

    def __eq__(self, other: "StageDescriptor"):
        return self.__dict__ == other.__dict__

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "StageDescriptor":
        """
        Creates the object from its config.
        """
        train_dims = config.get(cls._state_names.TRAIN_DIMS, ["kernel"])
        kwargs = {
            cls._state_names.TRAIN_DIMS: [ElasticityDim(dim) for dim in train_dims],
            cls._state_names.EPOCHS: config.get(cls._state_names.EPOCHS, 1),
            cls._state_names.REORG_WEIGHTS: config.get(cls._state_names.REORG_WEIGHTS, False),
            cls._state_names.WIDTH_INDICATOR: config.get(cls._state_names.WIDTH_INDICATOR, 1),
            cls._state_names.DEPTH_INDICATOR: config.get(cls._state_names.DEPTH_INDICATOR, 1),
            cls._state_names.BN_ADAPT: config.get(cls._state_names.BN_ADAPT, False),
            cls._state_names.INIT_LR: config.get(cls._state_names.INIT_LR, None),
            cls._state_names.EPOCHS_LR: config.get(cls._state_names.EPOCHS_LR, None),
            cls._state_names.SAMPLE_RATE: config.get(cls._state_names.SAMPLE_RATE, 1),
        }
        return cls(**kwargs)

    @classmethod
    def from_state(cls, state: Dict[str, Any]):
        """
        Creates the object from its state.

        :param state: Output of `get_state()` method.
        """
        kwargs = state.copy()
        train_dims = state[cls._state_names.TRAIN_DIMS]
        kwargs[cls._state_names.TRAIN_DIMS] = [ElasticityDim(dim) for dim in train_dims]
        return cls(**kwargs)

    def get_state(self) -> Dict[str, Any]:
        state_dict = {
            self._state_names.TRAIN_DIMS: [dim.value for dim in self.train_dims],
            self._state_names.EPOCHS: self.epochs,
            self._state_names.REORG_WEIGHTS: self.reorg_weights,
            self._state_names.WIDTH_INDICATOR: self.width_indicator,
            self._state_names.DEPTH_INDICATOR: self.depth_indicator,
            self._state_names.BN_ADAPT: self.bn_adapt,
            self._state_names.SAMPLE_RATE: self.sample_rate,
        }
        if self.init_lr is not None:
            state_dict["init_lr"] = self.init_lr
        if self.epochs_lr is not None:
            state_dict["epochs_lr"] = self.epochs_lr
        return state_dict
