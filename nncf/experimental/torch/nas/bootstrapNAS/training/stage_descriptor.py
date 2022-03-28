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

from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elasticity_dim import ElasticityDim


class SDescriptorParamNames:
    TRAIN_DIMS = 'train_dims'
    EPOCHS = 'epochs'
    REORG_WEIGHTS = 'reorg_weights'
    WIDTH_INDICATOR = 'width_indicator'
    DEPTH_INDICATOR = 'depth_indicator'
    BN_ADAPT = 'bn_adapt'


class StageDescriptor:
    """
    Describes parameters of the training stage. The stage defines active elastic dimension and its parameters.
    """
    _state_names = SDescriptorParamNames

    def __init__(self, train_dims: List[ElasticityDim],
                 epochs: int = 1,
                 reorg_weights: bool = False,
                 bn_adapt: bool = False,
                 depth_indicator: int = 1,
                 width_indicator: int = 1):
        self.train_dims = train_dims
        self.epochs = epochs
        self.depth_indicator = depth_indicator
        self.width_indicator = width_indicator
        self.reorg_weights = reorg_weights
        self.bn_adapt = bn_adapt

    def __eq__(self, other: 'StageDescriptor'):
        return self.__dict__ == other.__dict__

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'StageDescriptor':
        """
        Creates the object from its config.
        """
        train_dims = config.get(cls._state_names.TRAIN_DIMS, ['kernel'])
        kwargs = {
            cls._state_names.TRAIN_DIMS: [ElasticityDim.from_str(dim) for dim in train_dims],
            cls._state_names.EPOCHS: config.get(cls._state_names.EPOCHS, 1),
            cls._state_names.REORG_WEIGHTS: config.get(cls._state_names.REORG_WEIGHTS, False),
            cls._state_names.WIDTH_INDICATOR: config.get(cls._state_names.WIDTH_INDICATOR, 1),
            cls._state_names.DEPTH_INDICATOR: config.get(cls._state_names.DEPTH_INDICATOR, 1),
            cls._state_names.BN_ADAPT: config.get(cls._state_names.BN_ADAPT, False),
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
        kwargs[cls._state_names.TRAIN_DIMS] = [ElasticityDim.from_str(dim) for dim in train_dims]
        return cls(**kwargs)

    def get_state(self) -> Dict[str, Any]:
        return {
            self._state_names.TRAIN_DIMS: [dim.value for dim in self.train_dims],
            self._state_names.EPOCHS: self.epochs,
            self._state_names.REORG_WEIGHTS: self.reorg_weights,
            self._state_names.WIDTH_INDICATOR: self.width_indicator,
            self._state_names.DEPTH_INDICATOR: self.depth_indicator,
            self._state_names.BN_ADAPT: self.bn_adapt,
        }
