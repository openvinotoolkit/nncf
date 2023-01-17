"""
 Copyright (c) 2023 Intel Corporation
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

from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationCommand
from nncf.common.graph.transformations.commands import TransformationType


class OVTargetPoint(TargetPoint):
    def __init__(self, target_type: TargetType, target_node_name: str, port_id: int):
        super().__init__(target_type)
        self.target_node_name = target_node_name
        self.port_id = port_id

    def __eq__(self, other: 'OVTargetPoint') -> bool:
        return isinstance(other, OVTargetPoint) and \
               self.type == other.type and self.target_node_name == other.target_node_name and \
               self.port_id == other.port_id

    def __hash__(self) -> int:
        return hash((self.target_node_name, self.port_id, self._target_type))

    def __lt__(self, other: 'OVTargetPoint') -> bool:
        # The OVTargetPoint should have the way to compare.
        # NNCF has to be able returning the Quantization Target Points in the deterministic way.
        # MinMaxQuantizationAlgorithm returns the sorted Set of such OVTargetPoints.
        params = ['_target_type', 'target_node_name', 'port_id']
        for param in params:
            if self.__getattribute__(param) < other.__getattribute__(param):
                return True
            if self.__getattribute__(param) > other.__getattribute__(param):
                return False
        return False


class OVInsertionCommand(TransformationCommand):
    def __init__(self, target_point: OVTargetPoint):
        super().__init__(TransformationType.INSERT, target_point)

    def union(self, other: 'TransformationCommand') -> 'TransformationCommand':
        # Have a look at nncf/torch/graph/transformations/commands/PTInsertionCommand
        raise NotImplementedError()


class OVOutputInsertionCommand(OVInsertionCommand):
    def union(self, other: 'TransformationCommand') -> 'TransformationCommand':
        # Have a look at nncf/torch/graph/transformations/commands/PTInsertionCommand
        raise NotImplementedError()


class OVFQNodeRemovingCommand(TransformationCommand):
    """
    Removes FakeQuantize nodes from the model.
    """

    def __init__(self, target_point: OVTargetPoint):
        """
        :param target_point: The TargetPoint instance for the layer that contains information for removing.
        """
        super().__init__(TransformationType.REMOVE, target_point)

    def union(self, other: 'TransformationCommand') -> 'TransformationCommand':
        # Have a look at nncf/torch/graph/transformations/commands/PTInsertionCommand
        raise NotImplementedError()
