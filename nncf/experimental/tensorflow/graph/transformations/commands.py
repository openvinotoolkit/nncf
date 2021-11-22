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

from typing import Dict
from typing import Any
from typing import List
from typing import Optional

from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationCommand
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.common.graph.transformations.commands import TransformationType
from nncf.experimental.tensorflow.nncf_operation import NNCFOperation


class TFTargetPointStateNames:
    OP_NAME = 'op_name'
    OP_TYPE_NAME = 'op_type_name'
    PORT_ID = 'port_id'
    TARGET_TYPE = 'target_type'


class TFTargetPoint(TargetPoint):
    """
    Describes where the compression operation should be placed.
    """

    _state_names = TFTargetPointStateNames

    def __init__(self,
                 op_name: str,
                 op_type_name: str,
                 port_id: int,
                 target_type: TargetType):
        """
        Initializes target point for TensorFlow backend.

        :param op_name: Name of a node in the `FuncGraph`.
        :param op_type_name: Type of operation.
        :param port_id: Port id.
        :param target_type: Type of the target point.
        """
        super().__init__(target_type)
        self.op_name = op_name
        self.op_type_name = op_type_name
        self.port_id = port_id

    def __eq__(self, other: 'TFTargetPoint') -> bool:
        return isinstance(other, TFTargetPoint) and \
               self.type == other.type and \
               self.op_name == other.op_name and \
               self.op_type_name == other.op_type_name and \
               self.port_id == other.port_id

    def __str__(self) -> str:
        items = [
            super().__str__(),
            self.op_name,
            self.op_type_name,
            str(self.port_id),
        ]
        return ' '.join(items)

    def get_state(self) -> Dict[str, Any]:
        """
        Returns a dictionary with Python data structures (dict, list, tuple, str, int, float, True, False, None) that
        represents state of the object.

        :return: State of the object.
        """
        state = {
            self._state_names.OP_NAME: self.op_name,
            self._state_names.OP_TYPE_NAME: self.op_type_name,
            self._state_names.PORT_ID: self.port_id,
            self._state_names.TARGET_TYPE: self.type.get_state(),
        }
        return state

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> 'TFTargetPoint':
        """
        Creates the object from its state.

        :param state: Output of `get_state()` method.
        """
        kwargs = {
            cls._state_names.OP_NAME: state[cls._state_names.OP_NAME],
            cls._state_names.OP_TYPE_NAME: state[cls._state_names.OP_TYPE_NAME],
            cls._state_names.PORT_ID: state[cls._state_names.PORT_ID],
            cls._state_names.TARGET_TYPE: TargetType.from_state(state[cls._state_names.TARGET_TYPE]),
        }
        return cls(**kwargs)


class TFInsertionCommand(TransformationCommand):
    """
    """

    def __init__(self,
                 target_point: TFTargetPoint,
                 callable_object: Optional[NNCFOperation] = None,
                 priority: Optional[TransformationPriority] = None):
        """
        :param target_point:
        :param callable_object:
        :param priority:
        """
        super().__init__(TransformationType.INSERT, target_point)
        self.callable_objects = []
        if callable_object is not None:
            _priority = TransformationPriority.DEFAULT_PRIORITY if priority is None else priority
            self.callable_objects.append((callable_object, _priority))

    @property
    def insertion_objects(self) -> List[NNCFOperation]:
        """
        :return:
        """
        return [op for op, _ in self.callable_objects]

    def union(self, other: TransformationCommand) -> 'TFInsertionCommand':
        """
        :param other:
        :return:
        """
        if not self.check_command_compatibility(other):
            raise ValueError(
                '{} and {} commands could not be united'.format(type(self).__name__, type(other).__name__)
            )

        command = TFInsertionCommand(self.target_point)
        # TODO(andrey-churkin): Check order
        command.callable_objects = sorted([*self.callable_objects, *other.callable_objects], key=lambda x: x[1])
        return command
