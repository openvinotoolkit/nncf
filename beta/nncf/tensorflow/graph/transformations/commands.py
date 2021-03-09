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

from typing import Any, Callable, List, Optional

from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationCommand
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.common.graph.transformations.commands import TransformationType


class TFLayerPoint(TargetPoint):
    """
    `TFLayerPoint` defines an object or spot relative to the layer in the
    TensorFlow model graph. It can be the layer itself, layer weights, specific
    spots in the model graph, for example, insertion spots before/after layer
    and etc.
    """

    def __init__(self, target_type: TargetType, layer_name: str):
        super().__init__(target_type)
        self._layer_name = layer_name

    @property
    def layer_name(self) -> str:
        return self._layer_name

    def __eq__(self, other: Any) -> bool:
        if self.__class__ is other.__class__:
            return self.type == other.type and self.layer_name == other.layer_name
        return False

    def __str__(self) -> str:
        return super().__str__() + ' ' + self.layer_name


class TFLayer(TFLayerPoint):
    """
    `TFLayer` defines a layer in the TensorFlow model graph.

    For example, `TFLayer` is used to specify the layer in the removal command
    to remove from the model.
    """

    def __init__(self, layer_name: str):
        super().__init__(TargetType.LAYER, layer_name)


class TFBeforeLayer(TFLayerPoint):
    """
    `TFBeforeLayer` defines a spot before the layer in the TensorFlow model graph.

    For example, `TFBeforeLayer` is used in the insertion commands to specify
    where the new object should be inserted.
    """

    def __init__(self, layer_name: str, instance_index: int = 0, in_port: int = 0):
        super().__init__(TargetType.BEFORE_LAYER, layer_name)
        self._instance_index = instance_index
        self._in_port = in_port

    @property
    def instance_index(self) -> int:
        return self._instance_index

    @property
    def in_port(self) -> int:
        return self._in_port

    def __eq__(self, other: Any) -> bool:
        if self.__class__ is other.__class__:
            return self.type == other.type \
                   and self.layer_name == other.layer_name \
                   and self.instance_index == other.instance_index \
                   and self.in_port == other.in_port
        return False

    def __str__(self) -> str:
        return ' '.join([super().__str__(),
                         self.instance_index,
                         self.in_port])


class TFAfterLayer(TFLayerPoint):
    """
    `TFAfterLayer` defines a spot after the layer in the TensorFlow model graph.

    For example, `TFAfterLayer` is used in the insertion commands to specify
    where the new object should be inserted.
    """

    def __init__(self, layer_name: str, instance_index: int = 0, out_port: int = 0):
        super().__init__(TargetType.AFTER_LAYER, layer_name)
        self._instance_index = instance_index
        self._out_port = out_port

    @property
    def instance_index(self) -> int:
        return self._instance_index

    @property
    def out_port(self) -> int:
        return self._out_port

    def __eq__(self, other: Any) -> bool:
        if self.__class__ is other.__class__:
            return self.type == other.type \
                   and self.layer_name == other.layer_name \
                   and self.instance_index == other.instance_index \
                   and self._out_port == other.out_port
        return False

    def __str__(self) -> str:
        return ' '.join([super().__str__(),
                         self.instance_index,
                         self.out_port])


class TFLayerWeight(TFLayerPoint):
    """
    `TFLayerWeight` defines the layer weights.

    For example, `TFLayerWeight` is used in the insertion command to specify
    the layer weights for which an operation with weights should be inserted.
    """

    def __init__(self, layer_name: str, weights_attr_name: str):
        super().__init__(TargetType.OPERATION_WITH_WEIGHTS, layer_name)
        self._weights_attr_name = weights_attr_name

    @property
    def weights_attr_name(self) -> str:
        return self._weights_attr_name

    def __eq__(self, other: Any) -> bool:
        if self.__class__ is other.__class__:
            return self.type == other.type and \
                   self.layer_name == other.layer_name and \
                   self.weights_attr_name == other.weights_attr_name
        return False

    def __str__(self) -> str:
        return super().__str__() + ' ' + self.weights_attr_name


class TFOperationWithWeights(TFLayerWeight):
    """
    `TFOperationWithWeights` defines an operation with weights.

    For example, `TFOperationWithWeights` is used to specify the operation with
    weights in the removal command to remove from the model.
    """

    def __init__(self, layer_name: str, weights_attr_name: str, operation_name: str):
        super().__init__(layer_name, weights_attr_name)
        self._operation_name = operation_name

    @property
    def operation_name(self) -> str:
        return self._operation_name

    def __eq__(self, other: Any) -> bool:
        if self.__class__ is other.__class__:
            return self.type == other.type and \
                   self.layer_name == other.layer_name and \
                   self.weights_attr_name == other.weights_attr_name and \
                   self.operation_name == other.operation_name
        return False

    def __str__(self) -> str:
        return super().__str__() + ' ' + self.operation_name


class TFInsertionCommand(TransformationCommand):
    """
    Inserts objects at the target point in the TensorFlow model graph.
    """

    def __init__(self,
                 target_point: TargetPoint,
                 callable_object: Optional[Callable] = None,
                 priority: Optional[TransformationPriority] = None):
        super().__init__(TransformationType.INSERT, target_point)
        self.callable_objects = []
        if callable_object is not None:
            _priority = TransformationPriority.DEFAULT_PRIORITY \
                if priority is None else priority
            self.callable_objects.append((callable_object, _priority))

    @property
    def insertion_objects(self) -> List[Callable]:
        return [x for x, _ in self.callable_objects]

    def union(self, other: TransformationCommand) -> 'TFInsertionCommand':
        if not self.check_command_compatibility(other):
            raise ValueError('{} and {} commands could not be united'.format(
                type(self).__name__, type(other).__name__))

        com = TFInsertionCommand(self.target_point)
        com.callable_objects = self.callable_objects + other.callable_objects
        com.callable_objects = sorted(com.callable_objects, key=lambda x: x[1])
        return com


class TFRemovalCommand(TransformationCommand):
    """
    Removes the target object.
    """

    def __init__(self, target_point: TargetPoint):
        super().__init__(TransformationType.REMOVE, target_point)

    def union(self, other: TransformationCommand) -> 'TFRemovalCommand':
        raise NotImplementedError('A command of TFRemovalCommand type '
                                  'could not be united with another command')


class TFMultipleInsertionCommands(TransformationCommand):
    """
    A list of insertion commands combined by a common global target point but
    with different target points in between.

    For example, If a layer has multiple weight variables you can use this
    transformation command to insert operations with weights for each layer
    weights variable at one multiple insertion command.
    """

    def __init__(self,
                 target_point: TargetPoint,
                 check_target_points_fn: Optional[Callable] = None,
                 commands: Optional[List[TransformationCommand]] = None):
        super().__init__(TransformationType.MULTI_INSERT, target_point)
        self.check_target_points_fn = check_target_points_fn
        if check_target_points_fn is None:
            self.check_target_points_fn = lambda tp0, tp1: tp0 == tp1
        self._commands = []
        if commands is not None:
            for cmd in commands:
                self.add_insertion_command(cmd)

    @property
    def commands(self) -> List[TransformationCommand]:
        return self._commands

    def check_insertion_command(self, command: TransformationCommand) -> bool:
        if isinstance(command, TransformationCommand) and \
                command.type == TransformationType.INSERT and \
                self.check_target_points_fn(self.target_point, command.target_point):
            return True
        return False

    def add_insertion_command(self, command: TransformationCommand) -> None:
        if not self.check_insertion_command(command):
            raise ValueError('{} command could not be added'.format(
                type(command).__name__))

        for idx, cmd in enumerate(self.commands):
            if cmd.target_point == command.target_point:
                self.commands[idx] = cmd + command
                break
        else:
            self.commands.append(command)

    def union(self, other: TransformationCommand) -> 'TFMultipleInsertionCommands':
        if not self.check_command_compatibility(other):
            raise ValueError('{} and {} commands could not be united'.format(
                type(self).__name__, type(other).__name__))

        def make_check_target_points_fn(fn1, fn2):
            def check_target_points(tp0, tp1):
                return fn1(tp0, tp1) or fn2(tp0, tp1)
            return check_target_points

        check_target_points_fn = self.check_target_points_fn \
            if self.check_target_points_fn == other.check_target_points_fn else \
            make_check_target_points_fn(self.check_target_points_fn, other.check_target_points_fn)

        multi_cmd = TFMultipleInsertionCommands(
            self.target_point,
            check_target_points_fn,
            self.commands
        )
        for cmd in other.commands:
            multi_cmd.add_insertion_command(cmd)
        return multi_cmd
