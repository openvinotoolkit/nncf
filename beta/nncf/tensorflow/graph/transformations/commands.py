"""
 Copyright (c) 2020 Intel Corporation
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

from beta.nncf.utils.ordered_enum import OrderedEnum


class TransformationPriority(OrderedEnum):
    DEFAULT_PRIORITY = 0
    SPARSIFICATION_PRIORITY = 2
    QUANTIZATION_PRIORITY = 11
    PRUNING_PRIORITY = 1


class TransformationType(OrderedEnum):
    INSERT = 0
    MULTI_INSERT = 1
    REMOVE = 2


class TargetType(OrderedEnum):
    LAYER = 0
    BEFORE_LAYER = 1
    AFTER_LAYER = 2
    WEIGHT_OPERATION = 3


class TargetPoint:
    """
    The base class for all TargetPoints

    TargetPoint specifies the object in the model graph to which the
    transformation command will be applied. It can be layer, weight and etc.
    """
    def __init__(self, target_type):
        """
        Constructor

        :param target_type: target point type
        """
        self._target_type = target_type

    @property
    def type(self):
        return self._target_type

    def __eq__(self, other):
        if self.__class__ is other.__class__:
            return self.type == other.type
        return False

    def __str__(self):
        return str(self.type)

    def __hash__(self):
        return hash(str(self))


class TransformationCommand:
    """
    The base class for all transformation commands
    """
    def __init__(self, command_type, target_point):
        """
        Constructor

        :param command_type: transformation command type
        :param target_point: target point, the object in the model
        to which the transformation command will be applied.
        """
        self._command_type = command_type
        self._target_point = target_point

    @property
    def type(self):
        return self._command_type

    @property
    def target_point(self):
        return self._target_point

    def check_command_compatibility(self, command):
        return self.__class__ == command.__class__ and \
               self.type == command.type and \
               self.target_point == command.target_point

    def union(self, other):
        pass

    def __add__(self, other):
        return self.union(other)


class LayerPoint(TargetPoint):
    def __init__(self, target_type, layer_name):
        super().__init__(target_type)
        self._layer_name = layer_name

    @property
    def layer_name(self):
        return self._layer_name

    def __eq__(self, other):
        if self.__class__ is other.__class__:
            return self.type == other.type and self.layer_name == other.layer_name
        return False

    def __str__(self):
        return super().__str__() + ' ' + self.layer_name


class Layer(LayerPoint):
    def __init__(self, layer_name):
        super().__init__(TargetType.LAYER, layer_name)


class BeforeLayer(LayerPoint):
    def __init__(self, layer_name, instance_index=0, in_port=0):
        super().__init__(TargetType.BEFORE_LAYER, layer_name)
        self._instance_index = instance_index
        self._in_port = in_port

    @property
    def instance_index(self):
        return self._instance_index

    @property
    def in_port(self):
        return self._in_port

    def __eq__(self, other):
        if self.__class__ is other.__class__:
            return self.type == other.type \
                   and self.layer_name == other.layer_name \
                   and self.instance_index == other.instance_index \
                   and self.in_port == other.in_port
        return False

    def __str__(self):
        return ' '.join([super().__str__(),
                         self.instance_index,
                         self.in_port])


class AfterLayer(LayerPoint):
    def __init__(self, layer_name, instance_index=0, out_port=0):
        super().__init__(TargetType.AFTER_LAYER, layer_name)
        self._instance_index = instance_index
        self._out_port = out_port

    @property
    def instance_index(self):
        return self._instance_index

    @property
    def out_port(self):
        return self._out_port

    def __eq__(self, other):
        if self.__class__ is other.__class__:
            return self.type == other.type \
                   and self.layer_name == other.layer_name \
                   and self.instance_index == other.instance_index \
                   and self._out_port == other.out_port
        return False

    def __str__(self):
        return ' '.join([super().__str__(),
                         self.instance_index,
                         self.out_port])


class LayerWeight(LayerPoint):
    def __init__(self, layer_name, weights_attr_name):
        super().__init__(TargetType.WEIGHT_OPERATION, layer_name)
        self._weights_attr_name = weights_attr_name

    @property
    def weights_attr_name(self):
        return self._weights_attr_name

    def __eq__(self, other):
        if self.__class__ is other.__class__:
            return self.type == other.type and \
                   self.layer_name == other.layer_name and \
                   self.weights_attr_name == other.weights_attr_name
        return False

    def __str__(self):
        return super().__str__() + ' ' + self.weights_attr_name


class LayerWeightOperation(LayerWeight):
    def __init__(self, layer_name, weights_attr_name, operation_name):
        super().__init__(layer_name, weights_attr_name)
        self._operation_name = operation_name

    @property
    def operation_name(self):
        return self._operation_name

    def __eq__(self, other):
        if self.__class__ is other.__class__:
            return self.type == other.type and \
                   self.layer_name == other.layer_name and \
                   self.weights_attr_name == other.weights_attr_name and \
                   self.operation_name == other.operation_name
        return False

    def __str__(self):
        return super().__str__() + ' ' + self.operation_name


class InsertionCommand(TransformationCommand):
    def __init__(self, target_point, callable_object=None, priority=None):
        super().__init__(TransformationType.INSERT, target_point)
        self.callable_objects = []
        if callable_object is not None:
            _priority = TransformationPriority.DEFAULT_PRIORITY \
                if priority is None else priority
            self.callable_objects.append((callable_object, _priority))

    @property
    def insertion_objects(self):
        return [x for x, _ in self.callable_objects]

    def union(self, other):
        if not self.check_command_compatibility(other):
            raise ValueError('{} and {} commands could not be united'.format(
                type(self).__name__, type(other).__name__))

        com = InsertionCommand(self.target_point)
        com.callable_objects = self.callable_objects + other.callable_objects
        com.callable_objects = sorted(com.callable_objects, key=lambda x: x[1])
        return com


class RemovalCommand(TransformationCommand):
    def __init__(self, target_point):
        super().__init__(TransformationType.REMOVE, target_point)


class MultipleInsertionCommands(TransformationCommand):
    def __init__(self, target_point, check_target_point_fn=None, commands=None):
        super().__init__(TransformationType.MULTI_INSERT, target_point)
        self.check_target_point_fn = self.check_target_point \
            if check_target_point_fn is None else check_target_point_fn
        self._commands = []
        for cmd in commands:
            self.add_insertion_command(cmd)

    @property
    def commands(self):
        return self._commands

    def check_insertion_command(self, command):
        if isinstance(command, TransformationCommand) and \
                command.type == TransformationType.INSERT and \
                self.check_target_point_fn(self.target_point, command.target_point):
            return True
        return False

    def add_insertion_command(self, command):
        if not self.check_insertion_command(command):
            raise ValueError('{} command could not be added'.format(
                type(command).__name__))

        def find_command(target_point):
            for idx, cmd in enumerate(self.commands):
                if cmd.target_point == target_point:
                    return idx
            return None

        idx = find_command(command.target_point)
        if idx is None:
            self.commands.append(command)
        else:
            self.commands[idx] = self.commands[idx] + command

    def union(self, other):
        if not self.check_command_compatibility(other):
            raise ValueError('{} and {} commands could not be united'.format(
                type(self).__name__, type(other).__name__))

        def make_check_target_point_fn(fn1, fn2):
            def check_target_point(tp0, tp1):
                return fn1(tp0, tp1) and fn2(tp0, tp1)
            return check_target_point

        check_target_point_fn = self.check_target_point_fn \
            if self.check_target_point_fn == other.check_target_point_fn else \
            make_check_target_point_fn(self.check_target_point_fn, other.check_target_point_fn)

        multi_cmd = MultipleInsertionCommands(
            self.target_point,
            check_target_point_fn,
            self.commands
        )
        for cmd in other.commands:
            multi_cmd.add_insertion_command(cmd)
        return multi_cmd

    @staticmethod
    def check_target_point(tp0, tp1):
        return tp0 == tp1
