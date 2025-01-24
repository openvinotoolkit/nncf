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

from typing import Any, Callable, Dict, List, Optional, Union

from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationCommand
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.common.graph.transformations.commands import TransformationType
from nncf.common.stateful_classes_registry import TF_STATEFUL_CLASSES


class TFLayerPointStateNames:
    LAYER_NAME = "layer_name"
    TARGET_TYPE = "target_type"


@TF_STATEFUL_CLASSES.register()
class TFLayerPoint(TargetPoint):
    """
    `TFLayerPoint` defines an object or spot relative to the layer in the
    TensorFlow model graph. It can be the layer itself, layer weights, specific
    spots in the model graph, for example, insertion spots before/after layer
    and etc.
    """

    _state_names = TFLayerPointStateNames

    def __init__(self, target_type: TargetType, layer_name: str):
        super().__init__(target_type)
        self._layer_name = layer_name

    @property
    def layer_name(self) -> str:
        return self._layer_name

    def __eq__(self, other: "TFLayerPoint") -> bool:
        if isinstance(other, TFLayerPoint):
            return self.type == other.type and self.layer_name == other.layer_name
        return False

    def __str__(self) -> str:
        return super().__str__() + " " + self.layer_name

    def get_state(self) -> Dict[str, Any]:
        """
        Returns a dictionary with Python data structures (dict, list, tuple, str, int, float, True, False, None) that
        represents state of the object.

        :return: state of the object
        """
        return {
            self._state_names.TARGET_TYPE: self._target_type.get_state(),
            self._state_names.LAYER_NAME: self.layer_name,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "TFLayerPoint":
        """
        Creates the object from its state.

        :param state: Output of `get_state()` method.
        """
        kwargs = {
            cls._state_names.TARGET_TYPE: TargetType.from_state(state[cls._state_names.TARGET_TYPE]),
            cls._state_names.LAYER_NAME: state[cls._state_names.LAYER_NAME],
        }
        return cls(**kwargs)


class TFMultiLayerPoint:
    """
    `TFMultiLayerPoint` stores a list of target points that will be
    combined into shared callable object. Each point can be spots in the model
    graph: insertion spots before/after layer.
    """

    def __init__(self, target_points: List[TargetPoint]):
        self._target_points = target_points

    @property
    def target_points(self) -> List[TargetPoint]:
        return self._target_points

    def __str__(self) -> str:
        return f"TFMultiLayerPoint: {[str(t) for t in self._target_points]}"


class TFLayerStateNames:
    LAYER_NAME = "layer_name"


@TF_STATEFUL_CLASSES.register()
class TFLayer(TFLayerPoint):
    """
    `TFLayer` defines a layer in the TensorFlow model graph.

    For example, `TFLayer` is used to specify the layer in the removal command
    to remove from the model.
    """

    _state_names = TFLayerStateNames

    def __init__(self, layer_name: str):
        super().__init__(TargetType.LAYER, layer_name)

    def get_state(self) -> Dict[str, Any]:
        """
        Returns a dictionary with Python data structures (dict, list, tuple, str, int, float, True, False, None) that
        represents state of the object.

        :return: state of the object
        """
        return {
            self._state_names.LAYER_NAME: self.layer_name,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "TFLayer":
        """
        Creates the object from its state.

        :param state: Output of `get_state()` method.
        """
        return cls(**state)


class TFBeforeLayerStateNames:
    LAYER_NAME = "layer_name"
    INSTANCE_IDX = "instance_idx"
    INPUT_PORT_ID = "input_port_id"


@TF_STATEFUL_CLASSES.register()
class TFBeforeLayer(TFLayerPoint):
    """
    `TFBeforeLayer` defines a spot before the layer in the TensorFlow model graph.

    For example, `TFBeforeLayer` is used in the insertion commands to specify
    where the new object should be inserted.
    """

    _state_names = TFBeforeLayerStateNames

    def __init__(self, layer_name: str, instance_idx: int = 0, input_port_id: int = 0):
        super().__init__(TargetType.BEFORE_LAYER, layer_name)
        self._instance_idx = instance_idx
        self._input_port_id = input_port_id

    @property
    def instance_idx(self) -> int:
        return self._instance_idx

    @property
    def input_port_id(self) -> int:
        return self._input_port_id

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, TFBeforeLayer)
            and self.layer_name == other.layer_name
            and self.instance_idx == other.instance_idx
            and self.input_port_id == other.input_port_id
        )

    def __str__(self) -> str:
        return " ".join([super().__str__(), str(self.instance_idx), str(self.input_port_id)])

    def __hash__(self) -> int:
        return hash(str(self))

    def get_state(self) -> Dict[str, Any]:
        """
        Returns a dictionary with Python data structures (dict, list, tuple, str, int, float, True, False, None) that
        represents state of the object.

        :return: state of the object
        """
        return {
            self._state_names.LAYER_NAME: self.layer_name,
            self._state_names.INSTANCE_IDX: self.instance_idx,
            self._state_names.INPUT_PORT_ID: self.input_port_id,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "TFBeforeLayer":
        """
        Creates the object from its state.

        :param state: Output of `get_state()` method.
        """
        return cls(**state)


class TFAfterLayerStateNames:
    LAYER_NAME = "layer_name"
    INSTANCE_IDX = "instance_idx"
    OUTPUT_PORT_ID = "output_port_id"


@TF_STATEFUL_CLASSES.register()
class TFAfterLayer(TFLayerPoint):
    """
    `TFAfterLayer` defines a spot after the layer in the TensorFlow model graph.

    For example, `TFAfterLayer` is used in the insertion commands to specify
    where the new object should be inserted.
    """

    _state_names = TFAfterLayerStateNames

    def __init__(self, layer_name: str, instance_idx: int = 0, output_port_id: int = 0):
        super().__init__(TargetType.AFTER_LAYER, layer_name)
        self._instance_idx = instance_idx
        self._output_port_id = output_port_id

    @property
    def instance_idx(self) -> int:
        return self._instance_idx

    @property
    def output_port_id(self) -> int:
        return self._output_port_id

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, TFAfterLayer)
            and self.type == other.type
            and self.layer_name == other.layer_name
            and self.instance_idx == other.instance_idx
            and self._output_port_id == other.output_port_id
        )

    def __str__(self) -> str:
        return " ".join([super().__str__(), str(self.instance_idx), str(self.output_port_id)])

    def __hash__(self) -> int:
        return hash(str(self))

    def get_state(self) -> Dict[str, Any]:
        """
        Returns a dictionary with Python data structures (dict, list, tuple, str, int, float, True, False, None) that
        represents state of the object.

        :return: state of the object
        """
        return {
            self._state_names.LAYER_NAME: self.layer_name,
            self._state_names.INSTANCE_IDX: self.instance_idx,
            self._state_names.OUTPUT_PORT_ID: self.output_port_id,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "TFAfterLayer":
        """
        Creates the object from its state.

        :param state: Output of `get_state()` method.
        """
        return cls(**state)


class TFLayerWeightsStateNames:
    LAYER_NAME = "layer_name"
    WEIGHTS_ATTR_NAME = "weights_attr_name"


@TF_STATEFUL_CLASSES.register()
class TFLayerWeight(TFLayerPoint):
    """
    `TFLayerWeight` defines the layer weights.

    For example, `TFLayerWeight` is used in the insertion command to specify
    the layer weights for which an operation with weights should be inserted.
    """

    _state_names = TFLayerWeightsStateNames

    def __init__(self, layer_name: str, weights_attr_name: str):
        super().__init__(TargetType.OPERATION_WITH_WEIGHTS, layer_name)
        self._weights_attr_name = weights_attr_name

    @property
    def weights_attr_name(self) -> str:
        return self._weights_attr_name

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, TFLayerWeight)
            and self.type == other.type
            and self.layer_name == other.layer_name
            and self.weights_attr_name == other.weights_attr_name
        )

    def __str__(self) -> str:
        return super().__str__() + " " + self.weights_attr_name

    def __hash__(self) -> int:
        return hash(str(self))

    def get_state(self) -> Dict[str, Any]:
        """
        Returns a dictionary with Python data structures (dict, list, tuple, str, int, float, True, False, None) that
        represents state of the object.

        :return: state of the object
        """
        return {
            self._state_names.LAYER_NAME: self.layer_name,
            self._state_names.WEIGHTS_ATTR_NAME: self.weights_attr_name,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "TFLayerWeight":
        """
        Creates the object from its state.

        :param state: Output of `get_state()` method.
        """
        return cls(**state)


class TFOperationWithWeightsStateNames:
    LAYER_NAME = "layer_name"
    WEIGHTS_ATTR_NAME = "weights_attr_name"
    OPERATION_NAME = "operation_name"


@TF_STATEFUL_CLASSES.register()
class TFOperationWithWeights(TFLayerWeight):
    """
    `TFOperationWithWeights` defines an operation with weights.

    For example, `TFOperationWithWeights` is used to specify the operation with
    weights in the removal command to remove from the model.
    """

    _state_names = TFOperationWithWeightsStateNames

    def __init__(self, layer_name: str, weights_attr_name: str, operation_name: str):
        super().__init__(layer_name, weights_attr_name)
        self._operation_name = operation_name

    @property
    def operation_name(self) -> str:
        return self._operation_name

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, TFOperationWithWeights)
            and self.type == other.type
            and self.layer_name == other.layer_name
            and self.weights_attr_name == other.weights_attr_name
            and self.operation_name == other.operation_name
        )

    def __str__(self) -> str:
        return super().__str__() + " " + self.operation_name

    def __hash__(self) -> int:
        return hash(str(self))

    def get_state(self) -> Dict[str, Any]:
        """
        Returns a dictionary with Python data structures (dict, list, tuple, str, int, float, True, False, None) that
        represents state of the object.

        :return: state of the object
        """
        return {
            self._state_names.LAYER_NAME: self._layer_name,
            self._state_names.WEIGHTS_ATTR_NAME: self._weights_attr_name,
            self._state_names.OPERATION_NAME: self._operation_name,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "TFOperationWithWeights":
        """
        Creates the object from its state.

        :param state: Output of `get_state()` method.
        """
        return cls(**state)


class TFTransformationCommand(TransformationCommand):
    """
    The base class for all Tensorflow transformation commands.
    """

    def __init__(self, command_type: TransformationType, target_point: TargetPoint):
        """
        Constructor.

        :param command_type: Type of the transformation command.
        :param target_point: Target point, the object or spot in the model graph
            to which the transformation command will be applied.
        """
        super().__init__(command_type, target_point)

    def check_command_compatibility(self, command: "TFTransformationCommand") -> bool:
        return (
            isinstance(command, TFTransformationCommand)
            and self.type == command.type
            and self.target_point == command.target_point
        )

    def union(self, other: "TFTransformationCommand") -> "TFTransformationCommand":
        raise NotImplementedError()

    def __add__(self, other: "TFTransformationCommand") -> "TFTransformationCommand":
        return self.union(other)


class TFInsertionCommand(TFTransformationCommand):
    """
    Inserts objects at the target point in the TensorFlow model graph.
    """

    def __init__(
        self,
        target_point: Union[TargetPoint, TFMultiLayerPoint],
        callable_object: Optional[Callable] = None,
        priority: Optional[TransformationPriority] = None,
    ):
        super().__init__(TransformationType.INSERT, target_point)
        self.callable_objects = []
        if callable_object is not None:
            _priority = TransformationPriority.DEFAULT_PRIORITY if priority is None else priority
            self.callable_objects.append((callable_object, _priority))

    @property
    def insertion_objects(self) -> List[Callable]:
        return [x for x, _ in self.callable_objects]

    def union(self, other: TFTransformationCommand) -> "TFInsertionCommand":
        if isinstance(self.target_point, TFMultiLayerPoint):
            raise NotImplementedError(
                "A command of TFInsertionCommand type with TFMultiLayerPoint "
                "could not be united with another command"
            )

        if not self.check_command_compatibility(other):
            raise ValueError("{} and {} commands could not be united".format(type(self).__name__, type(other).__name__))

        com = TFInsertionCommand(self.target_point)
        com.callable_objects = self.callable_objects + other.callable_objects
        com.callable_objects = sorted(com.callable_objects, key=lambda x: x[1])
        return com


class TFRemovalCommand(TFTransformationCommand):
    """
    Removes the target object.
    """

    def __init__(self, target_point: TargetPoint):
        super().__init__(TransformationType.REMOVE, target_point)

    def union(self, other: TFTransformationCommand) -> "TFRemovalCommand":
        raise NotImplementedError("A command of TFRemovalCommand type could not be united with another command")


class TFMultipleInsertionCommands(TFTransformationCommand):
    """
    A list of insertion commands combined by a common global target point but
    with different target points in between.

    For example, If a layer has multiple weight variables you can use this
    transformation command to insert operations with weights for each layer
    weights variable at one multiple insertion command.
    """

    def __init__(
        self,
        target_point: TargetPoint,
        check_target_points_fn: Optional[Callable] = None,
        commands: Optional[List[TFTransformationCommand]] = None,
    ):
        super().__init__(TransformationType.MULTI_INSERT, target_point)
        self.check_target_points_fn = check_target_points_fn
        if check_target_points_fn is None:
            self.check_target_points_fn = lambda tp0, tp1: tp0 == tp1
        self._commands = []
        if commands is not None:
            for cmd in commands:
                self.add_insertion_command(cmd)

    @property
    def commands(self) -> List[TFTransformationCommand]:
        return self._commands

    def check_insertion_command(self, command: TFTransformationCommand) -> bool:
        if (
            isinstance(command, TFTransformationCommand)
            and command.type == TransformationType.INSERT
            and self.check_target_points_fn(self.target_point, command.target_point)
        ):
            return True
        return False

    def add_insertion_command(self, command: TFTransformationCommand) -> None:
        if not self.check_insertion_command(command):
            raise ValueError("{} command could not be added".format(type(command).__name__))

        for idx, cmd in enumerate(self.commands):
            if cmd.target_point == command.target_point:
                self.commands[idx] = cmd + command
                break
        else:
            self.commands.append(command)

    def union(self, other: TFTransformationCommand) -> "TFMultipleInsertionCommands":
        if not self.check_command_compatibility(other):
            raise ValueError("{} and {} commands could not be united".format(type(self).__name__, type(other).__name__))

        def make_check_target_points_fn(fn1, fn2):
            def check_target_points(tp0, tp1):
                return fn1(tp0, tp1) or fn2(tp0, tp1)

            return check_target_points

        check_target_points_fn = (
            self.check_target_points_fn
            if self.check_target_points_fn == other.check_target_points_fn
            else make_check_target_points_fn(self.check_target_points_fn, other.check_target_points_fn)
        )

        multi_cmd = TFMultipleInsertionCommands(self.target_point, check_target_points_fn, self.commands)
        for cmd in other.commands:
            multi_cmd.add_insertion_command(cmd)
        return multi_cmd
