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

from nncf.common.utils.ordered_enum import OrderedEnum


class TransformationPriority(OrderedEnum):
    DEFAULT_PRIORITY = 0
    FP32_TENSOR_STATISTICS_OBSERVATION = 1
    PRUNING_PRIORITY = 2
    SPARSIFICATION_PRIORITY = 3
    QUANTIZATION_PRIORITY = 11


class TransformationType(OrderedEnum):
    INSERT = 0
    MULTI_INSERT = 1
    REMOVE = 2


class TargetType(OrderedEnum):
    LAYER = 0
    BEFORE_LAYER = 1
    AFTER_LAYER = 2
    PRE_LAYER_OPERATION = 3
    POST_LAYER_OPERATION = 4
    OPERATION_WITH_WEIGHTS = 5
    OPERATOR_PRE_HOOK = 6
    OPERATOR_POST_HOOK = 7


class TargetPoint:
    """
    The base class for all target points.

    A target point is an object or spot in the model graph. It can be a layer,
    weights, position before or after layer and etc.

    For example, the transformation commands use `TargetPoint` to specify
    the target point in the model graph to which the transformation command
    will be applied.
    """

    def __init__(self, target_type: TargetType):
        """
        Constructor

        :param target_type: Type of the target point
        """
        self._target_type = target_type

    @property
    def type(self) -> TargetType:
        return self._target_type

    def __eq__(self, other: Any) -> bool:
        if self.__class__ is other.__class__:
            return self.type == other.type
        return False

    def __str__(self) -> str:
        return str(self.type)

    def __hash__(self) -> int:
        return hash(str(self))


class TransformationCommand:
    """
    The base class for all transformation commands
    """

    def __init__(self, command_type: TransformationType, target_point: TargetPoint):
        """
        Constructor

        :param command_type: Type of the transformation command
        :param target_point: Target point, the object or spot in the model graph
            to which the transformation command will be applied.
        """
        self._command_type = command_type
        self._target_point = target_point

    @property
    def type(self) -> TransformationType:
        return self._command_type

    @property
    def target_point(self) -> TargetPoint:
        return self._target_point

    def check_command_compatibility(self, command: 'TransformationCommand') -> bool:
        return self.__class__ == command.__class__ and \
               self.type == command.type and \
               self.target_point == command.target_point

    def union(self, other: 'TransformationCommand') -> 'TransformationCommand':
        raise NotImplementedError()

    def __add__(self, other: 'TransformationCommand') -> 'TransformationCommand':
        return self.union(other)
