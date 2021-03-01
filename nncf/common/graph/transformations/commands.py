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
    """
    Defines priorities for compression and service operations that are
    added as modifications to the original model graph in order to turn it into a
    'compressed' model, i.e. a model that supports compression-aware training and
    export. Certain compression algorithms may need to act on one and the same
    spot in the model's control flow graph, and proper priority-based ordering is
    required for functionally correct results that are invariant w.r.t. compression
    algorithm application ordering.

    Rationales:
    * quantization should occur after sparsification/pruning, otherwise the dependency
    of sparsity level on the value threshold becomes discontinuous which impacts the
    corresponding algos.
    """
    DEFAULT_PRIORITY = 0
    FP32_TENSOR_STATISTICS_OBSERVATION = 1
    PRUNING_PRIORITY = 2
    SPARSIFICATION_PRIORITY = 3
    QUANTIZATION_PRIORITY = 11


class TargetType(OrderedEnum):
    """
    Describes the types of actions that can be applied to the NNCF internal graph representation
    of the compressed model.

    Definitions used below: TF-nodes - visible in TF-graph, TF-operations - invisible in TF-graph
    Both TF-nodes and TF-operations may be used for hooking in TF.

    `LAYER` - add or remove a TF-node in a TF-graph
    `BEFORE_LAYER` - additional TF-node in TF-graph before TF-layer op
    `AFTER_LAYER` - additional TF-node in TF-graph after TF-layer op
    `PRE_LAYER_OPERATION` - A pre-operation that can access attrs of the associated PT-module or TF-layer
    `POST_LAYER_OPERATION` - A post-operation that can access attrs of the associated PT-module or TF-layer
    `OPERATION_WITH_WEIGHTS` - same as PRE_LAYER_OPERATION, but targets weights of the layer/module specifically
    `OPERATOR_PRE_HOOK` - A pre-operation in PT that does not in general access module attrs - N/A in TF
    `OPERATOR_POST_HOOK` - A post-operation in PT that does not in general access module attrs - N/A in TF

    Notes: `PRE_LAYER_OPERATION`, `POST_LAYER_OPERATION` and `OPERATION_WITH_WEIGHTS` add a TF-operation in TF, not a
            TF-node. In PT, these map to module pre- and post-ops.
    """
    LAYER = 0
    BEFORE_LAYER = 1
    AFTER_LAYER = 2
    PRE_LAYER_OPERATION = 3
    POST_LAYER_OPERATION = 4
    OPERATION_WITH_WEIGHTS = 5
    OPERATOR_PRE_HOOK = 6
    OPERATOR_POST_HOOK = 7


class TransformationType(OrderedEnum):
    """
    Defines the sub-types of transformations w.r.t. `TargetType`.
    `TransformationType` defines *what* to do, while `TargetType` more concerns itself with
    *where* to do it.
    """
    INSERT = 0
    MULTI_INSERT = 1
    REMOVE = 2

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
