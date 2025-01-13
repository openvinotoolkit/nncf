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

from enum import Enum
from typing import Any, Callable, Dict, List, Union

import torch

from nncf.common.graph import NNCFNodeName
from nncf.common.graph.transformations.commands import Command
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationCommand
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.common.graph.transformations.commands import TransformationType

DEFAULT_HOOKS_GROUP_NAME = "default_hooks_group"


class PTTargetPointStateNames:
    TARGET_NODE_NAME = "target_node_name"
    INPUT_PORT = "input_port_id"
    TARGET_TYPE = "target_type"


class PTTargetPoint(TargetPoint):
    _OPERATION_TYPES = [
        TargetType.PRE_LAYER_OPERATION,
        TargetType.POST_LAYER_OPERATION,
        TargetType.OPERATION_WITH_WEIGHTS,
    ]
    _HOOK_TYPES = [TargetType.OPERATOR_PRE_HOOK, TargetType.OPERATOR_POST_HOOK]

    _LAYER_TYPE = [TargetType.LAYER]

    _state_names = PTTargetPointStateNames

    def __init__(self, target_type: TargetType, target_node_name: NNCFNodeName, *, input_port_id: int = None):
        super().__init__(target_type)
        self.target_node_name = target_node_name
        self.target_type = target_type
        if self.target_type not in self._OPERATION_TYPES + self._HOOK_TYPES + self._LAYER_TYPE:
            raise NotImplementedError("Unsupported target type: {}".format(target_type))

        self.input_port_id = input_port_id

    def __eq__(self, other: "PTTargetPoint"):
        return (
            isinstance(other, PTTargetPoint)
            and self.target_type == other.target_type
            and self.target_node_name == other.target_node_name
            and self.input_port_id == other.input_port_id
        )

    def __str__(self):
        prefix = str(self.target_type)
        retval = prefix
        if self.target_type in self._OPERATION_TYPES + self._LAYER_TYPE:
            retval += " {}".format(self.target_node_name)
        elif self.target_type in self._HOOK_TYPES:
            if self.input_port_id is not None:
                retval += " {}".format(self.input_port_id)
            retval += " " + str(self.target_node_name)
        return retval

    def __hash__(self):
        return hash(str(self))

    def get_state(self) -> Dict[str, Any]:
        """
        Returns a dictionary with Python data structures (dict, list, tuple, str, int, float, True, False, None) that
        represents state of the object.

        :return: state of the object
        """
        return {
            self._state_names.TARGET_TYPE: self.target_type.get_state(),
            self._state_names.INPUT_PORT: self.input_port_id,
            self._state_names.TARGET_NODE_NAME: self.target_node_name,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "PTTargetPoint":
        """
        Creates the object from its state.

        :param state: Output of `get_state()` method.
        """
        kwargs = {
            cls._state_names.TARGET_TYPE: TargetType.from_state(state[cls._state_names.TARGET_TYPE]),
            cls._state_names.INPUT_PORT: state[cls._state_names.INPUT_PORT],
            cls._state_names.TARGET_NODE_NAME: state[cls._state_names.TARGET_NODE_NAME],
        }
        return cls(**kwargs)


class PTCommand(Command):
    """
    The base class for all Command for PyTorch.
    """

    def requires_graph_rebuild(self):
        """
        Return boolean flag to rebuild graph of model.

        :return: Boolean flag.
        """
        return False


class PTTransformationCommand(TransformationCommand):
    """
    The base class for all TransformationCommand for PyTorch.
    """

    def requires_graph_rebuild(self):
        """
        Return boolean flag to rebuild graph of model.

        :return: Boolean flag.
        """
        return False


class PTInsertionCommand(PTTransformationCommand):
    """
    Insertion operation to the models.
    """

    def __init__(
        self,
        point: PTTargetPoint,
        fn: Callable,
        priority: Union[TransformationPriority, int] = TransformationPriority.DEFAULT_PRIORITY,
        hooks_group_name: str = DEFAULT_HOOKS_GROUP_NAME,
    ):
        super().__init__(TransformationType.INSERT, point)
        self.fn: Callable = fn
        self.priority: TransformationPriority = priority
        self.hooks_group_name = hooks_group_name

    def requires_graph_rebuild(self):
        # Rebuild graph when adding quantization nodes.
        return self.priority == TransformationPriority.QUANTIZATION_PRIORITY


class ExtraCompressionModuleType(Enum):
    EXTERNAL_QUANTIZER = 0
    EXTERNAL_OP = 1


class PTSharedFnInsertionCommand(PTTransformationCommand):
    def __init__(
        self,
        target_points: List[PTTargetPoint],
        fn: Callable,
        op_unique_name: str,
        compression_module_type: ExtraCompressionModuleType = ExtraCompressionModuleType.EXTERNAL_OP,
        priority: Union[TransformationPriority, int] = TransformationPriority.DEFAULT_PRIORITY,
        hooks_group_name: str = DEFAULT_HOOKS_GROUP_NAME,
    ):
        super().__init__(TransformationType.INSERT, None)
        self.target_points = target_points
        self.fn = fn
        self.op_name = op_unique_name
        self.compression_module_type = compression_module_type
        self.priority = priority
        self.hooks_group_name = hooks_group_name

    def requires_graph_rebuild(self):
        return True


class PTModelExtractionCommand(PTCommand):
    """
    Extracts submodel based on the sub-model input and output names
    """

    def __init__(self, input_node_names: List[str], output_node_names: List[str]):
        """
        :param node_name: Node name that will be extracted.
        """
        super().__init__(TransformationType.EXTRACT)
        self.input_node_names = input_node_names
        self.output_node_names = output_node_names


class PTBiasCorrectionCommand(PTTransformationCommand):
    """
    Corrects bias value in the model based on the input value.
    """

    def __init__(self, target_point: PTTargetPoint, bias_value: torch.Tensor):
        """
        :param target_point: The TargetPoint instance for the correction that contains layer's information.
        :param bias_value: The bias shift value that will be added to the original bias value.
        """
        super().__init__(TransformationType.CHANGE, target_point)
        self.bias_value = bias_value


class PTWeightUpdateCommand(PTTransformationCommand):
    """
    Corrects weight value in the model based on the input value.
    """

    def __init__(self, target_point: PTTargetPoint, weight_value: torch.Tensor):
        """
        :param target_point: The TargetPoint instance for the correction that contains layer's information.
        :param weight_value: The new weight value that will be used instead of the original weight value.
        """
        super().__init__(TransformationType.CHANGE, target_point)
        self.weight_value = weight_value
