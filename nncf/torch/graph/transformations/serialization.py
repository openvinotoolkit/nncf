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
from typing import Any, Dict, Union

from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.torch.graph.transformations.commands import ExtraCompressionModuleType
from nncf.torch.graph.transformations.commands import PTInsertionCommand
from nncf.torch.graph.transformations.commands import PTSharedFnInsertionCommand
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.graph.transformations.commands import PTTransformationCommand
from nncf.torch.layer_utils import COMPRESSION_MODULES

COMPRESSION_STATE_ATTR = "compression_state"
SUPPORTED_COMMANDS = (PTSharedFnInsertionCommand, PTInsertionCommand)


def serialize_transformations(transformations_layout: TransformationLayout) -> Dict[str, Any]:
    """
    Serializes given transformation layout to a dict.

    :param transformation_layout: Given transformation layout.
    :return: Serialized representation of given transformation layout as a dict.
    """
    transformation_commands = []
    for command in transformations_layout.transformations:
        transformation_commands.append(serialize_command(command))

    return {COMPRESSION_STATE_ATTR: transformation_commands}


def deserialize_transformations(serialized_transformation_layout: Dict[str, Any]) -> TransformationLayout:
    """
    Deserializes given serialized transformation layout.

    :param serialized_transformation_layout: Given serialized transformation layout.
    :return: The deserialized transformation layout.
    """
    transformation_layout = TransformationLayout()
    for serialized_command in serialized_transformation_layout[COMPRESSION_STATE_ATTR]:
        command = deserialize_command(serialized_command)
        transformation_layout.register(command)

    return transformation_layout


def serialize_command(command: PTTransformationCommand) -> Dict[str, Any]:
    """
    Serializes given command layout to a dict.

    :param command: Given command.
    :return: Serialized representation of given command as a dict.
    """
    if not isinstance(command, SUPPORTED_COMMANDS):
        raise RuntimeError(f"Command type {command.__class__} is not supported.")

    serialized_transformation = dict()
    serialized_transformation["type"] = command.__class__.__name__
    if isinstance(command, PTSharedFnInsertionCommand):
        serialized_transformation["target_points"] = [point.get_state() for point in command.target_points]
        serialized_transformation["op_name"] = command.op_name
        serialized_transformation["compression_module_type"] = command.compression_module_type.value
    elif isinstance(command, PTInsertionCommand):
        serialized_transformation["target_point"] = command.target_point.get_state()

    # Check compression module is registered
    compression_module_name = command.fn.__class__.__name__
    if compression_module_name not in COMPRESSION_MODULES.registry_dict:
        raise RuntimeError(
            f"Could not serialize compression module with name {compression_module_name}."
            " Please register your module in the COMPRESSION_MODULES registry."
        )
    serialized_transformation["compression_module_name"] = compression_module_name
    serialized_transformation["fn_config"] = command.fn.get_config()
    serialized_transformation["hooks_group_name"] = command.hooks_group_name
    priority = command.priority
    serialized_transformation["priority"] = priority.value if isinstance(priority, Enum) else priority
    return serialized_transformation


def deserialize_command(serialized_command: Dict[str, Any]) -> Union[PTInsertionCommand, PTSharedFnInsertionCommand]:
    """
    Deserializes given serialized command.

    :param serialized_command: Given serialized command.
    :return: The deserialized command.
    """
    if serialized_command["type"] not in (command_cls.__name__ for command_cls in SUPPORTED_COMMANDS):
        raise RuntimeError(f"Command type {serialized_command['type']} is not supported.")

    module_cls = COMPRESSION_MODULES.get(serialized_command["compression_module_name"])
    fn = module_cls.from_config(serialized_command["fn_config"])
    priority = serialized_command["priority"]
    if priority in iter(TransformationPriority):
        priority = TransformationPriority(priority)

    if serialized_command["type"] == PTInsertionCommand.__name__:
        target_point = PTTargetPoint.from_state(serialized_command["target_point"])
        return PTInsertionCommand(
            point=target_point, fn=fn, priority=priority, hooks_group_name=serialized_command["hooks_group_name"]
        )

    target_points = [PTTargetPoint.from_state(state) for state in serialized_command["target_points"]]
    return PTSharedFnInsertionCommand(
        target_points=target_points,
        fn=fn,
        op_unique_name=serialized_command["op_name"],
        compression_module_type=ExtraCompressionModuleType(serialized_command["compression_module_type"]),
        priority=priority,
        hooks_group_name=serialized_command["hooks_group_name"],
    )
