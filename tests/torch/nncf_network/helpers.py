# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import itertools

import torch

from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.torch.graph.transformations.commands import ExtraCompressionModuleType
from nncf.torch.graph.transformations.commands import PTInsertionCommand
from nncf.torch.graph.transformations.commands import PTSharedFnInsertionCommand
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.layer_utils import COMPRESSION_MODULES
from tests.torch.helpers import DummyOpWithState


class SimplestModel(torch.nn.Module):
    INPUT_SIZE = [1, 1, 32, 32]

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return self.conv(x)


AVAILABLE_TARGET_TYPES = (
    TargetType.OPERATION_WITH_WEIGHTS,
    TargetType.OPERATOR_PRE_HOOK,
    TargetType.OPERATOR_POST_HOOK,
    TargetType.PRE_LAYER_OPERATION,
    TargetType.POST_LAYER_OPERATION,
)


class InsertionCommandBuilder:
    """
    Contains methods which allows to build all possible commands
    for the TwoConvTestModel
    """

    NNCF_CONV_NODES_NAMES = [
        "TwoConvTestModel/Sequential[features]/Sequential[0]/NNCFConv2d[0]/conv2d_0",
        "TwoConvTestModel/Sequential[features]/Sequential[1]/NNCFConv2d[0]/conv2d_0",
    ]
    CONV_NODES_NAMES = [
        "TwoConvTestModel/Sequential[features]/Sequential[0]/Conv2d[0]/conv2d_0",
        "TwoConvTestModel/Sequential[features]/Sequential[1]/Conv2d[0]/conv2d_0",
    ]

    TRACE_VS_NODE_NAMES = {True: CONV_NODES_NAMES, False: NNCF_CONV_NODES_NAMES}

    @classmethod
    def create_pt_insertion_command(
        cls,
        target_type: TargetType,
        priority: TransformationPriority,
        trace_parameters: bool,
        fn=None,
        group: str = "default_group",
    ):
        target_point = PTTargetPoint(
            target_type=target_type, target_node_name=cls.TRACE_VS_NODE_NAMES[trace_parameters][0], input_port_id=0
        )
        if fn is None:
            fn = DummyOpWithState("DUMMY_STATE")
        return PTInsertionCommand(point=target_point, fn=fn, priority=priority, hooks_group_name=group)

    @classmethod
    def create_pt_shared_fn_insertion_command(
        cls,
        target_type: TargetType,
        priority: TransformationPriority,
        trace_parameters: bool,
        compression_module_type: ExtraCompressionModuleType,
        fn=None,
        group: str = "default_group",
        op_unique_name: str = "UNIQUE_NAME",
    ):
        target_points = []

        for node_name in cls.TRACE_VS_NODE_NAMES[trace_parameters]:
            target_points.append(PTTargetPoint(target_type=target_type, target_node_name=node_name, input_port_id=0))
        if fn is None:
            fn = DummyOpWithState("DUMMY_STATE")
        return PTSharedFnInsertionCommand(
            target_points=target_points,
            fn=fn,
            compression_module_type=compression_module_type,
            op_unique_name=op_unique_name,
            priority=priority,
            hooks_group_name=group,
        )

    @staticmethod
    def get_command_builders():
        return (
            InsertionCommandBuilder.create_pt_insertion_command,
            functools.partial(
                InsertionCommandBuilder.create_pt_shared_fn_insertion_command,
                compression_module_type=ExtraCompressionModuleType.EXTERNAL_OP,
            ),
            functools.partial(
                InsertionCommandBuilder.create_pt_shared_fn_insertion_command,
                compression_module_type=ExtraCompressionModuleType.EXTERNAL_QUANTIZER,
            ),
        )

    @classmethod
    def get_command_builders_with_types(cls):
        return tuple(zip(cls.get_command_builders(), cls.COMMAND_TYPES))

    COMMAND_TYPES = [PTInsertionCommand, PTSharedFnInsertionCommand, PTSharedFnInsertionCommand]
    PRIORITIES = (TransformationPriority.QUANTIZATION_PRIORITY, TransformationPriority.QUANTIZATION_PRIORITY.value + 1)

    @classmethod
    def get_all_available_commands(
        cls, dummy_op_state, trace_parameters, skip_model_transformer_unsupported=False
    ) -> TransformationLayout:
        """
        Returns all possible commands to insert:
        all target types x all command class x all compression module types x different priorities.
        """
        layout = TransformationLayout()
        for idx, (target_type, (command_builder, command_type), priority) in enumerate(
            itertools.product(
                AVAILABLE_TARGET_TYPES, zip(cls.get_command_builders(), cls.COMMAND_TYPES), cls.PRIORITIES
            )
        ):
            if command_type is PTSharedFnInsertionCommand:
                if skip_model_transformer_unsupported and target_type in [
                    TargetType.PRE_LAYER_OPERATION,
                    TargetType.POST_LAYER_OPERATION,
                ]:
                    continue
                command = cls._create_command(
                    command_builder,
                    target_type,
                    priority,
                    dummy_op_state,
                    op_unique_name=f"UNIQUE_NAME_{idx}",
                    trace_parameters=trace_parameters,
                )
            else:
                command = cls._create_command(
                    command_builder, target_type, priority, dummy_op_state, trace_parameters=trace_parameters
                )

            layout.register(command)
        return layout

    @staticmethod
    def _create_command(
        command_builder,
        target_type,
        priority,
        dummy_op_state,
        trace_parameters,
        op_unique_name=None,
    ):
        group_name = "CUSTOM_HOOKS_GROUP_NAME"

        if DummyOpWithState.__name__ not in COMPRESSION_MODULES.registry_dict:
            registered_dummy_op_cls = COMPRESSION_MODULES.register()(DummyOpWithState)
        else:
            registered_dummy_op_cls = DummyOpWithState
        dummy_op = registered_dummy_op_cls(dummy_op_state)
        if op_unique_name is None:
            command = command_builder(
                target_type, priority, fn=dummy_op, group=group_name, trace_parameters=trace_parameters
            )
        else:
            command = command_builder(
                target_type,
                priority,
                fn=dummy_op,
                group=group_name,
                op_unique_name=op_unique_name,
                trace_parameters=trace_parameters,
            )

        return command
