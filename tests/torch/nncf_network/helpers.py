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

import functools
import itertools
from typing import Optional, Type

import torch

from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.torch.graph.transformations.commands import ExtraCompressionModuleType
from nncf.torch.graph.transformations.commands import PTInsertionCommand
from nncf.torch.graph.transformations.commands import PTSharedFnInsertionCommand
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.graph.transformations.commands import PTTransformationCommand
from nncf.torch.layer_utils import COMPRESSION_MODULES
from tests.torch.helpers import DummyOpWithState
from tests.torch.helpers import TwoConvTestModel
from tests.torch.helpers import TwoSharedConvTestModel


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
    for the given torch.nn.Module. Target module should have
    NNCF_CONV_NODES_NAMES and CONV_NODES_NAMES with names of
    target model convolutions and names of nncf-wrapped target model convolutions.
    Convolutions should be placed inside nn.sequential in .features attribute
    for test compatibility.
    """

    AVAILABLE_MODELS = (TwoConvTestModel, TwoSharedConvTestModel)

    def __init__(self, model_cls: Type[torch.nn.Module]):
        self.model_cls = model_cls

    TRACE_VS_NODE_NAMES = {True: "CONV_NODES_NAMES", False: "NNCF_CONV_NODES_NAMES"}

    @staticmethod
    def get_input_port_id(target_type: TargetType, trace_parameters: bool) -> Optional[int]:
        if target_type is TargetType.OPERATOR_PRE_HOOK:
            return 0
        if trace_parameters and target_type in [TargetType.PRE_LAYER_OPERATION, TargetType.OPERATION_WITH_WEIGHTS]:
            return 1
        return None

    def create_pt_insertion_command(
        self,
        target_type: TargetType,
        priority: TransformationPriority,
        trace_parameters: bool,
        fn: Optional[torch.nn.Module] = None,
        group: str = "default_group",
        op_unique_name: Optional[str] = None,
    ):
        attr_name = self.TRACE_VS_NODE_NAMES[trace_parameters]
        target_point = PTTargetPoint(
            target_type=target_type,
            target_node_name=getattr(self.model_cls, attr_name)[0],
            input_port_id=self.get_input_port_id(target_type, trace_parameters),
        )
        if fn is None:
            fn = DummyOpWithState("DUMMY_STATE")
        return PTInsertionCommand(point=target_point, fn=fn, priority=priority, hooks_group_name=group)

    def create_pt_shared_fn_insertion_command(
        self,
        target_type: TargetType,
        priority: TransformationPriority,
        trace_parameters: bool,
        compression_module_type: ExtraCompressionModuleType,
        fn: Optional[torch.nn.Module] = None,
        group: str = "default_group",
        op_unique_name: str = "UNIQUE_NAME",
    ):
        target_points = []
        attr_name = self.TRACE_VS_NODE_NAMES[trace_parameters]
        for node_name in getattr(self.model_cls, attr_name):
            target_points.append(
                PTTargetPoint(
                    target_type=target_type,
                    target_node_name=node_name,
                    input_port_id=self.get_input_port_id(target_type, trace_parameters),
                )
            )
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

    def get_command_builders(self):
        """
        Get all command builders available and their types in a tuple of pairs.
        """
        return (
            (self.create_pt_insertion_command, PTInsertionCommand),
            (
                functools.partial(
                    self.create_pt_shared_fn_insertion_command,
                    compression_module_type=ExtraCompressionModuleType.EXTERNAL_OP,
                ),
                PTSharedFnInsertionCommand,
            ),
            (
                functools.partial(
                    self.create_pt_shared_fn_insertion_command,
                    compression_module_type=ExtraCompressionModuleType.EXTERNAL_QUANTIZER,
                ),
                PTSharedFnInsertionCommand,
            ),
        )

    # Check priority as an enum member and as an int
    PRIORITIES = (TransformationPriority.QUANTIZATION_PRIORITY, TransformationPriority.QUANTIZATION_PRIORITY.value + 1)

    def get_all_available_commands(
        self, dummy_op_state, trace_parameters, skip_model_transformer_unsupported=False
    ) -> TransformationLayout:
        """
        Returns all possible commands to insert:
        all target types x all command class x all compression module types x different priorities.
        """
        layout = TransformationLayout()
        for idx, (target_type, (command_builder, command_type), priority) in enumerate(
            itertools.product(AVAILABLE_TARGET_TYPES, self.get_command_builders(), self.PRIORITIES)
        ):
            if skip_model_transformer_unsupported and self.is_unsupported_by_transformer_command(
                command_type, target_type
            ):
                continue
            command = self.create_one_command(
                command_builder,
                target_type,
                priority,
                dummy_op_state,
                op_unique_name=f"UNIQUE_NAME_{idx}",
                trace_parameters=trace_parameters,
            )

            layout.register(command)
        return layout

    @staticmethod
    def is_unsupported_by_transformer_command(command_type: PTTransformationCommand, target_type: TargetType) -> bool:
        """
        Returns True if insertion parameters don't supported by the PTModelTransformer otherwise False.
        """
        return command_type is PTSharedFnInsertionCommand and target_type in [
            TargetType.PRE_LAYER_OPERATION,
            TargetType.POST_LAYER_OPERATION,
        ]

    @staticmethod
    def create_one_command(
        command_builder,
        target_type,
        priority,
        dummy_op_state,
        trace_parameters,
        op_unique_name,
    ):
        """
        Creates command with specified parameters and dummy op.
        """
        # Register dummy op name in the COMPRESSION_MODULES
        if DummyOpWithState.__name__ not in COMPRESSION_MODULES.registry_dict:
            registered_dummy_op_cls = COMPRESSION_MODULES.register()(DummyOpWithState)
        else:
            registered_dummy_op_cls = DummyOpWithState
        dummy_op = registered_dummy_op_cls(dummy_op_state)

        # Build the command
        group_name = "CUSTOM_HOOKS_GROUP_NAME"
        return command_builder(
            target_type,
            priority,
            fn=dummy_op,
            group=group_name,
            op_unique_name=op_unique_name,
            trace_parameters=trace_parameters,
        )
