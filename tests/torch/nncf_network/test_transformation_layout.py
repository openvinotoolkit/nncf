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

import itertools
from typing import Tuple, Type, Union

import pytest
import torch

from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.torch import wrap_model
from nncf.torch.graph.transformations.commands import ExtraCompressionModuleType
from nncf.torch.graph.transformations.commands import PTInsertionCommand
from nncf.torch.graph.transformations.commands import PTSharedFnInsertionCommand
from nncf.torch.graph.transformations.layout import PTTransformationLayout
from nncf.torch.model_transformer import PTModelTransformer
from tests.torch.helpers import commands_are_equal
from tests.torch.nncf_network.helpers import AVAILABLE_TARGET_TYPES
from tests.torch.nncf_network.helpers import InsertionCommandBuilder

TARGET_TYPE_VS_TARGET_TYPE_DICT_FOR_NOT_REPLACED_MODULES = {
    TargetType.PRE_LAYER_OPERATION: TargetType.OPERATOR_PRE_HOOK,
    TargetType.POST_LAYER_OPERATION: TargetType.OPERATOR_POST_HOOK,
    TargetType.OPERATION_WITH_WEIGHTS: TargetType.OPERATOR_PRE_HOOK,
    TargetType.OPERATOR_PRE_HOOK: TargetType.OPERATOR_PRE_HOOK,
    TargetType.OPERATOR_POST_HOOK: TargetType.OPERATOR_POST_HOOK,
}


@pytest.fixture(name="trace_parameters", params=(True, False))
def trace_parameters_fixture(request) -> bool:
    return request.param


def _get_trace_params_target_types_command_builders_and_models_cls() -> (
    Tuple[bool, Type[torch.nn.Module], TargetType, callable]
):
    """
    Returns list of all avaliable command builders
    """
    retval = []
    for (
        trace_parameters,
        model_cls,
        target_type,
    ) in itertools.product(
        (True, False),
        InsertionCommandBuilder.AVAILABLE_MODELS,
        AVAILABLE_TARGET_TYPES,
    ):
        for command_builder, command_cls in InsertionCommandBuilder(model_cls).get_command_builders():
            if not trace_parameters and InsertionCommandBuilder.is_unsupported_by_transformer_command(
                command_cls, target_type
            ):
                print(f"PTSharedFnInsertionCommand is not supporting target type {target_type}")
                continue
            retval.append((trace_parameters, model_cls, target_type, command_builder))
    return retval


def _translate_target_types(trace_parameters: bool, command: Union[PTInsertionCommand, PTSharedFnInsertionCommand]):
    """
    Translates target types in case trace_parameters is True
    """
    if not trace_parameters:
        return
    if isinstance(command, PTInsertionCommand):
        target_points = [command.target_point]
    else:
        target_points = command.target_points

    for target_point in target_points:
        new_target_type = TARGET_TYPE_VS_TARGET_TYPE_DICT_FOR_NOT_REPLACED_MODULES[target_point.type]
        target_point._target_type = new_target_type
        target_point.target_type = new_target_type


@pytest.mark.parametrize(
    "trace_parameters,model_cls,target_type,command_builder",
    _get_trace_params_target_types_command_builders_and_models_cls(),
)
def test_get_applied_modification_commands(
    model_cls: Type[torch.nn.Module], command_builder: callable, target_type: TargetType, trace_parameters: bool
):
    model = model_cls()
    nncf_model = wrap_model(model, torch.zeros(model_cls.INPUT_SHAPE), trace_parameters=trace_parameters)
    model_transformer = PTModelTransformer(nncf_model)

    layout = PTTransformationLayout()
    command = command_builder(target_type, TransformationPriority.DEFAULT_PRIORITY, trace_parameters=trace_parameters)
    layout.register(command)
    model_transformer.transform(layout)

    applied_commands = nncf_model.nncf.transformation_layout()

    assert len(applied_commands.transformations) == 1
    applied_command = applied_commands.transformations[0]
    _translate_target_types(trace_parameters, command)
    assert commands_are_equal(command, applied_command, check_priority=False, check_hooks_group_name=False)


@pytest.mark.parametrize(
    "trace_parameters,model_cls,target_type,command_builder",
    _get_trace_params_target_types_command_builders_and_models_cls(),
)
def test_priority_of_get_applied_modification_commands(
    command_builder: callable, model_cls: Type[torch.nn.Module], target_type: TargetType, trace_parameters: bool
):
    layout = PTTransformationLayout()
    commands = dict()
    for priority in (0, 2, 1):
        command = command_builder(
            target_type, priority, op_unique_name=f"UNIQUE_NAME_{priority}", trace_parameters=trace_parameters
        )
        layout.register(command)
        commands[priority] = command

    model = model_cls()
    nncf_model = wrap_model(model, torch.zeros(model_cls.INPUT_SHAPE), trace_parameters=trace_parameters)
    model_transformer = PTModelTransformer(nncf_model)

    model_transformer.transform(layout)

    applied_commands = nncf_model.nncf.transformation_layout()
    assert len(applied_commands.transformations) == len(commands)
    for applied_command in applied_commands.transformations:
        command = commands[applied_command.priority]
        _translate_target_types(trace_parameters, command)
        assert commands_are_equal(command, applied_command, check_priority=False, check_hooks_group_name=False)


@pytest.mark.parametrize("model_cls", InsertionCommandBuilder.AVAILABLE_MODELS)
def test_all_possible_combinations_of_commands_for_get_applied_commands(
    model_cls: Type[torch.nn.Module], trace_parameters: bool
):
    dummy_state = "DummyState"
    commands = InsertionCommandBuilder(model_cls).get_all_available_commands(
        dummy_state, skip_model_transformer_unsupported=not trace_parameters, trace_parameters=trace_parameters
    )

    model = model_cls()
    nncf_model = wrap_model(model, torch.zeros(model_cls.INPUT_SHAPE), trace_parameters=trace_parameters)
    model_transformer = PTModelTransformer(nncf_model)

    model_transformer.transform(commands)

    applied_commands = nncf_model.nncf.transformation_layout()
    assert len(applied_commands.transformations) == len(commands.transformations)
    for command in commands.transformations:
        _translate_target_types(trace_parameters, command)
        eq_commands = (
            commands_are_equal(command, applied_command, check_priority=False, check_hooks_group_name=False)
            for applied_command in applied_commands.transformations
        )
        if sum(map(int, eq_commands)) != 1:
            raise RuntimeError(f"Command {command} has no pair in recovered commands")


@pytest.mark.parametrize("target_type", (TargetType.OPERATION_WITH_WEIGHTS, TargetType.OPERATOR_PRE_HOOK))
@pytest.mark.parametrize("model_cls", InsertionCommandBuilder.AVAILABLE_MODELS)
def test_get_applied_modification_commands_broken_call_hook(
    model_cls: Type[torch.nn.Module], target_type: TargetType, trace_parameters: bool
):
    model = model_cls()
    nncf_model = wrap_model(model, torch.zeros(model_cls.INPUT_SHAPE), trace_parameters=trace_parameters)
    model_transformer = PTModelTransformer(nncf_model)

    layout = PTTransformationLayout()
    command = InsertionCommandBuilder(model_cls).create_pt_shared_fn_insertion_command(
        target_type=target_type,
        priority=0,
        compression_module_type=ExtraCompressionModuleType.EXTERNAL_OP,
        trace_parameters=trace_parameters,
    )
    layout.register(command)
    model_transformer.transform(layout)

    nncf_model.nncf.external_op.clear()
    with pytest.raises(AssertionError):
        nncf_model.nncf.transformation_layout()
