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

from collections import defaultdict
from typing import Dict, List

from torch import nn

from nncf.common.graph.model_transformer import ModelTransformer
from nncf.common.graph.transformations.commands import Command
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.experimental.torch2.commands import PT2InsertionCommand
from nncf.experimental.torch2.function_hook.hook_storage import RemovableHookHandle
from nncf.experimental.torch2.function_hook.nncf_graph.nncf_graph_builder import GraphModelWrapper
from nncf.experimental.torch2.function_hook.wrapper import register_post_function_hook
from nncf.experimental.torch2.function_hook.wrapper import register_pre_function_hook
from nncf.torch.graph.transformations.commands import PTTargetPoint


class PT2ModelTransformer(ModelTransformer[GraphModelWrapper]):
    """
    Applies transformations upon PyTorch model.
    """

    def __init__(self, model: GraphModelWrapper):
        super().__init__(model)

        self._command_transformation_ordered_pairs = [
            (PT2InsertionCommand, self._apply_insertion_transformation),
        ]

    def transform(self, transformation_layout: TransformationLayout) -> GraphModelWrapper:
        """
        Applies transformations to the model using an out-of-place approach.
        The transformations do not affect the original model, and a new model
        is returned with the transformations applied. If there are no transformations,
        returns a new instance of the original model.

        :param transformation_layout: Transformation commands.
        :return: The new instance of a model with applied transformations.
        """

        transformations = transformation_layout.transformations
        aggregated_transformations: Dict[type, List[Command]] = defaultdict(list)
        for transformation in transformations:
            transformation_cls = transformation.__class__
            if transformation_cls not in [x[0] for x in self._command_transformation_ordered_pairs]:
                msg = f"Unsupported transformation: {transformation_cls}"
                raise ValueError(msg)
            aggregated_transformations[transformation.__class__].append(transformation)

        model = self._model.model

        for transformation_cls, transformation_fn in self._command_transformation_ordered_pairs:
            transformations = aggregated_transformations[transformation_cls]
            if transformations:
                model = transformation_fn(model, transformations)  # type: ignore[arg-type]
        return self._model

    def _apply_insertion_transformation(
        self, model: nn.Module, transformations: List[PT2InsertionCommand]
    ) -> nn.Module:
        """
        Applies insertion transformation to the model.

        :param command: Insertion transformation command.
        """
        for command in transformations:
            target_points = command.target_points
            hook_module = command.hook_module
            handle_storage = command.handle_storage

            for target_point in target_points:
                handle = insert_hook(model, hook_module, target_point)
                if handle_storage is not None:
                    handle_storage.append(handle)
        return model


def insert_hook(model: nn.Module, hook: nn.Module, target_point: PTTargetPoint) -> RemovableHookHandle:
    """
    Inserts hooks into the model.

    :param model: Pytorch model.
    :param hook: Hook to insert.
    :param target_point: Target point to insert hooks.
    """
    target_name = target_point.target_node_name
    port_id = target_point.input_port_id or 0

    if target_point.type in (
        TargetType.OPERATOR_PRE_HOOK,
        TargetType.PRE_LAYER_OPERATION,
        TargetType.OPERATION_WITH_WEIGHTS,
    ):
        return register_pre_function_hook(model=model, op_name=target_name, port_id=port_id, hook=hook)
    return register_post_function_hook(model=model, op_name=target_name, port_id=port_id, hook=hook)
