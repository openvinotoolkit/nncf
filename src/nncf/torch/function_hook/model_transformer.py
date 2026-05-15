# Copyright (c) 2026 Intel Corporation
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
from typing import Any, Callable, cast

from torch import nn

from nncf.common.graph.model_transformer import ModelTransformer
from nncf.common.graph.transformations.commands import Command
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.torch.function_hook.commands import PT2ConstUpdateCommand
from nncf.torch.function_hook.commands import PT2InsertionCommand
from nncf.torch.function_hook.hook_storage import RemovableHookHandle
from nncf.torch.function_hook.nncf_graph.nncf_graph_builder import GraphModelWrapper
from nncf.torch.function_hook.wrapper import register_post_function_hook
from nncf.torch.function_hook.wrapper import register_pre_function_hook
from nncf.torch.graph.transformations.commands import PTBiasCorrectionCommand
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.model_graph_manager import set_const_data
from nncf.torch.model_graph_manager import update_fused_bias
from nncf.torch.utils import get_model_device
from nncf.torch.utils import is_multidevice

TRANSFORMATION_PAIRS = tuple[tuple[type[Any], Callable[[GraphModelWrapper, list[Any]], GraphModelWrapper]], ...]


class PT2ModelTransformer(ModelTransformer[GraphModelWrapper]):
    """
    Applies transformations upon PyTorch model.
    """

    def __init__(self, model: GraphModelWrapper):
        super().__init__(model)

        self._command_transformation_ordered_pairs: TRANSFORMATION_PAIRS = (
            (PT2InsertionCommand, self._apply_insertion_transformations),
            (PTBiasCorrectionCommand, self._apply_bias_correction_transformations),
            (PT2ConstUpdateCommand, self._apply_const_update_transformations),
        )

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
        aggregated_transformations: dict[type, list[Command]] = defaultdict(list)
        for transformation in transformations:
            transformation_cls = transformation.__class__
            if transformation_cls not in [x[0] for x in self._command_transformation_ordered_pairs]:
                msg = f"Unsupported transformation: {transformation_cls}"
                raise ValueError(msg)
            aggregated_transformations[transformation.__class__].append(transformation)

        model = self._model
        for transformation_cls, transformation_fn in self._command_transformation_ordered_pairs:
            transformations = aggregated_transformations[transformation_cls]
            if transformations:
                model = transformation_fn(model, transformations)

        if aggregated_transformations.get(PT2InsertionCommand, []):
            model.reset_graph()
        return model

    def _apply_insertion_transformations(
        self, wrapped_model: GraphModelWrapper, transformations: list[PT2InsertionCommand]
    ) -> GraphModelWrapper:
        """
        Applies insertion transformation to the model.

        :param wrapped_model: Model to apply transformations.
        :param command: Insertion transformation command.
        """
        device = None
        if not is_multidevice(self._model.model):
            device = get_model_device(self._model.model)

        for command in transformations:
            target_points = command.target_points
            hook_module = command.hook_module
            handle_storage = command.handle_storage

            if device is not None:
                hook_module.to(device)

            for target_point in target_points:
                handle = insert_hook(wrapped_model.model, hook_module, target_point)
                if handle_storage is not None:
                    handle_storage.append(handle)
        return wrapped_model

    @staticmethod
    def _apply_bias_correction_transformations(
        wrapped_model: GraphModelWrapper, transformations: list[PTBiasCorrectionCommand]
    ) -> GraphModelWrapper:
        """
        Applies bias correction transformations on the model.

        :param model: Model to apply transformations.
        :param transformations: List of the bias correction transformations.
        :return: Model with corrected bias.
        """
        for transformation in transformations:
            pt_target_point = cast(PTTargetPoint, transformation.target_point)
            update_fused_bias(
                target_node_name=pt_target_point.target_node_name,
                new_bias=transformation.bias_value,
                nncf_graph=wrapped_model.get_graph(),
                model=wrapped_model.model,
            )
        return wrapped_model

    @staticmethod
    def _apply_const_update_transformations(
        wrapped_model: GraphModelWrapper, transformations: list[PT2ConstUpdateCommand]
    ) -> GraphModelWrapper:
        """
        Applies const data update transformations on the model.

        :param model: Model to apply transformations.
        :param transformations: List of the const data update transformations.
        :return: Model with corrected bias.
        """
        for transformation in transformations:
            node = transformation.node
            value = transformation.value
            set_const_data(value, node, wrapped_model.model)

        return wrapped_model


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
