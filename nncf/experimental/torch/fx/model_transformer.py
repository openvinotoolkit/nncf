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

from collections import defaultdict

# from functools import partial
from typing import Callable, List, Union

import torch
import torch.fx
from torch.fx.passes.split_utils import split_by_tags

from nncf.common.graph.model_transformer import ModelTransformer
from nncf.common.graph.transformations.commands import Command
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.common.graph.transformations.commands import TransformationType
from nncf.torch.graph.transformations.commands import PTModelExtractionCommand
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.graph.transformations.layout import PTTransformationLayout


class FXModuleInsertionCommand(Command):
    def __init__(
        self,
        target_points: List[PTTargetPoint],
        module_to_insert: torch.nn.Module,
        priority: Union[TransformationPriority, int] = TransformationPriority.DEFAULT_PRIORITY,
    ):
        super().__init__(TransformationType.INSERT)
        self.target_points = target_points
        self.module_to_insert = module_to_insert
        self.priority = priority


class FXApplyTransformationCommand(Command):
    def __init__(
        self,
        transformation_fn: Callable[[torch.fx.GraphModule], None],
        priority: Union[TransformationPriority, int] = TransformationPriority.DEFAULT_PRIORITY,
    ):
        super().__init__(TransformationType.INSERT)
        self.tranformation_fn = transformation_fn
        self.priority = priority


class FXModelTransformer(ModelTransformer):
    """
    Applies transformations upon Torch FX model.
    """

    # TODO: manage priorities of transformations

    def __init__(self, model: torch.fx.GraphModule):
        super().__init__(model)

        self._command_transformation_ordered_pairs = [
            # TODO: Move the module insertion command to a transformation
            (FXApplyTransformationCommand, self._apply_transformation),
            (FXModuleInsertionCommand, self._apply_module_insertion),
            (PTModelExtractionCommand, self._apply_model_extraction),
        ]

    def transform(self, transformation_layout: PTTransformationLayout) -> torch.fx.GraphModule:
        transformations = transformation_layout.transformations
        aggregated_transformations = defaultdict(list)
        for transformation in transformations:
            aggregated_transformations[transformation.__class__].append(transformation)

        model = self._model
        for transformation_cls, transformation_fn in self._command_transformation_ordered_pairs:
            transformations = aggregated_transformations[transformation_cls]
            if transformations:
                model = transformation_fn(model, transformations)

        # Do not eliminate dead code as
        # the dead code is coputing statistics :)
        # model.graph.eliminate_dead_code()
        model.recompile()
        return model

    @staticmethod
    def _apply_model_extraction(
        model: torch.fx.GraphModule,
        transformations: List[PTModelExtractionCommand],
    ) -> torch.fx.GraphModule:
        transformation = transformations[-1]
        assert len(transformation.input_node_names) == 1
        assert transformation.input_node_names == transformation.output_node_names
        node_name = transformation.input_node_names[0]

        tags = ["before", "extracted", "after"]
        i = 0
        for node in model.graph.nodes:
            if node.name == node_name:
                node.tag = tags[1]
                weights = [node.all_input_nodes[1]]
                while weights:
                    w_node = weights.pop()
                    assert w_node.tag in tags[0:2]
                    w_node.tag = tags[1]
                    weights.extend(w_node.all_input_nodes)
                i = 2
                continue
            node.tag = tags[i]

        splitted_gm = split_by_tags(model, tags)
        return splitted_gm.extracted

    @staticmethod
    def _apply_module_insertion(
        model: torch.fx.GraphModule,
        transformations: List[FXModuleInsertionCommand],
    ) -> torch.fx.GraphModule:
        """
        Applies insertion of PTSharedFnInsertionCommand commands. For each command method inserts
        a torch module to the torch.fx.GraphModule and inserts call hooks for each command target points.

        :param model: Model to apply transformations.
        :param transformations: List of the bias correction transformations.
        :param device: Target device for the insertion functions. Applies only to
            functions which are subclassed from torch.nn.Module. Do nothing in case device is None.
        :return: A modified torch.fx.GraphModule.
        """
        for transformation in transformations:
            # Set fn to the model as an attribute
            module_to_insert = transformation.module_to_insert
            module_name_in_model = (
                ";".join(
                    "_".join((tp.target_node_name, str(tp.input_port_id), str(tp.target_type.value)))
                    for tp in transformation.target_points
                )
                + "_"
                + str(id(module_to_insert))
            )
            assert not hasattr(model, module_name_in_model)
            setattr(model, module_name_in_model, module_to_insert)
            # Insert call_module nodes to the model
            for target_point in transformation.target_points:
                FXModelTransformer._create_call_module_node(model.graph, target_point, module_name_in_model)
        return model

    @staticmethod
    def get_graph_node_by_name(graph, name):
        for node in graph.nodes:
            if node.name == name:
                return node
        raise RuntimeError(f"Node with name {name} is not found")

    @staticmethod
    def _get_target_node(graph: torch.fx.Graph, target_point: PTTargetPoint):
        target_type = target_point.target_type
        target_node = FXModelTransformer.get_graph_node_by_name(graph, target_point.target_node_name)
        if target_type in [TargetType.OPERATOR_PRE_HOOK, TargetType.OPERATION_WITH_WEIGHTS]:
            target_node = target_node.all_input_nodes[target_point.input_port_id]
        elif target_type == TargetType.OPERATOR_POST_HOOK:
            pass
        else:
            raise RuntimeError(f"Unsupported target type: {target_type} for target_point: {target_point}")
        return target_node

    @staticmethod
    def _create_call_module_node(graph: torch.fx.Graph, target_point: PTTargetPoint, module_name: str):
        target_node = FXModelTransformer._get_target_node(graph, target_point)
        with graph.inserting_after(target_node):
            graph.create_node("call_module", module_name, (target_node,), {}, name=module_name + "_graph_node")

    @staticmethod
    def _apply_transformation(
        model: torch.fx.GraphModule,
        transformations: List[FXApplyTransformationCommand],
    ) -> torch.fx.GraphModule:
        for transformation in transformations:
            transformation.tranformation_fn(model)
        return model
