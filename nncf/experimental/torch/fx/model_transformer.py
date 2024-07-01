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
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.common.graph.transformations.commands import TransformationType
from nncf.torch.graph.transformations.commands import PTModelExtractionCommand
from nncf.torch.graph.transformations.layout import PTTransformationLayout


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
            (FXApplyTransformationCommand, self._apply_transformation),
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
        # the dead code is computing statistics :)
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
    def get_graph_node_by_name(graph, name):
        for node in graph.nodes:
            if node.name == name:
                return node
        raise RuntimeError(f"Node with name {name} is not found")

    @staticmethod
    def _apply_transformation(
        model: torch.fx.GraphModule,
        transformations: List[FXApplyTransformationCommand],
    ) -> torch.fx.GraphModule:
        for transformation in transformations:
            transformation.tranformation_fn(model)
        return model
