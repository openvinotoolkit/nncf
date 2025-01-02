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
from typing import List, Set

import torch
import torch.fx

from nncf.common.graph.model_transformer import ModelTransformer
from nncf.experimental.torch.fx.commands import FXApplyTransformationCommand
from nncf.experimental.torch.fx.node_utils import get_graph_node_by_name
from nncf.torch.graph.transformations.commands import PTModelExtractionCommand
from nncf.torch.graph.transformations.layout import PTTransformationLayout


class FXModelTransformer(ModelTransformer):
    """
    Applies transformations upon Torch FX model.
    FXApplyTransformationCommands are made inplace,
    PTModelExtractionCommands do not change the input model.
    """

    def __init__(self, model: torch.fx.GraphModule):
        super().__init__(model)

        self._command_transformation_ordered_pairs = [
            (FXApplyTransformationCommand, self._apply_transformation),
            (PTModelExtractionCommand, self._apply_model_extraction),
        ]

    def transform(self, transformation_layout: PTTransformationLayout) -> torch.fx.GraphModule:
        """
        Transforms the target model according to given transformation layout.

        :param transformation_layout: Given transformation layout.
        :return: Target model transformered according to the given transformation layout.
        """
        # TODO(dlyakhov): Manage priorities of transformations.
        transformations = transformation_layout.transformations
        aggregated_transformations = defaultdict(list)
        for transformation in transformations:
            aggregated_transformations[transformation.__class__].append(transformation)

        model = self._model
        for transformation_cls, transformation_fn in self._command_transformation_ordered_pairs:
            transformations = aggregated_transformations[transformation_cls]
            if transformations:
                model = transformation_fn(model, transformations)

        # Do not use model.graph.eliminate_dead_code()
        # because the computational statistics code
        # is interpolated as dead code.
        model.recompile()
        return model

    @staticmethod
    def _traverse_graph(
        input_nodes: List[torch.fx.Node],
        stop_nodes: Set[torch.fx.Node],
        visited: Set[torch.fx.Node],
    ) -> None:
        """
        Traverses through the graph starting with the input nodes and
        stopping for the stop nodes and the visited nodes. As the result,
        it modifies the visited container with all nodes visited during the traverse.

        :param input_nodes: Given input nodes.
        :param stop_nodes: Given stop nodes.
        :param visited: Set of already visited nodes.
        """

        while input_nodes:
            in_node = input_nodes.pop()
            if in_node.name in visited or in_node.name in stop_nodes:
                continue

            visited.add(in_node.name)
            # Any constant is a stop op during the traversing procedure.
            if in_node.op == "get_attr":
                continue
            input_nodes.extend(in_node.all_input_nodes)
            input_nodes.extend(list(in_node.users))

    @staticmethod
    def _apply_model_extraction(
        model: torch.fx.GraphModule,
        transformations: List[PTModelExtractionCommand],
    ) -> torch.fx.GraphModule:
        """
        Returns a submodel extracted from the given model by the given transformation.

        :param model: Given model.
        :param transformations: List of one transformation which specifies
            how to retrieve a submodule from the model. In case list contains
            more than one element this function raises an assert.
        :return: Returns a submodel extracted from the given model by the given transformation.
        """

        transformation = transformations[-1]
        stop_nodes = set(transformation.input_node_names + transformation.output_node_names)
        visited = set()

        for node_name in transformation.input_node_names:
            node = get_graph_node_by_name(model.graph, node_name)
            visited.add(node.name)
            target_inputs = node.all_input_nodes[1:]
            if node.name not in transformation.output_node_names:
                target_inputs += list(node.users)
            FXModelTransformer._traverse_graph(target_inputs, stop_nodes, visited)

        for node_name in transformation.output_node_names:
            node = get_graph_node_by_name(model.graph, node_name)
            visited.add(node.name)
            if node.name not in transformation.input_node_names:
                FXModelTransformer._traverse_graph(node.all_input_nodes, stop_nodes, visited)

        extracted_graph = torch.fx.Graph()
        value_remap = {}

        def remap_fn(node: torch.fx.Node):
            return value_remap.get(node)  # noqa F821

        visited_outputs_names = []
        for node in model.graph.nodes:
            if node.name not in visited:
                continue
            if node.op == "output":
                visited_outputs_names.append(node.name)
                continue
            value_remap[node] = extracted_graph.node_copy(node, remap_fn)

        for input_name in transformation.input_node_names:
            node_with_input = get_graph_node_by_name(extracted_graph, input_name)
            with extracted_graph.inserting_before(node_with_input):
                graph_input_name = input_name + "_input"
                graph_input = extracted_graph.create_node(
                    op="placeholder",
                    target=graph_input_name,
                    name=graph_input_name,
                )

            args = list(node_with_input.args)
            args[0] = graph_input
            node_with_input.args = tuple(args)

        # Merge new output with the original output in case
        # the original output is requested in the extracted graph.
        nodes_with_output = []
        for name in transformation.output_node_names:
            nodes_with_output.append(
                name if name in visited_outputs_names else get_graph_node_by_name(extracted_graph, name)
            )

        for idx, node in enumerate(nodes_with_output):
            if isinstance(node, torch.fx.Node):
                continue
            # Current node is the original graph output.
            # Should be replaced by its arguments.
            output_node = get_graph_node_by_name(model.graph, node)
            args = output_node.args[0]
            if isinstance(args, torch.fx.Node):
                # Case of non tuple output.
                args = value_remap[args]
            else:
                # Case of tuple output.
                args = [value_remap[n] for n in args]
                # Unpack target output args in case
                # the only one arg is presented.
                if len(args) == 1:
                    args = args[0]
            nodes_with_output[idx] = args

        last_node = list(extracted_graph.nodes)[-1]
        with extracted_graph.inserting_after(last_node):
            graph_output_name = "output"
            extracted_graph.create_node(
                "output",
                graph_output_name,
                (tuple(nodes_with_output),),
                name=graph_output_name,
            )

        return torch.fx.GraphModule(model, extracted_graph)

    @staticmethod
    def _apply_transformation(
        model: torch.fx.GraphModule,
        transformations: List[FXApplyTransformationCommand],
    ) -> torch.fx.GraphModule:
        """
        Applies transformations to the given model.

        :param model: Target model.
        :param transformations: Transformations to apply to the model.
        :return: Target model after all transformations were applied.
        """
        for transformation in transformations:
            transformation.transformation_fn(model)
        return model
