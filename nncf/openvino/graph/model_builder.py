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
from collections import deque
from typing import Dict, List, Tuple

import openvino.runtime as ov
from openvino.runtime import opset13 as opset
from openvino.runtime.utils.node_factory import NodeFactory

from nncf.openvino.graph.model_transformer import OVModelTransformer
from nncf.openvino.graph.node_utils import get_parameter_node_name
from nncf.openvino.graph.node_utils import get_result_node_name


class OVModelBuilder:
    """
    The purpose of the ModelBuilder is to build a new OpenVINO model from input and output points.
    This Builder was created to reduce the number of model cloning that is required for ModelTransformer to work.
    """

    def __init__(self):
        self._node_factory = NodeFactory()

    @staticmethod
    def _create_parameter(node_name: str, node_input: ov.Input) -> ov.Node:
        """
        A method that contains steps to create a Parameter for a new model using a specific template.
        """
        port_id = node_input.get_index()
        parameter_name = get_parameter_node_name(node_name, port_id)
        return opset.parameter(
            shape=node_input.get_partial_shape(),
            dtype=node_input.get_element_type(),
            name=parameter_name,
        )

    @staticmethod
    def _create_result(node_name: str, node_output: ov.Input) -> ov.Node:
        """
        A method that contains steps to create a Result for a new model using a specific template.
        """
        port_id = node_output.get_index()
        result_name = get_result_node_name(node_name, port_id=port_id)
        result = opset.result(node_output, name=result_name)
        result.get_output_tensor(0).set_names({result_name})
        return result

    def _collect_graph_nodes(
        self,
        input_ids: List[Tuple[str, int]],
        output_ids: List[Tuple[str, int]],
        node_mapping: Dict[str, ov.Node],
    ) -> List[ov.Node]:
        """
        A method for aggregating layers to be further cloned.
        Aggregation is designed in such a way that layers are listed from right to left,
        as they pass from bottom to top. This is done in order to find all constants in the model and
        to start graph creation from them (as well as Parameter layers), because
        OpenVINO graph is created from top-down and cannot be created otherwise.

        Legend: w - weigths, c - convert, il/ih - input low/high, ol/oh - output low/high
        (w)
         |
        (c) (il) (ih) (ol) (oh)
          \   |   |   /   /
           (fake quantize) (parameter)
                  \           /
                  (convolution)
                        |
                    (result)
        Based on the above graph, the return value would look like this:
        [convolution, parameter, fake quantize, oh, ol, ih, il, c, w]

        :param input_ids: List of the points in the special format - (node_name, port_id).
            This helps to point to the precise part of the model that may be used to define the subgraph inputs.
        :param output_ids: List of the points in the special format - (node_name, port_id).
            This helps to point to the precise part of the model that may be used to define the subgraph outputs.
        :param node_mapping: Original nodes mapping.
        :return: List of the ov.Nodes to clone.
        """
        # Creating a list as a deque for FIFO layer acquisition and retrieval
        lookup_nodes = deque(node_mapping[n] for n, _ in output_ids)
        graph_nodes = []

        while lookup_nodes:
            lookup_node = lookup_nodes.popleft()
            lookup_name = lookup_node.get_friendly_name()
            node_inputs = lookup_node.inputs()
            graph_nodes.append(lookup_node)
            # Reversing to lookup nodes from right to left
            for node_input in reversed(node_inputs):
                port_id = node_input.get_index()
                if (lookup_name, port_id) in input_ids:
                    # We create Parameters here to avoid double creation in the future since it is not an original node,
                    # but we need to have it as input for next node.
                    parameter = self._create_parameter(lookup_name, node_input)
                    lookup_nodes.append(parameter)
                    continue
                parent_node = node_input.get_source_output().get_node()
                lookup_nodes.append(parent_node)

        return graph_nodes

    def build(
        self,
        input_ids: List[Tuple[str, int]],
        output_ids: List[Tuple[str, int]],
        node_mapping: Dict[str, ov.Node],
    ) -> ov.Model:
        """
        The basic method of the algorithm. This method uses an aggregated list of layers to be recreated.
        Let us take a graph of this kind as an example:

        Legend: w - weigths, c - convert, il/ih - input low/high, ol/oh - output low/high
        (w)
         |
        (c) (il) (ih) (ol) (oh)
          \   |   |   /   /
           (fake quantize) (parameter)
                  \           /
                  (convolution)
                        |
                    (result)

        The externally collected list of layers will look like this:
        [convolution, parameter, fake quantize, oh, ol, ih, il, c, w]

        Next, this list will be circled from right to left. At the same time, the list of already created layers
        will be filled from left to right, which will be used in the traversal step also, from left to right,
        in order to keep the order of the original layer inputs.
        For example:

            graph_nodes = [convolution, parameter, fake quantize, oh, ol, ih, il, c, w]
            clone_nodes = []

        *creating w - weight node.*
            graph_nodes = [convolution, parameter, fake quantize, oh, ol, ih, il, c]
            clone_nodes = [w]

        *creating c - convert node.
        Based on the .inputs() output, we'll use the already created w-weight node to fill in the convert input.
        As the result, weight node would be removed from the clone_nodes list and convert node would be placed here.*
            graph_nodes = [convolution, parameter, fake quantize, oh, ol, ih, il]
            clone_nodes = [c]

        *creating il/ih - input low/high, ol/oh - output low/high nodes.
        Since these nodes are constants and do not require any nodes as inputs, cloned nodes will not be used.*
            graph_nodes = [convolution, parameter, fake quantize, oh, ol, ih, il]
            clone_nodes = [c, il, ih, ol, oh]

        *creating fake quantize node.
        This node requires to have input values in a specific order.
        All previous nodes will be connected/used for fake quantize, from left to right.*
            graph_nodes = [convolution, parameter]
            clone_nodes = [f]

        *creating parameter node.
        In this step, the list of parameters will also be filled out with the new node.*
            graph_nodes = [convolution]
            clone_nodes = [f, parameter]

        *creating convolution node.
        This node also requires to have inputs in a specific order.
        All previous nodes will be connected/used for convolution, from left to right. Also,
        the outputs verification step will show here that one of the convolution outputs is in the output_ids list.
        This means that the Result node would be created and placed into the results list.*
            graph_nodes = []
            clone_nodes = [convolution]

        The last step is to create a subgraph model based on the parameters & results lists.

        :param input_ids: List of the points in the special format - (node_name, port_id).
            This helps to point to the precise part of the model that may be used to define the subgraph inputs.
        :param output_ids: List of the points in the special format - (node_name, port_id).
            This helps to point to the precise part of the model that may be used to define the subgraph outputs.
        :param node_mapping: Original nodes mapping.
        :return: Builded ov.Model based on parameters.
        """

        parameters, results = [], []
        clone_nodes = deque()

        # Collecting nodes that declares the graph.
        graph_nodes = self._collect_graph_nodes(input_ids, output_ids, node_mapping)

        while graph_nodes:
            graph_node = graph_nodes.pop()
            node_type = graph_node.get_type_name()
            node_name = graph_node.get_friendly_name()

            # To create the new OpenVINO nodes, we need to provide all possible layer attributes.
            attrs = graph_node.get_attributes()
            attrs["name"] = node_name

            if node_type == "Constant":
                # Constants creation is apart due to specific behavior.
                clone_node = OVModelTransformer._create_constant(
                    graph_node.get_data(), dtype=graph_node.get_element_type(), name=attrs["name"]
                )
            elif node_type == "Parameter":
                # We've created Parameter nodes on the previous step.
                clone_node = graph_node
                parameters.append(clone_node)
            else:
                # We have to have args as the inputs since all of them are nodes and are required to be as input.
                args = [clone_nodes.popleft() for _ in graph_node.inputs()]

                clone_node = self._node_factory.create(node_type, args, attrs)

                for node_output in clone_node.outputs():
                    port_id = node_output.get_index()
                    if (node_name, port_id) in output_ids:
                        result = self._create_result(node_name, node_output)
                        results.append(result)

            clone_nodes.append(clone_node)

        return ov.Model(results, parameters)
