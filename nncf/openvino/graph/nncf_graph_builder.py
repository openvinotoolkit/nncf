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
from typing import List, Type

import openvino.runtime as ov

from nncf.common.graph import NNCFGraph
from nncf.common.graph.layer_attributes import Dtype
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.openvino.graph.layer_attributes import OVLayerAttributes
from nncf.openvino.graph.metatypes.groups import OPERATIONS_WITH_CONST_PORT_ID
from nncf.openvino.graph.metatypes.openvino_metatypes import OVConvolutionBackpropDataMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVGroupConvolutionBackpropDataMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVGRUSequenceMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVLSTMSequenceMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVMatMulMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import get_node_metatype
from nncf.openvino.graph.metatypes.openvino_metatypes import get_operation_const_op
from nncf.openvino.graph.node_utils import get_weighted_layer_attributes


class GraphConverter:
    """
    Builds the NNCFGraph from an OpenVINO model.
    """

    @staticmethod
    def convert_to_nncf_dtype(ov_type: ov.Type) -> Dtype:
        """
        Converts the primitive types from the OpenVINO domain to the NNCF domain.

        :param ov_dtype: OpenVINO primitive typename.
        :return: NNCF primitive type.
        """
        type_name = ov_type.get_type_name()
        conversion_map = {
            "f16": "float",
            "bf16": "float",
            "f32": "float",
            "f64": "float",
            "i4": "int",
            "i8": "int",
            "i16": "int",
            "i32": "int",
            "i64": "int",
            "u1": "int",
            "u4": "int",
            "u8": "int",
            "u16": "int",
            "u32": "int",
            "u64": "int",
            "boolean": "int",
            "string": "int",
        }
        if type_name not in conversion_map:
            raise NotImplementedError(f"NNCF is not yet supported OpenVINO data type: {type_name}.")
        return Dtype(conversion_map[type_name])

    @staticmethod
    def _filter_weight_input_ports(inputs: List[ov.Input], metatype: Type[OperatorMetatype]) -> List[ov.Input]:
        """
        Specifies the possible weight ports of the OpenVINO node.

        :param inputs: OpenVINO node inputs.
        :param metatype: NNCF meta type which corresponds to operation.
        :return: OpenVINO node inputs that may contain weights.
        """
        if metatype in [OVConvolutionBackpropDataMetatype, OVGroupConvolutionBackpropDataMetatype]:
            return inputs[:2]
        if metatype == OVGRUSequenceMetatype:
            return inputs[:5]
        if metatype == OVLSTMSequenceMetatype:
            return inputs[:6]
        return inputs

    @staticmethod
    def _add_edges_to_nncf_graph(model: ov.Model, graph: NNCFGraph) -> None:
        """
        Adds edges between NNCFNodes to the NNCFGraph.

        :param model: OpenVINO model.
        :param graph: NNCFGraph.
        """
        for op in model.get_ops():
            in_node_id = graph.get_node_by_name(op.get_friendly_name()).node_id
            for output_port_id, out in enumerate(op.outputs()):
                node_vs_target_inputs = defaultdict(list)
                for inp in out.get_target_inputs():
                    node_vs_target_inputs[inp.get_node()].append(inp)

                for out_node, inputs in node_vs_target_inputs.items():
                    tensor_shape = list(out.partial_shape.get_max_shape())
                    output_node_id = graph.get_node_by_name(out_node.get_friendly_name()).node_id
                    nncf_dtype = GraphConverter.convert_to_nncf_dtype(out.get_element_type())

                    parallel_inputs = None
                    if len(inputs) > 1:
                        parallel_inputs = [inp.get_index() for inp in inputs[1:]]

                    graph.add_edge_between_nncf_nodes(
                        from_node_id=in_node_id,
                        to_node_id=output_node_id,
                        tensor_shape=tensor_shape,
                        input_port_id=inputs[0].get_index(),
                        output_port_id=output_port_id,
                        dtype=nncf_dtype,
                        parallel_input_port_ids=parallel_inputs,
                    )

    @staticmethod
    def _add_nncf_node(node: ov.Node, graph: NNCFGraph) -> None:
        """
        Creates NNCFNode from OpenVINO node and adds to the NNCFGraph.

        :param node: OpenVINO node.
        :param graph: NNCFGraph.
        """
        node_type = node.get_type_name()
        metatype = get_node_metatype(node)
        ignored_algorithms = GraphConverter._get_ignored_algorithms(node)
        graph.add_nncf_node(
            node_name=node.get_friendly_name(),
            node_type=node_type,
            node_metatype=metatype,
            ignored_algorithms=ignored_algorithms,
        )

    @staticmethod
    def _get_ignored_algorithms(node: ov.Node) -> List[str]:
        """
        Creates a list of the ignored algorithms corresponding with
        the ignored_algorithms option of add_nncf_node method.

        :param node: OpenVINO node.
        :return: List of the ignored algorithms.
        """
        ignored_algorithms = []
        if "nncf_smooth_quant" in node.get_friendly_name():
            ignored_algorithms.append("ptq_quantization")
        return ignored_algorithms

    @staticmethod
    def create_nncf_graph(model: ov.Model) -> NNCFGraph:
        """
        Creates NNCFGraph from OpenVINO Model.
        All nodes from model which have valid metatype are added to NNCFGraph.
        Then, corresponding edges are added to the NNCFGraph with shape, type, output and input port ids.

        :param model: OpenVINO model.
        :return: NNCFGraph.
        """
        nncf_graph = NNCFGraph()
        visited = set()
        read_value_nodes = [op for op in model.get_ops() if op.get_type_name() == "ReadValue"]
        inference_nodes = model.get_parameters() + read_value_nodes

        while inference_nodes:
            node = inference_nodes[0]
            inference_nodes = inference_nodes[1:]
            if node.get_friendly_name() not in visited:
                GraphConverter._add_nncf_node(node, nncf_graph)
                visited.add(node.get_friendly_name())
                for out in node.outputs():
                    for inp in sorted(out.get_target_inputs(), key=lambda inp: inp.get_node().get_friendly_name()):
                        inference_nodes.append(inp.get_node())

        for node in model.get_ops():
            metatype = get_node_metatype(node)
            # Add nodes from constant subgraphs
            node_name = node.get_friendly_name()
            if node_name not in visited:
                GraphConverter._add_nncf_node(node, nncf_graph)
            # Set const port id
            elif metatype in OPERATIONS_WITH_CONST_PORT_ID:
                const_attrs, act_attrs = {}, {}
                for inp in GraphConverter._filter_weight_input_ports(node.inputs(), metatype):
                    inp_name = inp.get_source_output().get_node().get_friendly_name()
                    if inp_name in visited:
                        continue

                    const_port_id = inp.get_index()
                    const_node = get_operation_const_op(node, const_port_id)
                    if const_node is None:
                        continue

                    if GraphConverter.convert_to_nncf_dtype(const_node.get_element_type()) == Dtype.INTEGER:
                        continue

                    const_attrs[const_port_id] = {
                        "name": const_node.get_friendly_name(),
                        "shape": tuple(const_node.get_output_shape(0)),
                        "dtype": const_node.output(0).get_element_type().get_type_name(),
                    }

                    if metatype == OVMatMulMetatype:
                        act_port_id = abs(const_port_id - 1)
                        attribute_names = ["transpose_a", "transpose_b"]
                        node_attributes = node.get_attributes()
                        const_transpose_name = attribute_names[const_port_id]
                        const_attrs[const_port_id]["transpose"] = node_attributes[const_transpose_name]
                        act_attrs["transpose"] = node_attributes[attribute_names[act_port_id]]
                    elif metatype == OVGRUSequenceMetatype:
                        node_attributes = node.get_attributes()
                        act_attrs["linear_before_reset"] = node_attributes["linear_before_reset"]

                    if const_attrs or act_attrs:
                        nncf_node = nncf_graph.get_node_by_name(node_name)
                        layer_attributes = get_weighted_layer_attributes(node, metatype, const_attrs)
                        nncf_node.layer_attributes = OVLayerAttributes(const_attrs, layer_attributes, act_attrs)

        GraphConverter._add_edges_to_nncf_graph(model, nncf_graph)
        return nncf_graph
