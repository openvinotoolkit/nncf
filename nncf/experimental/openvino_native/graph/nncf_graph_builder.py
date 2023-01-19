"""
 Copyright (c) 2023 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from typing import List, Optional, Type
import openvino.runtime as ov

from nncf.common.graph import BaseLayerAttributes
from nncf.common.graph import NNCFGraph
from nncf.common.graph.layer_attributes import Dtype
from nncf.common.graph.operator_metatypes import OperatorMetatype

from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OV_OPERATOR_METATYPES
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import GENERAL_WEIGHT_LAYER_METATYPES
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVConvolutionBackpropDataMetatype


class GraphConverter:
    """
    Builds the NNCFGraph from an OpenVINO model.
    """

    @staticmethod
    def convert_to_nncf_dtype(ov_dtype: str) -> Dtype:
        """
        Converts the primitive types from the OpenVINO domain to the NNCF domain.

        :param ov_dtype: OpenVINO primitive typename.
        :return: NNCF primitive type.
        """
        conversion_map = {
            'f16': 'float',
            'f32': 'float',
            'f64': 'float',
            'i4': 'int',
            'i8': 'int',
            'i32': 'int',
            'i64': 'int',
            'u1': 'int',
            'u4': 'int',
            'u8': 'int',
            'u32': 'int',
            'u64': 'int',
        }
        if ov_dtype not in conversion_map:
            raise NotImplementedError(f'NNCF is not yet supported OpenVINO data type: {ov_dtype}.')
        return Dtype(conversion_map[ov_dtype])

    @staticmethod
    def _filter_weight_input_ports(inputs: List[ov.Input], metatype: Type[OperatorMetatype]) -> List[ov.Input]:
        """
        Specifies the possible weight ports of the OpenVINO node.

        :param inputs: OpenVINO node inputs.
        :param metatype: NNCF meta type which corresponds to operation.
        :return: OpenVINO node inputs that may contain weights.
        """
        if metatype == OVConvolutionBackpropDataMetatype:
            return inputs[:2]
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
                for inp in out.get_target_inputs():
                    out_node = inp.get_node()
                    tensor_shape = list(out.shape)
                    output_node_id = graph.get_node_by_name(out_node.get_friendly_name()).node_id
                    ov_dtype = out.get_element_type().get_type_name()
                    nncf_dtype = GraphConverter.convert_to_nncf_dtype(ov_dtype)
                    graph.add_edge_between_nncf_nodes(
                        from_node_id=in_node_id,
                        to_node_id=output_node_id,
                        tensor_shape=tensor_shape,
                        input_port_id=inp.get_index(),
                        output_port_id=output_port_id,
                        dtype=Dtype(nncf_dtype)
                    )

    @staticmethod
    def _add_nncf_node(node: ov.Node, graph: NNCFGraph) -> None:
        """
        Creates NNCFNode from OpenVINO node and adds to the NNCFGraph.

        :param node: OpenVINO node.
        :param graph: NNCFGraph.
        """
        node_type = node.get_type_name()
        metatype = OV_OPERATOR_METATYPES.get_operator_metatype_by_op_name(node_type)
        graph.add_nncf_node(node_name=node.get_friendly_name(),
                            node_type=node_type,
                            node_metatype=metatype)

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
        inference_nodes = model.get_parameters()

        while inference_nodes:
            node = inference_nodes[0]
            inference_nodes = inference_nodes[1:]
            if node.get_friendly_name() not in visited:
                GraphConverter._add_nncf_node(node, nncf_graph)
                visited.add(node.get_friendly_name())
                for out in node.outputs():
                    for inp in sorted(out.get_target_inputs(),
                                      key=lambda inp: inp.get_node().get_friendly_name()):
                        inference_nodes.append(inp.get_node())

        for node in model.get_ops():
            node_type = node.get_type_name()
            metatype = OV_OPERATOR_METATYPES.get_operator_metatype_by_op_name(node_type)
            # Add nodes from constant subgraphs
            if node.get_friendly_name() not in visited:
                GraphConverter._add_nncf_node(node, nncf_graph)
            # Set weight port id
            elif metatype in GENERAL_WEIGHT_LAYER_METATYPES:
                for inp in GraphConverter._filter_weight_input_ports(node.inputs(), metatype):
                    inp_name = inp.get_source_output().get_node().get_friendly_name()
                    if inp_name not in visited:
                        nncf_node = nncf_graph.get_node_by_name(node.get_friendly_name())
                        nncf_node.layer_attributes = OVWeightedLayerAttributes(weight_port_id=inp.get_index())
                        break

        GraphConverter._add_edges_to_nncf_graph(model, nncf_graph)
        return nncf_graph


class OVWeightedLayerAttributes(BaseLayerAttributes):
    """
    This class stores weight and bias port indices of layers for the algorithms.
    """

    def __init__(self, weight_port_id: Optional[int] = None):
        """
        :param weight_port_id: Index of weight port. Should be None if layer without weights.
        """
        self.weight_port_id = weight_port_id
