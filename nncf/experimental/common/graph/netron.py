"""
 Copyright (c) 2022 Intel Corporation
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
import xml.etree.ElementTree as ET
from typing import Callable, Dict, List, Optional, Tuple

from nncf.common.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode


class Tags:
    NET = "net"
    NODES = "layers"
    EDGES = "edges"
    NODE = "layer"
    EDGE = "edge"
    DATA = "data"
    INPUT = "input"
    OUTPUT = "output"
    PORT = "port"
    DIM = "dim"


class PortDesc:
    def __init__(self, port_id: str, shape: Optional[List[int]] = None, precision: str = None):
        self.port_id = port_id
        if shape is None:
            shape = []
        self.shape = shape
        self.precision = precision

    def as_xml_element(self) -> ET.Element:
        port = ET.Element(Tags.PORT, id=self.port_id)
        if self.precision:
            port.set("precision", self.precision)
        for i in self.shape:
            dim = ET.Element(Tags.DIM)
            dim.text = str(i)
            port.append(dim)
        return port


class NodeDesc:
    def __init__(
        self,
        node_id: str,
        name: str,
        type: str,
        attrs: Optional[Dict[str, str]] = None,
        inputs: Optional[List[PortDesc]] = None,
        outputs: Optional[List[PortDesc]] = None,
    ):
        self.node_id = node_id
        self.name = name
        self.type = type
        if attrs is None:
            attrs = {}
        self.attrs = attrs
        self.inputs = inputs
        self.outputs = outputs

    def as_xml_element(self) -> ET.Element:
        node = ET.Element(Tags.NODE, id=self.node_id, name=self.name, type=self.type)
        ET.SubElement(node, Tags.DATA, self.attrs)

        if self.inputs:
            input = ET.SubElement(node, Tags.INPUT)
            for port in self.inputs:
                input.append(port.as_xml_element())

        if self.outputs:
            output = ET.SubElement(node, Tags.OUTPUT)
            for port in self.outputs:
                output.append(port.as_xml_element())

        return node


class EdgeDesc:
    def __init__(self, from_node: str, from_port: str, to_node: str, to_port: str):
        self.from_node = from_node
        self.from_port = from_port
        self.to_node = to_node
        self.to_port = to_port

    def as_xml_element(self) -> ET.Element:
        attrs = {
            "from-layer": self.from_node,
            "from-port": self.from_port,
            "to-layer": self.to_node,
            "to-port": self.to_port,
        }
        edge = ET.Element(Tags.EDGE, attrs)
        return edge


GET_ATTRIBUTES_FN_TYPE = Callable[[NNCFNode], Dict[str, str]]


# TODO(andrey-churkin): Add support for `PortDesc.precision` param.
def get_graph_desc(
    graph: NNCFGraph, include_fq_params: bool = False, get_attributes_fn: Optional[GET_ATTRIBUTES_FN_TYPE] = None
) -> Tuple[List[NodeDesc], List[EdgeDesc]]:
    if get_attributes_fn is None:
        get_attributes_fn = lambda x: {
            "metatype": str(x.metatype.name),
        }
    include_node: Dict[int, bool] = {}
    edges = []
    for edge in graph.get_all_edges():
        if not include_fq_params and edge.to_node.node_type == "FakeQuantize" and edge.input_port_id != 0:
            include_node[edge.from_node.node_id] = False
            continue

        edges.append(
            EdgeDesc(
                from_node=str(edge.from_node.node_id),
                from_port=str(edge.output_port_id),
                to_node=str(edge.to_node.node_id),
                to_port=str(edge.input_port_id),
            )
        )

    nodes = []
    for node in graph.get_all_nodes():
        if not include_node.get(node.node_id, True):
            continue

        inputs = []
        for edge in graph.get_input_edges(node):
            if not include_fq_params and node.node_type == "FakeQuantize" and edge.input_port_id != 0:
                continue

            inputs.append(PortDesc(port_id=str(edge.input_port_id), shape=edge.tensor_shape))

        outputs = []
        for edge in graph.get_output_edges(node):
            outputs.append(
                PortDesc(
                    port_id=str(edge.output_port_id),
                    shape=edge.tensor_shape,
                )
            )

        nodes.append(
            NodeDesc(
                node_id=str(node.node_id),
                name=node.node_name,
                type=node.node_type.title(),
                attrs=get_attributes_fn(node),
                inputs=inputs,
                outputs=outputs,
            )
        )

    return nodes, edges


def save_for_netron(
    graph: NNCFGraph,
    save_path: str,
    graph_name: str = "Graph",
    include_fq_params: bool = False,
    get_attributes_fn: Optional[GET_ATTRIBUTES_FN_TYPE] = None,
):
    node_descs, edge_descs = get_graph_desc(graph, include_fq_params, get_attributes_fn)

    net = ET.Element(Tags.NET, name=graph_name)

    nodes = ET.SubElement(net, Tags.NODES)
    for node in node_descs:
        nodes.append(node.as_xml_element())

    edges = ET.SubElement(net, Tags.EDGES)
    for edge in edge_descs:
        edges.append(edge.as_xml_element())

    # ET.indent(net)  # Only Python 3.9
    with open(save_path, "wb") as f:
        f.write(ET.tostring(net))
