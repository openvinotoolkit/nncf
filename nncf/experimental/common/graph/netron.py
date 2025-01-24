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

# Since we are not reading XML, but creating it, the package security message is irrelevant
import xml.etree.ElementTree as ET  # nosec
from typing import Callable, Dict, List, Optional, Tuple

from nncf.common.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.layer_attributes import Dtype


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
    """
    Represents a port description in the computational graph.
    """

    def __init__(self, port_id: str, precision: str, shape: Optional[List[int]] = None):
        """
        :param port_id: The identifier of the port.
        :param precision: Precision of the tensor corresponding to the port, either "fp32" or "i32".
        :param shape: The shape of the tensor. Defaults to an empty list if not provided.
        """
        self.port_id = port_id
        if shape is None:
            shape = []
        self.shape = shape
        self.precision = precision

    def as_xml_element(self) -> ET.Element:
        """
        Converts the PortDesc object into an XML element.

        :return: The Element representing the port in XML
        """
        port = ET.Element(Tags.PORT, id=self.port_id, precision=self.precision)

        for i in self.shape:
            dim = ET.Element(Tags.DIM)
            dim.text = str(i)
            port.append(dim)
        return port


class NodeDesc:
    """
    Represents a node description in the computational graph.
    """

    def __init__(
        self,
        node_id: str,
        name: str,
        node_type: str,
        attrs: Optional[Dict[str, str]] = None,
        inputs: Optional[List[PortDesc]] = None,
        outputs: Optional[List[PortDesc]] = None,
    ):
        """
        :param node_id: The identifier of the node.
        :param name: The name of the node.
        :param type: The type of the node.
        :param attrs: Additional attributes of the node. Default empty dictionary.
        :param inputs: List of input ports for the node.
        :param outputs: List of output ports for the node.
        """
        self.node_id = node_id
        self.name = name
        self.type = node_type
        if attrs is None:
            attrs = {}
        self.attrs = attrs
        self.inputs = inputs
        self.outputs = outputs

    def as_xml_element(self) -> ET.Element:
        """
        Converts the NodeDesc object into an XML element.

        :return: The Element representing the node in XML
        """
        node = ET.Element(Tags.NODE, id=self.node_id, name=self.name, type=self.type)
        ET.SubElement(node, Tags.DATA, self.attrs)

        if self.inputs:
            input_ = ET.SubElement(node, Tags.INPUT)
            for port in self.inputs:
                input_.append(port.as_xml_element())

        if self.outputs:
            output = ET.SubElement(node, Tags.OUTPUT)
            for port in self.outputs:
                output.append(port.as_xml_element())

        return node


class EdgeDesc:
    """
    Represents an edge description in the computational graph.
    """

    def __init__(self, from_node: str, from_port: str, to_node: str, to_port: str):
        """
        :param from_node: The identifier of the source node.
        :param from_port: The identifier of the output port of the source node.
        :param to_node: The identifier of the target node.
        :param to_port: The identifier of the input port of the target node.
        """
        self.from_node = from_node
        self.from_port = from_port
        self.to_node = to_node
        self.to_port = to_port

    def as_xml_element(self) -> ET.Element:
        """
        Converts the EdgeDesc object into an XML element.

        :return: The Element representing the edge in XML
        """
        attrs = {
            "from-layer": self.from_node,
            "from-port": self.from_port,
            "to-layer": self.to_node,
            "to-port": self.to_port,
        }
        edge = ET.Element(Tags.EDGE, attrs)
        return edge


GET_ATTRIBUTES_FN_TYPE = Callable[[NNCFNode], Dict[str, str]]


def convert_nncf_dtype_to_ov_dtype(dtype: Dtype) -> str:
    """
    Converts a nncf dtype to an openvino dtype string.

    :param dtype: The data type to be converted.
    :return: The openvino dtype string corresponding to the given data type.
    """

    dummy_precision_map: Dict[Dtype, str] = {Dtype.INTEGER: "i32", Dtype.FLOAT: "f32"}

    return dummy_precision_map[dtype]


def get_graph_desc(
    graph: NNCFGraph, include_fq_params: bool = False, get_attributes_fn: Optional[GET_ATTRIBUTES_FN_TYPE] = None
) -> Tuple[List[NodeDesc], List[EdgeDesc]]:
    """
    Retrieves descriptions of nodes and edges from an NNCFGraph.

    :param graph: The NNCFGraph instance to extract descriptions from.
    :param include_fq_params: Whether to include FakeQuantize parameters in the description.
    :param get_attributes_fn: A function to retrieve additional attributes for nodes.
        Defaults to a function returning {"metatype": str(x.metatype.name)}.
    :return: A tuple containing lists of NodeDesc and EdgeDesc objects
        representing the nodes and edges of the NNCFGraph.
    """

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
            inputs.append(
                PortDesc(
                    port_id=str(edge.input_port_id),
                    precision=convert_nncf_dtype_to_ov_dtype(edge.dtype),
                    shape=edge.tensor_shape,
                )
            )

        outputs = []
        for edge in graph.get_output_edges(node):
            outputs.append(
                PortDesc(
                    port_id=str(edge.output_port_id),
                    precision=convert_nncf_dtype_to_ov_dtype(edge.dtype),
                    shape=edge.tensor_shape,
                )
            )

        nodes.append(
            NodeDesc(
                node_id=str(node.node_id),
                name=node.node_name,
                node_type=node.node_type.title(),
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
    """
    Save the NNCFGraph information in an XML file suitable for visualization with Netron.

    :param graph: The NNCFGraph instance to visualize.
    :param save_path: The path to save the Netron-compatible file.
    :param graph_name: The name of the graph. Defaults to "Graph".
    :param include_fq_params: Whether to include FakeQuantize parameters in the visualization.
    :param get_attributes_fn: A function to retrieve additional attributes for nodes.
        Defaults to a function returning {"metatype": str(x.metatype.name)}.
    """

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
