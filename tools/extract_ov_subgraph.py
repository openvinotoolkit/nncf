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

import argparse
import os
import shutil
import xml.etree.ElementTree as ET
from copy import copy
from copy import deepcopy
from pathlib import Path
from pprint import pprint
from typing import Any, Dict

import defusedxml.ElementTree as dET
import networkx as nx


def xml_to_dict(element: ET.Element):
    result = {}
    if element.attrib:
        result["attributes"] = element.attrib
    for child in element:
        child_dict = xml_to_dict(child)
        if child.tag in result:
            if isinstance(result[child.tag], list):
                result[child.tag].append(child_dict)
            else:
                result[child.tag] = [result[child.tag], child_dict]
        else:
            result[child.tag] = child_dict
    if element.text:
        result["text"] = element.text
    if element.tail:
        result["tail"] = element.tail
    return result


def dict_to_xml(data: Any, parent: ET.Element):
    if isinstance(data, dict):
        for tag_name, value in data.items():
            if tag_name == "attributes":
                parent.attrib.update(value)
            elif tag_name == "text":
                parent.text = value
            elif tag_name == "tail":
                parent.tail = value
            elif isinstance(value, list):
                for item in value:
                    elem = ET.SubElement(parent, tag_name)
                    dict_to_xml(item, elem)
            else:
                elem = ET.SubElement(parent, tag_name)
                dict_to_xml(value, elem)
    else:
        parent.text = str(data)


def get_edges(xml_dict: Dict):
    def add_edge(edges: Dict, from_layer: int, from_port: int, to_layer: int, to_port: int):
        if from_layer not in edges:
            edges[from_layer] = {}
        if from_port not in edges[from_layer]:
            edges[from_layer][from_port] = {}
        assert (to_layer, to_port) not in edges[from_layer][from_port]
        edges[from_layer][from_port][(to_layer, to_port)] = {}

    edges = {}
    for edge in xml_dict["edges"]["edge"]:
        edge = edge["attributes"]

        from_layer = int(edge["from-layer"])
        from_port = int(edge["from-port"])
        to_layer = int(edge["to-layer"])
        to_port = int(edge["to-port"])

        add_edge(edges, from_layer, from_port, to_layer, to_port)
        add_edge(edges, to_layer, to_port, from_layer, from_port)

    return edges


def get_nodes(xml_dict: Dict, edges: Dict):
    all_node_names = set()
    nodes = {}
    for node in xml_dict["layers"]["layer"]:
        try:
            attributes = node["attributes"]
            data = node["data"]["attributes"] if "data" in node else None
            inp = node.get("input", None)
            out = node.get("output", None)

            node_id = int(attributes["id"])
            node_name = attributes["name"]
            node_type = attributes["type"]

            assert node_name not in all_node_names
            all_node_names.add(node_name)

            assert node_id not in nodes
            nodes[node_id] = {
                "name": node_name,
                "type": node_type,
            }

            node_dtype = data["element_type"] if data is not None and "element_type" in data else None
            node_shape = data["shape"] if data is not None and "shape" in data else None
            if node_dtype is not None:
                nodes[node_id]["dtype"] = node_dtype
            if node_shape is not None:
                nodes[node_id]["shape"] = node_shape

            input_ports = [] if inp is None else inp["port"]
            output_ports = [] if out is None else out["port"]
            if isinstance(input_ports, dict):
                input_ports = [input_ports]
            if isinstance(output_ports, dict):
                output_ports = [output_ports]

            for port, is_input in zip(
                input_ports + output_ports, [True] * len(input_ports) + [False] * len(output_ports)
            ):
                from_port = int(port["attributes"]["id"])
                precision = port["attributes"]["precision"]
                if "dim" in port["attributes"]:
                    dim = port["attributes"]["dim"]
                elif "dim" in port:
                    dim = port["dim"]
                else:
                    dim = []
                if isinstance(dim, dict):
                    dim = [dim]
                shape = tuple(int(it["text"]) for it in dim)

                # Update properties of the edges leading from this port
                if from_port not in edges[node_id]:
                    # Some edge descriptions may be missing in execution graph
                    continue
                else:
                    edge = edges[node_id][from_port]
                for (to_node_id, to_port), edge_properties_dict in edge.items():
                    for name, value in zip(("precision", "shape", "is_input"), (precision, shape, is_input)):
                        assert name not in edge_properties_dict
                        edge_properties_dict[name] = value
        except Exception as e:
            pprint(node)
            raise e

    return nodes


def create_nx_graph(xml_dict: Dict):
    def get_node_label(nodes: Dict, node_id: int):
        return nodes[node_id]["name"]

    def get_edge_label(edges: Dict, nodes: Dict, from_node: int, from_port: int, to_node: int, to_port: int):
        edge_properties = edges[from_node][from_port][(to_node, to_port)]
        return f'"{edge_properties["shape"]}\n{from_port}->{to_port}"'

    edges = get_edges(xml_dict)
    nodes = get_nodes(xml_dict, edges)

    G = nx.Graph()

    # Add nodes
    for node_id, node_properties in nodes.items():
        node_properties_copy = copy(node_properties)
        node_properties_copy["id"] = node_id
        G.add_node(get_node_label(nodes, node_id), **node_properties_copy)

    # Add edges
    for node_id, from_port_dict in edges.items():
        for from_port, to_port_dict in from_port_dict.items():
            for (to_node_id, to_port), edge_properties in to_port_dict.items():
                G.add_edge(
                    u_of_edge=get_node_label(nodes, node_id),
                    v_of_edge=get_node_label(nodes, to_node_id),
                    label=get_edge_label(edges, nodes, node_id, from_port, to_node_id, to_port),
                    **edge_properties,
                )

    return G


def write_xml(xml_dict: Dict, filepath: Path):
    write_root = ET.Element("net")
    dict_to_xml(xml_dict, write_root)
    xml_str = ET.tostring(write_root).decode()
    xml_str = '<?xml version="1.0"?>\n' + xml_str + "\n"
    with open(filepath, "w") as f:
        f.write(xml_str)


def take_model_subgraph(xml_dict: Dict, source_node_name: str, distance: int):
    # Create networkx graph from IR xml dictionary
    G = create_nx_graph(xml_dict)

    # Traverse graph from target node
    dfs_tree = nx.traversal.dfs_tree(G, source=source_node_name, depth_limit=distance)
    node_names = set(dfs_tree.nodes)
    node_ids = set([G.nodes[it]["id"] for it in node_names])

    # Keep only the visited nodes
    result_xml_dict = deepcopy(xml_dict)
    result_xml_dict["layers"]["layer"] = []
    for layer in xml_dict["layers"]["layer"]:
        node_name = layer["attributes"]["name"]
        if node_name in node_names:
            result_xml_dict["layers"]["layer"].append(layer)

    # Keep only the edges that connect the visited nodes
    result_xml_dict["edges"]["edge"] = []
    for edge in xml_dict["edges"]["edge"]:
        from_layer = int(edge["attributes"]["from-layer"])
        to_layer = int(edge["attributes"]["to-layer"])
        if from_layer in node_ids or to_layer in node_ids:
            result_xml_dict["edges"]["edge"].append(edge)

    return result_xml_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract a subgraph from a model in OpenVINO Intermediate Representation format.\n\nSubgraph is "
        "taken around a given node. Use distance parameter to control how many nodes around the given one to include. "
        "The resulting subgraph is saved next to the input .xml file or at --output_path if provided. Additionally, a "
        "symbolic link targeting the original .bin file is created.",
        epilog="Usage examples:\n"
        '  python ir_subgraph.py openvino.xml "Constant_1116858"\n'
        '  python ir_subgraph.py openvino.xml "Constant_1116858" --distance 5\n'
        '  python ir_subgraph.py openvino.xml "Constant_1116858" --output-path ./subgraphs\n'
        '  python ir_subgraph.py openvino.xml "Constant_1116858" --output-path ./subgraphs/Constant_1116858.xml\n',
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("input-path", help="Input IR path.")
    parser.add_argument("node", help="Target node name.")
    parser.add_argument("--distance", type=int, default=10, help="Distance around the target node (default 10).")
    parser.add_argument(
        "--output-path",
        dest="output_path",
        help="Output IR path. Can either be a file path with .xml extension or a directory path.",
    )

    args = parser.parse_args()

    input_path = Path(args.__dict__["input-path"])
    node_name = args.node
    distance = args.distance
    output_path = Path(args.output_path) if args.output_path is not None else None

    if distance <= 0:
        raise ValueError("Distance should be positive")

    if output_path is None or output_path.suffix == "":
        output_filename = f"{input_path.stem}_{Path(node_name).stem}_{distance}.xml"
        if output_path is None:
            output_dir = input_path.parent
            output_path = input_path.parent / output_filename
        else:
            output_dir = output_path
            output_path = output_dir / output_filename
    else:
        output_dir = output_path.parent

    if output_path.exists():
        raise ValueError(f"There is already and IR at {output_path}. Exiting.")

    # Read IR xml as dict
    tree = dET.parse(input_path)
    root = tree.getroot()
    xml_dict = xml_to_dict(root)

    # Take subgraph
    subgraph_xml_dict = take_model_subgraph(xml_dict, source_node_name=node_name, distance=distance)

    # Save subgraph xml
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    write_xml(subgraph_xml_dict, output_path)

    # Create a symbolic link to original .bin file
    bin_input_path = input_path.with_suffix(".bin")
    bin_output_path = output_path.with_suffix(".bin")
    if bin_output_path.exists():
        os.remove(bin_output_path)
    try:
        bin_output_path.symlink_to(os.path.relpath(bin_input_path, bin_output_path.parent))
    except OSError as e:
        if "[WinError 1314]" in str(e):
            if bin_input_path.exists():
                print("Copying original .bin file because can't create a symbolic link due to lack of admin privileges")
                shutil.copy(bin_input_path, bin_output_path)
            else:
                print("Didn't create a copy of original .bin file because it is missing")
        else:
            raise e

    print("Saved at:", output_path)
