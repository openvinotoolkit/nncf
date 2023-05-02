# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import abstractmethod
from argparse import ArgumentParser
from collections import defaultdict
from typing import Any, Dict, List

import openvino.runtime as ov


class ATTRS:
    class NodeAttr:
        def __init__(self) -> None:
            self.container = defaultdict(list)

        @property
        def name(self):
            return self.__class__.__name__

        @abstractmethod
        def get_attr_from_node(self, node: ov.Node, info: Dict[str, Any]):
            pass

        def register_node(self, node: ov.Node, info: Dict[str, Any]):
            attr = self.get_attr_from_node(node, info)
            self.container[attr].append(node.get_friendly_name())

        def get_stat(self, node: ov.Node, info: Dict[str, Any]):
            attr = self.get_attr_from_node(node, info)
            return attr, self.container[attr]

    class PrevNodeTypeAttr(NodeAttr):
        def get_attr_from_node(self, node: ov.Node, info: Dict[str, Any]):
            return get_prev_node(node, info["prev_node_port_id"]).type_info.name

    class NextNodeTypeAttr(NodeAttr):
        def get_attr_from_node(self, node: ov.Node, info: Dict[str, Any]):
            return get_next_nodes(node, info["next_node_port_id"])[0].type_info.name

    class ShapeAttr(NodeAttr):
        def get_attr_from_node(self, node: ov.Node, info: Dict[str, Any] = None):
            return tuple(node.shape)

    class PrevEdgeLen(NodeAttr):
        def get_attr_from_node(self, node: ov.Node, info: Dict[str, Any]):
            return len(get_prev_node(node, info["prev_node_port_id"]).output(0).target_inputs)

    STOP_NODES = ["Input", "Constant", "Parameter", "Output", "Result"]

    class NNodesTypeAttrs(NodeAttr):
        def __init__(self, n) -> None:
            super().__init__()
            self._n = n

        @property
        def name(self):
            return super().name + str(self._n)

        @abstractmethod
        def get_following_node(self, node, info):
            pass

        @abstractmethod
        def get_node_info(self, node, info):
            pass

        def get_attr_from_node(self, node: ov.Node, info: Dict[str, Any]):
            retval = []
            following_node = node
            for _ in range(self._n):
                following_node = self.get_following_node(following_node, info)
                retval.append(self.get_node_info(following_node, info))
                if following_node.type_info.name in ATTRS.STOP_NODES:
                    break
            return tuple(retval)

    class NPrevNodesTypeAttrs(NNodesTypeAttrs):
        def get_following_node(self, node, info):
            return get_prev_node(node, info["prev_node_port_id"])

        def get_node_info(self, node, info):
            return node.type_info.name

    class NNextNodesTypeAttrs(NNodesTypeAttrs):
        def get_following_node(self, node, info):
            return get_next_nodes(node, info["next_node_port_id"])[0]

        def get_node_info(self, node, info):
            return node.type_info.name

    class NPrevNodesShapeAttrs(NNodesTypeAttrs):
        def get_following_node(self, node, info):
            return get_prev_node(node, info["prev_node_port_id"])

        def get_node_info(self, node, info):
            return tuple(node.shape)

    class NNextNodesTypeAttrs(NNodesTypeAttrs):
        def get_following_node(self, node, info):
            return get_next_nodes(node, info["next_node_port_id"])[0]

        def get_node_info(self, node, info):
            return tuple(node.shape)

    class NumNodes(NodeAttr):
        def __init__(self, target_node_type: str) -> None:
            super().__init__()
            self._target_node_type = target_node_type

        @property
        def name(self):
            return super().name + self._target_node_type

        @abstractmethod
        def get_following_node(self, node, info):
            pass

        def get_attr_from_node(self, node: ov.Node, info: Dict[str, Any] = None):
            idx = 0
            while True:
                following_node = self.get_following_node(node, info)
                idx += 1

                if following_node.type_info.name in [self._target_node_type] + ATTRS.STOP_NODES:
                    break
                node = following_node

            if following_node.type_info.name != self._target_node_type:
                idx = -1
            return idx

    class NumNodesAfter(NumNodes):
        def get_following_node(self, node, info):
            return get_prev_node(node, 0)

    class NumNodesBefore(NumNodes):
        def get_following_node(self, node, info):
            return get_next_nodes(node, 0)[0]


def get_prev_node(node, input_port):
    return node.input(input_port).get_source_output().get_node()


def get_next_nodes(node, output_port):
    return [x.get_node() for x in node.output(output_port).target_inputs]


def get_attributes_by_is(ov_model, is_info: List[Dict[str, Any]]):
    for info in is_info:
        attributes = info["attrs"]
        type_ = info["type"]
        print(100 * "#")
        print(f"For type {type_}:")
        for node in ov_model.get_ordered_ops():
            if node.type_info.name == type_:
                for attr in attributes:
                    attr.register_node(node, info)

        for node in ov_model.get_ordered_ops():
            if node.get_friendly_name() in info["nodes"]:
                print(100 * "/")
                print(f"For node {node.get_friendly_name()}")
                for attr in attributes:
                    print(100 * "-")
                    cur_attr, owners = attr.get_stat(node, info)
                    # owners = '>10' if len(owners) > 10 else owners
                    print(f"{attr.name}, {cur_attr} : {owners}")


def detr_resnet50_criteriums(ov_model):
    is_info = [
        {
            "type": "Convolution",
            "attrs": (
                ATTRS.PrevNodeTypeAttr(),
                ATTRS.NextNodeTypeAttr(),
                ATTRS.ShapeAttr(),
                ATTRS.NumNodesAfter("MaxPool"),
            ),
            "prev_node_port_id": 0,
            "next_node_port_id": 0,
            "nodes": [
                "Conv_107",
                "Conv_112",
                "Conv_123",
                "Conv_132",
                "Conv_154",
                "Conv_165",
                "Conv_238",
                "Conv_426/WithoutBiases",
            ],
        },
        {
            "type": "MatMul",
            "attrs": (
                ATTRS.PrevNodeTypeAttr(),
                ATTRS.NextNodeTypeAttr(),
                ATTRS.ShapeAttr(),
                ATTRS.NumNodesAfter("Reshape"),
                ATTRS.NumNodesBefore("Sigmoid"),
            ),
            "prev_node_port_id": 0,
            "next_node_port_id": 0,
            "nodes": [
                "MatMul_557",
                "MatMul_619",
                "MatMul_2077",
                "MatMul_2675",
                "MatMul_2678",
            ],
        },
    ]
    get_attributes_by_is(ov_model, is_info)


MODEL_ID_TO_IGNORED_SCOPE_CRITERIUM_BUILDER = {
    "detr_resnet50": detr_resnet50_criteriums,
}


def print_attributes_by_is(model_name, xml_path, bin_path):
    ov_model = ov.Core().read_model(model=xml_path, weights=bin_path)
    MODEL_ID_TO_IGNORED_SCOPE_CRITERIUM_BUILDER[model_name](ov_model)


def main(argv):
    parser = ArgumentParser()
    parser.add_argument("--model-name", help="Model name specified in IS_TO_BUILED_MAP map", required=True)
    parser.add_argument("--xml-path", help="Path to input IR xml file", required=True)
    parser.add_argument("--bin-path", help="Path to input IR bin file", required=True)

    args = parser.parse_args(args=argv)
    print_attributes_by_is(args.model_name, args.xml_path, args.bin_path)
