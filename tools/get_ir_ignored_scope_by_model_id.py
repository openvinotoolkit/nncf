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

import sys
from argparse import ArgumentParser
from typing import List, Optional

import openvino.runtime as ov
from ir_ignored_scope_criterium_search import ATTRS
from ir_ignored_scope_criterium_search import get_next_nodes
from ir_ignored_scope_criterium_search import get_prev_node


def print_ignored_scope_by_model_name(model_name: str, xml_path: str, bin_path: str) -> None:
    if model_name not in MODEL_ID_TO_IGNORED_SCOPE_BUILDER_MAP:
        print(f"Model {model_name} is not found")
        return

    ov_model = ov.Core().read_model(model=xml_path, weights=bin_path)
    is_names = MODEL_ID_TO_IGNORED_SCOPE_BUILDER_MAP[model_name](ov_model)
    print(f"Ignored scope for the model {model_name}")
    if is_names is None:
        print("Error occured during ignored scope colletion")
        return

    for name in is_names:
        print(name)


def mobilenet_v3_large_tf_torch_is(ov_model: ov.Model) -> Optional[List[str]]:
    for node in ov_model.get_ordered_ops():
        if node.type_info.name == "GroupConvolution":
            return [node.get_friendly_name()]
    return None


def mobilenet_v3_large_tf2_is(ov_model: ov.Model) -> Optional[List[str]]:
    retval = []

    for node in ov_model.get_ordered_ops():
        if node.type_info.name == "GroupConvolution":
            retval.append(node.get_friendly_name())
        if node.type_info.name == "Add":
            prev_nodes_names = {get_prev_node(node, i).type_info.name for i in range(2)}
            if "Add" in prev_nodes_names and "HSwish" in prev_nodes_names:
                retval.append(node.get_friendly_name())
                assert len(retval) == 2
                return retval
    return None


def ssd_resnet34_1200_caffe_is(ov_model: ov.Model) -> Optional[List[str]]:
    for result in ov_model.get_results():
        prev_node = get_prev_node(result, 0)
        if prev_node.type_info.name == "Add":
            return [prev_node.get_friendly_name()]
    return None


def east_resnet_v1_50_tf_is(ov_model: ov.Model) -> Optional[List[str]]:
    idx = 0
    for node in ov_model.get_ordered_ops()[::-1]:
        if node.type_info.name == "Concat":
            if idx < 1:
                idx += 1
                continue
            next_nodes = get_next_nodes(node, 0)
            assert len(next_nodes) == 1
            next_node = next_nodes[0]
            assert next_node.type_info.name == "Convolution"
            return [next_node.get_friendly_name()]
    return None


def denoise_is(ov_model: ov.Model):
    add_idx = 0
    concat_conv_idx = 0
    avg_pool_concat_idx = 0
    retval = []
    mul_found = False
    for node in ov_model.get_ordered_ops():
        if node.type_info.name == "Add":
            add_idx += 1
            if add_idx == 2:
                retval.append(node.get_friendly_name())

        if node.type_info.name == "Multiply" and not mul_found:
            for i in range(2):
                prev_node = get_prev_node(node, i)
                if prev_node.type_info.name == "Unsqueeze":
                    input_ = get_prev_node(prev_node, 0)
                    if input_.get_friendly_name() == "t_param":
                        retval.append(node.get_friendly_name())
                        mul_found = True
                    break
            else:
                RuntimeError("Could not find target multiply node")

        if node.type_info.name == "Concat":
            next_node = get_next_nodes(node, 0)[0]
            if next_node.type_info.name == "Convolution":
                concat_conv_idx += 1
                if concat_conv_idx in [2, 3, 4]:
                    retval.append(next_node.get_friendly_name())

        if node.type_info.name == "AvgPool":
            next_node = get_next_nodes(node, 0)[0]
            if next_node.type_info.name == "Convolution":
                avg_pool_concat_idx += 1
                if avg_pool_concat_idx in [2, 4]:
                    retval.append(next_node.get_friendly_name())
                if avg_pool_concat_idx in [2, 3, 4]:
                    retval.append(node.get_friendly_name())

    assert len(retval) == 10
    return retval


def detr_resnet50_is(ov_model: ov.Model) -> Optional[List[str]]:
    print("WARNING: node MatMul_2077 is not present in this criteriums")
    criteriums = [
        {
            "type": "Convolution",
            "checks": [
                {
                    "cond": lambda node: ATTRS.ShapeAttr().get_attr_from_node(node) == (1, 64, 400, 569),
                    "amount": 1,
                    "collected": 0,
                },
                {
                    "cond": lambda node: ATTRS.PrevNodeTypeAttr().get_attr_from_node(node, {"prev_node_port_id": 0})
                    == "MaxPool",
                    "amount": 2,
                    "collected": 0,
                },
                {
                    "cond": lambda node: ATTRS.NumNodesAfter("MaxPool").get_attr_from_node(node, {}) == 14,
                    "amount": 1,
                    "collected": 0,
                },
                {
                    "cond": lambda node: ATTRS.NumNodesAfter("MaxPool").get_attr_from_node(node, {}) == 31,
                    "amount": 2,
                    "collected": 0,
                },
                {
                    "cond": lambda node: ATTRS.NumNodesAfter("MaxPool").get_attr_from_node(node, {}) == 91,
                    "amount": 1,
                    "collected": 0,
                },
                {
                    "cond": lambda node: ATTRS.ShapeAttr().get_attr_from_node(node) == (1, 256, 25, 36),
                    "amount": 1,
                    "collected": 0,
                },
            ],
        },
        {
            "type": "MatMul",
            "checks": [
                {
                    "cond": lambda node: ATTRS.NumNodesAfter("Reshape").get_attr_from_node(node) == 6,
                    "amount": 1,
                    "collected": 0,
                },
                {
                    "cond": lambda node: ATTRS.NumNodesAfter("Reshape").get_attr_from_node(node) == 10,
                    "amount": 1,
                    "collected": 0,
                },
                {
                    "cond": lambda node: ATTRS.NumNodesBefore("Sigmoid").get_attr_from_node(node) == 5,
                    "amount": 1,
                    "collected": 0,
                },
                {
                    "cond": lambda node: ATTRS.NumNodesBefore("Sigmoid").get_attr_from_node(node) == 2,
                    "amount": 1,
                    "collected": 0,
                },
            ],
        },
    ]
    return _get_is_from_criteriums(ov_model, criteriums)


def _get_is_from_criteriums(ov_model, criteriums):
    retval = []
    for node in ov_model.get_ordered_ops():
        for cr in criteriums:
            if node.type_info.name == cr["type"]:
                for check in cr["checks"]:
                    if check["cond"](node):
                        check["collected"] += 1
                        retval.append(node.get_friendly_name())
                        continue
    for cr in criteriums:
        for check in cr["checks"]:
            assert check["amount"] == check["collected"], f'Checks for type {cr["type"]} failed'
    return retval


MODEL_ID_TO_IGNORED_SCOPE_BUILDER_MAP = {
    "mobilenet_v3_large_tf": mobilenet_v3_large_tf_torch_is,
    "mobilenet_v3_large_tf2": mobilenet_v3_large_tf2_is,
    "ssd_resnet34_1200_caffe2": ssd_resnet34_1200_caffe_is,
    "east_resnet_v1_50_tf": east_resnet_v1_50_tf_is,
    "Denoise": denoise_is,
    "detr_resnet50": detr_resnet50_is,
}


def main(argv):
    parser = ArgumentParser()
    parser.add_argument("--model-name", help="Model name specified in IS_TO_BUILED_MAP map", required=True)
    parser.add_argument("--xml-path", help="Path to input IR xml file", required=True)
    parser.add_argument("--bin-path", help="Path to input IR bin file", required=True)

    args = parser.parse_args(args=argv)
    print_ignored_scope_by_model_name(args.model_name, args.xml_path, args.bin_path)


if __name__ == "__main__":
    main(sys.argv[1:])
