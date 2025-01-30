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

import sys
from argparse import ArgumentParser
from typing import List, Optional

import openvino.runtime as ov


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


def get_prev_node(node, input_port):
    return node.input(input_port).get_source_output().get_node()


def get_next_nodes(node, output_port):
    return [x.get_node() for x in node.output(output_port).target_inputs]


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


MODEL_ID_TO_IGNORED_SCOPE_BUILDER_MAP = {
    "mobilenet_v3_large_tf": mobilenet_v3_large_tf_torch_is,
    "mobilenet_v3_large_tf2": mobilenet_v3_large_tf2_is,
    "ssd_resnet34_1200_caffe2": ssd_resnet34_1200_caffe_is,
    "east_resnet_v1_50_tf": east_resnet_v1_50_tf_is,
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
