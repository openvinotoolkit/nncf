"""
 Copyright (c) 2020 Intel Corporation
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
import torch

from nncf.dynamic_graph.version_agnostic_op_names import get_version_agnostic_name
from nncf.registry import Registry


def get_input_masks(nx_node, nx_graph):
    """
    Return input masks for all inputs of nx_node.
    """
    input_edges = list(nx_graph.in_edges(nx_node['key']))
    input_masks = [nx_graph.nodes[input_node]['output_mask'] for input_node, _ in input_edges]
    return input_masks


def identity_mask_propagation(nx_node, nx_graph):
    """
    Propagates input mask through nx_node.
    """
    input_masks = get_input_masks(nx_node, nx_graph)
    assert len(input_masks) == 1
    nx_node['input_masks'] = input_masks
    nx_node['output_mask'] = input_masks[0]


def fill_input_masks(nx_node, nx_graph):
    """
    Return all input masks and all input masks with all None replaced by identity masks.
    """
    input_edges = sorted(list(nx_graph.in_edges(nx_node['key'])), key=lambda edge: nx_graph.edges[edge]['in_port'])
    input_masks = [nx_graph.nodes[input_node]['output_mask'] for input_node, _ in input_edges]

    filled_input_masks = []
    for i, mask in enumerate(input_masks):
        if mask is None:
            mask = torch.ones(nx_graph.edges[input_edges[i]]['activation_shape'][1])
        filled_input_masks.append(mask)
    return input_masks, filled_input_masks


class PruningOperationsMetatypeRegistry(Registry):
    def __init__(self, name):
        super().__init__(name)
        self._op_name_to_op_class = {}

    def register(self, name=None):
        name_ = name
        super_register = super()._register

        def wrap(obj):
            cls_name = name_
            if cls_name is None:
                cls_name = obj.__name__

            super_register(obj, cls_name)
            op_names = obj.get_all_op_aliases()
            for name in op_names:
                name = get_version_agnostic_name(name)
                if name not in self._op_name_to_op_class:
                    self._op_name_to_op_class[name] = obj
                else:
                    assert self._op_name_to_op_class[name] == obj, \
                        "Inconsistent operator type registry - single patched op name maps to multiple metatypes!"
            return obj

        return wrap

    def get_operator_metatype_by_op_name(self, op_name: str):
        if op_name in self._op_name_to_op_class:
            return self._op_name_to_op_class[op_name]
        return None
