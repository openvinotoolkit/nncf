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

from functools import partial
from typing import List

from nncf import NNCFConfig
from nncf.common.graph import NNCFGraph, NNCFNode
from nncf.common.pruning.utils import traverse_function
from nncf.common.utils.helpers import matches_any
from nncf.common.utils.logger import logger


def get_concat_axis(input_shapes: List[List[int]], output_shapes: List[List[int]]) -> int:
    """
    Returns concatenation axis by given input and output shape of concat node.

    :param input_shapes: Input_shapes of given concat node.
    :param output_shapes: Output_shapes of given concat node.
    :returns: Concatenation axis of given concat node.
    """
    axis = None
    none_dim = None
    for idx, (dim_in, dim_out) in enumerate(zip(input_shapes[0], output_shapes[0])):
        if dim_in != dim_out:
            axis = idx
            break
        if dim_in is None:
            none_dim = idx

    if axis is None:
        if none_dim is None:
            axis = -1
            logger.warning('Identity concat node detected')
        else:
            axis = none_dim

    return axis


def get_first_nodes_of_type(graph: NNCFGraph, op_types: List[str]) -> List[NNCFNode]:
    """
    Looking for first node in graph with type in `op_types`.
    First == layer with type in `op_types`, that there is a path from the input such that there are no other
    operations with type in `op_types` on it.

    :param op_types: Types of modules to track.
    :param graph: Graph to work with.
    :return: List of all first nodes with type in `op_types`.
    """
    graph_roots = graph.get_input_nodes()  # NNCFNodes here

    visited = {node_id: False for node_id in graph.get_all_node_ids()}
    partial_traverse_function = partial(traverse_function,
                                        type_check_fn=lambda x: x in op_types,
                                        visited=visited)

    first_nodes_of_type = []
    for root in graph_roots:
        first_nodes_of_type.extend(graph.traverse_graph(root, partial_traverse_function))
    return first_nodes_of_type


def get_split_axis(input_shapes: List[List[int]], output_shapes: List[List[int]]) -> int:
    """
    Returns split/chunk axis by given input and output shape of split/chunk node.

    :param input_shapes: Input_shapes of given split/chunk node.
    :param output_shapes: Output_shapes of given split/chunk node.
    :returns: Split/Chunk axis of given split/chunk node.
    """
    axis = None
    for idx, (dim_in, dim_out) in enumerate(zip(input_shapes[0], output_shapes[0])):
        if dim_in != dim_out:
            axis = idx
            break

    if axis is None:
        axis = -1
        logger.warning('Identity split/concat node detected')

    return axis


def check_config_matches_graph(config: NNCFConfig, graph: NNCFGraph) -> None:
    """
    Rise RuntimeError in case if configs does not match graph.

    :param config: An instance of NNCFConfig that defines compression methods.
    :param graph: The model graph.
    """

    node_list = graph.get_all_nodes()

    def _find_unused_paterns(patterns):
        if not patterns:
            return []

        if not isinstance(patterns, list):
            patterns = [patterns]

        used_patterns = set()
        for node in node_list:
            for pattern in patterns:
                if matches_any(node.node_name, pattern):
                    used_patterns.add(pattern)
        return list(set(patterns) - used_patterns)

    def _check_scopes(scope_name):
        err_msg = ""
        unused_pattern = _find_unused_paterns(config.get(scope_name))
        if unused_pattern:
            err_msg += f"NNCF config in global part contains unused patterns in '{scope_name}': {unused_pattern}\n"

        algo_configs = config.get("compression", [])
        algo_configs = [algo_configs] if isinstance(algo_configs, dict) else algo_configs
        for algo_config in algo_configs:
            algo_name = algo_config.get("algorithm")
            unused_pattern = _find_unused_paterns(algo_config.get(scope_name))
            if unused_pattern:
                err_msg += f"NNCF config for '{algo_name}' algorithm contains unused patterns for " \
                           f"'{scope_name}': {unused_pattern}\n"
        if err_msg:
            raise RuntimeError(err_msg)

    _check_scopes("ignored_scopes")
    _check_scopes("target_scopes")
