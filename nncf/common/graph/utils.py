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
from typing import Union

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


def check_scope_names_match_graph(config: NNCFConfig, graph: NNCFGraph) -> None:
    """
    Raise RuntimeError in case if scope names in NNCF config do not match model graph.

    :param config: An instance of NNCFConfig.
    :param graph: The model graph.
    """

    node_list = graph.get_all_nodes()

    def _find_not_matched_paterns(patterns: Union[List, str]) -> List:
        if not patterns:
            return []

        if not isinstance(patterns, list):
            patterns = [patterns]

        matched_patterns = set()
        for node in node_list:
            for pattern in patterns:
                if matches_any(node.node_name, pattern):
                    matched_patterns.add(pattern)
        return list(set(patterns) - matched_patterns)

    def _check_scopes(config_part_name: str, cfg_part: NNCFConfig) -> str:
        not_matched_ignored_scopes = _find_not_matched_paterns(cfg_part.get("ignored_scopes", []))
        not_matched_target_scopes = _find_not_matched_paterns(cfg_part.get("target_scopes", []))

        err_msg = ""
        if not_matched_ignored_scopes:
            err_msg += f" - in {config_part_name} 'ignored_scopes': {not_matched_ignored_scopes}\n"
        if not_matched_target_scopes:
            err_msg += f" - in {config_part_name} 'target_scopes': {not_matched_target_scopes}\n"
        return err_msg

    err_message = ""
    err_message += _check_scopes("global part", config)

    algo_configs = config.get("compression", [])
    algo_configs = [algo_configs] if not isinstance(algo_configs, list) else algo_configs
    for algo_config in algo_configs:
        err_message += _check_scopes(f"'{algo_config['algorithm']}' algorithm", algo_config)

    if err_message:
        err_message = "No match has been found among the model operations " \
                      "for the following ignored/target scope definitions:\n" \
                      + err_message \
                      + "Refer to the original_graph.dot to discover the operations " \
                      "in the model currently visible to NNCF and specify the ignored/target " \
                      "scopes in terms of the names there."

        raise RuntimeError(err_message)
