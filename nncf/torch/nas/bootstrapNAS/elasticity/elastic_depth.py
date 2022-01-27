"""
 Copyright (c) 2019-2021 Intel Corporation
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
import random
from enum import Enum
from itertools import combinations
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import networkx as nx

from nncf.common.graph.transformations.commands import TransformationCommand
from nncf.common.utils.logger import logger as nncf_logger
from nncf.experimental.torch.search_building_blocks.search_blocks import BUILDING_BLOCKS
from nncf.experimental.torch.search_building_blocks.search_blocks import BuildingBlock
from nncf.experimental.torch.search_building_blocks.search_blocks import GROUPED_BLOCK_IDS
from nncf.experimental.torch.search_building_blocks.search_blocks import ORDINAL_IDS
from nncf.experimental.torch.search_building_blocks.search_blocks import get_building_blocks
from nncf.torch.dynamic_graph.context import TracingContext
from nncf.torch.nas.bootstrapNAS.elasticity.base_handler import ELASTICITY_BUILDERS
from nncf.torch.nas.bootstrapNAS.elasticity.base_handler import SingleElasticHandler
from nncf.torch.nas.bootstrapNAS.elasticity.base_handler import SingleElasticityBuilder
from nncf.torch.nas.bootstrapNAS.elasticity.elastic_width import ElasticWidthHandler
from nncf.torch.nas.bootstrapNAS.elasticity.elasticity_dim import ElasticityDim
from nncf.torch.nncf_network import NNCFNetwork

block_id = int
ElasticDepthConfig = List[block_id]  # list of block indexes
ElasticDepthSearchSpace = List[ElasticDepthConfig]  # grouped list of block indexes


class ElasticDepthHandler(SingleElasticHandler):
    def __init__(self, target_model: NNCFNetwork,
                 skipped_blocks: BUILDING_BLOCKS,
                 skip_dependencies: GROUPED_BLOCK_IDS,
                 all_skipped_nodes_per_skipped_block_idxs: Dict[int, List[str]],
                 ordinal_ids: ORDINAL_IDS,
                 width_handler: Optional[ElasticWidthHandler] = None):
        super().__init__()
        self._target_model = target_model
        self._width_handler = width_handler
        self._tracing_context = self._target_model.get_tracing_context()  # type: TracingContext
        self._skipped_blocks = skipped_blocks
        self._skip_dependencies = skip_dependencies
        self._ordinal_ids = ordinal_ids
        self._all_skipped_nodes_per_skipped_block_idxs = all_skipped_nodes_per_skipped_block_idxs
        self._depth_indicator = 1
        self._is_search_space_obsolete = True
        self._cached_search_space = None

    def get_transformation_commands(self) -> List[TransformationCommand]:
        return []

    @property
    def depth_indicator(self):
        return self._depth_indicator

    @depth_indicator.setter
    def depth_indicator(self, depth_indicator):
        self._depth_indicator = depth_indicator
        self._is_search_space_obsolete = True

    def get_search_space(self) -> ElasticDepthSearchSpace:
        if not self._is_search_space_obsolete:
            return self._cached_search_space
        range_block_ids = range(0, len(self._skipped_blocks))
        # TODO(nlyalyus): can be a huge search space. no need to iterate and filter all of them?
        possible_depth_configs = [list(combinations(range_block_ids, i + 1)) for i in range_block_ids]
        possible_depth_configs = [y for x in possible_depth_configs for y in x]
        valid_depth_configs = []
        for d_sample in possible_depth_configs:
            if list(d_sample) == self._remove_inconsistent_blocks(list(d_sample)):
                valid_depth_configs.append(list(d_sample))
        self._cached_search_space = valid_depth_configs
        self._is_search_space_obsolete = False
        return valid_depth_configs

    def get_active_config(self) -> ElasticDepthConfig:
        return self._tracing_context.active_block_idxs

    def activate_random_subnet(self):
        num_blocks = len(self._tracing_context.skipped_blocks)
        config = []
        for i in range(0, num_blocks):
            is_skipped = bool(random.getrandbits(1))  # random.randint(0, 1)
            if is_skipped:
                config.append(i)
        self.set_config(config)

    def activate_minimal_subnet(self):
        config = list(range(0, len(self._tracing_context.skipped_blocks)))
        self.set_config(config)

    def activate_maximal_subnet(self):
        config = []
        self.set_config(config)

    def activate_supernet(self):
        self.activate_maximal_subnet()

    def set_config(self, config: ElasticDepthConfig):
        config = self._remove_inconsistent_blocks(config)
        self._tracing_context.set_active_skipped_block(config)

    def get_kwargs_for_flops_counting(self) -> Dict[str, Any]:
        op_addresses_to_skip = []
        active_block_idxs = self._tracing_context.active_block_idxs
        if active_block_idxs is not None:
            for idx in active_block_idxs:
                op_addresses_to_skip.extend(self._all_skipped_nodes_per_skipped_block_idxs[idx])
        return {'op_addresses_to_skip': op_addresses_to_skip}

    def _remove_inconsistent_blocks(self, config: ElasticDepthConfig) -> ElasticDepthConfig:
        config = self._remove_blocks_with_inconsistent_width(config)
        return self._remove_blocks_skipped_non_progressively(config)

    def _remove_blocks_with_inconsistent_width(self, config: ElasticDepthConfig) -> ElasticDepthConfig:
        result = config
        if self._width_handler is not None and self._width_handler.is_active:
            pairs_of_nodes = [self._tracing_context.skipped_blocks[block_idx] for block_idx in config]
            if pairs_of_nodes:
                indexes_of_pairs = self._width_handler.find_pairs_of_nodes_with_different_width(pairs_of_nodes)
                if indexes_of_pairs:
                    result = [element for idx, element in enumerate(config) if idx not in indexes_of_pairs]
                    nncf_logger.debug(
                        f'The blocks with indexes {indexes_of_pairs} are not skipped to avoid inconsistency with width')
        return result

    def _remove_blocks_skipped_non_progressively(self, config: ElasticDepthConfig) -> ElasticDepthConfig:
        assert self._skip_dependencies is not None, 'Please include depth dependencies in conf. Pending automation.'
        to_remove = []
        for c in config:
            for group in self._skip_dependencies.values():
                found = False
                index = group.index(c) if c in group else None
                if index is not None:
                    found = True
                    if len(group) - index > self.depth_indicator:
                        # nncf_logger.debug(f'{c} did not pass the depth_indicator test')
                        to_remove.append(c)
                        break
                    valids = [group[index]]
                    for i in range(index + 1, len(group)):
                        if group[i] in config:
                            valids.append(group[i])
                        else:
                            # nncf_logger.debug(f'{c} or {valids} did not satisfy requirement of next static block')
                            for v in valids:
                                to_remove.append(v)
                            break
                if found:
                    break
        for r in to_remove:
            config.remove(r)
            nncf_logger.debug(f'The block #{r} is not skipped to not violate progressive shrinking')
        return config


class ElasticDepthMode(Enum):
    MANUAL = 'manual'
    AUTO = 'auto'

    @classmethod
    def from_str(cls, mode: str) -> 'ElasticDepthMode':
        if mode == ElasticDepthMode.MANUAL.value:
            return ElasticDepthMode.MANUAL
        if mode == ElasticDepthMode.AUTO.value:
            return ElasticDepthMode.AUTO
        raise RuntimeError(f"Unknown elasticity depth mode: {mode}."
                           f"List of supported: {[e.value for e in ElasticDepthMode]}")


class EDBuilderStateNames:
    SKIPPED_BLOCKS = 'skipped_blocks'
    SKIPPED_BLOCKS_DEPENDENCIES = 'skipped_blocks_dependencies'
    ORDINAL_IDS = 'ordinal_ids'
    MODE = 'mode'


@ELASTICITY_BUILDERS.register(ElasticityDim.DEPTH)
class ElasticDepthBuilder(SingleElasticityBuilder):
    _state_names = EDBuilderStateNames

    def __init__(self, elasticity_params: Optional[Dict[str, Any]] = None,
                 ignored_scopes: Optional[List[str]] = None,
                 target_scopes: Optional[List[str]] = None):
        super().__init__(ignored_scopes, target_scopes, elasticity_params)
        self._mode = ElasticDepthMode.from_str(self._elasticity_params.get('mode', ElasticDepthMode.AUTO.value))
        self._skipped_blocks = []
        self._skip_dependencies = {}
        self._ordinal_ids = None
        self._max_block_size = self._elasticity_params.get('max_block_size', 50)
        self._min_block_size = self._elasticity_params.get('min_block_size', 6)
        self._allow_nested_blocks = self._elasticity_params.get('allow_nested_blocks', False)
        self._allow_linear_combination = self._elasticity_params.get('allow_linear_combination', False)

        if self._mode == ElasticDepthMode.MANUAL:
            self._skipped_blocks = [BuildingBlock(*b) for b in self._elasticity_params.get('skipped_blocks', [])]
            self._skip_dependencies = self._elasticity_params.get('skipped_blocks_dependencies', {})
            self._ordinal_ids = self._elasticity_params.get('ordinal_ids', None)

    def build(self, target_model: NNCFNetwork,
              width_handler: Optional[ElasticWidthHandler] = None) -> ElasticDepthHandler:
        tracing_context = target_model.get_tracing_context()
        tracing_context._elastic_depth = True

        if self._mode == ElasticDepthMode.AUTO:
            if not self._skipped_blocks and not self._skip_dependencies:
                self._skipped_blocks, self._ordinal_ids, self._skip_dependencies = get_building_blocks(target_model,
                                                                                                       max_block_size=self._max_block_size,
                                                                                                       min_block_size=self._min_block_size,
                                                                                                       allow_nested_blocks=self._allow_nested_blocks,
                                                                                                       allow_linear_combination=self._allow_linear_combination)
                str_bs = [str(block) for block in self._skipped_blocks]
                nncf_logger.info('\n'.join(['\n\"Found building blocks:\": [', ',\n'.join(str_bs), ']']))

        tracing_context.set_elastic_blocks(self._skipped_blocks, self._ordinal_ids)
        all_skipped_nodes_per_skipped_block_idxs = self._get_all_skipped_nodes(target_model, self._skipped_blocks)

        return ElasticDepthHandler(target_model, self._skipped_blocks, self._skip_dependencies,
                                   all_skipped_nodes_per_skipped_block_idxs, width_handler)

    def load_state(self, state: Dict[str, Any]) -> None:
        skipped_blocks_from_state = state[self._state_names.SKIPPED_BLOCKS]
        skip_dependencies_from_state = state[self._state_names.SKIPPED_BLOCKS_DEPENDENCIES]
        skipped_blocks = [BuildingBlock.from_state(bb_state) for bb_state in skipped_blocks_from_state]
        ordinal_ids_from_state = state[self._state_names.ORDINAL_IDS]
        if self._skipped_blocks and self._skipped_blocks != skipped_blocks or \
            self._skip_dependencies and self._skip_dependencies != skip_dependencies_from_state or \
            self._ordinal_ids and self._ordinal_ids != ordinal_ids_from_state:
            nncf_logger.warning('Elasticity parameters were provided in two places: on init and on loading '
                                'state. The one from state is taken by ignoring the ones from init.')
        self._mode = ElasticDepthMode.from_str(state[self._state_names.MODE])
        self._skipped_blocks = skipped_blocks
        self._skip_dependencies = skip_dependencies_from_state
        self._ordinal_ids = ordinal_ids_from_state

    def get_state(self) -> Dict[str, Any]:
        return {
            self._state_names.MODE: self._mode.value,
            self._state_names.SKIPPED_BLOCKS: list(map(lambda x: x.get_state(), self._skipped_blocks)),
            self._state_names.SKIPPED_BLOCKS_DEPENDENCIES: self._skip_dependencies,
            self._state_names.ORDINAL_IDS: self._ordinal_ids
        }

    @staticmethod
    def _get_all_skipped_nodes(target_model: NNCFNetwork, skipped_blocks) -> Dict[int, List[str]]:
        graph = target_model.get_original_graph()
        all_skipped_nodes_per_skipped_block_idxs = {}
        for idx, block in enumerate(skipped_blocks):

            start_node_key, end_node_key = None, None
            for node in graph._nx_graph._node.values():
                if block.start_node == str(node['node_name']):
                    start_node_key = node['key']
                if block.end_node == str(node['node_name']):
                    end_node_key = node['key']
            simple_paths = nx.all_simple_paths(graph._nx_graph, start_node_key, end_node_key)
            all_nodes_in_block = set()
            for node_keys_in_path in simple_paths:
                for node_key in node_keys_in_path:
                    all_nodes_in_block.add(str(graph._nx_graph._node[node_key]['node_name']))
            start_op_address = str(graph._nx_graph._node[start_node_key]['node_name'])
            all_nodes_in_block.remove(start_op_address)
            all_skipped_nodes_per_skipped_block_idxs[idx] = list(all_nodes_in_block)
        return all_skipped_nodes_per_skipped_block_idxs
