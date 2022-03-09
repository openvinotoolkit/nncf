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
import random
from enum import Enum
from itertools import combinations
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from nncf.common.graph import NNCFNodeName
from nncf.common.graph.transformations.commands import TransformationCommand
from nncf.common.utils.logger import logger as nncf_logger
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.base_handler import ELASTICITY_BUILDERS
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.base_handler import ELASTICITY_HANDLERS_MAP
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.base_handler import SingleElasticityBuilder
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.base_handler import SingleElasticityHandler
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elastic_width import ElasticWidthHandler
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elasticity_dim import ElasticityDim
from nncf.experimental.torch.search_building_blocks.search_blocks import BUILDING_BLOCKS
from nncf.experimental.torch.search_building_blocks.search_blocks import BuildingBlock
from nncf.experimental.torch.search_building_blocks.search_blocks import GROUPED_BLOCK_IDS
from nncf.experimental.torch.search_building_blocks.search_blocks import ORDINAL_IDS
from nncf.experimental.torch.search_building_blocks.search_blocks import get_building_blocks
from nncf.experimental.torch.search_building_blocks.search_blocks import get_group_of_dependent_blocks
from nncf.torch.dynamic_graph.context import TracingContext
from nncf.torch.nncf_network import NNCFNetwork

block_id = int
ElasticDepthConfig = List[block_id]  # list of block indexes
ElasticDepthSearchSpace = List[ElasticDepthConfig]  # grouped list of block indexes


class ElasticDepthHandler(SingleElasticityHandler):
    """
    An interface for handling elastic depth dimension in the network, i.e. skip some layers in the model.
    """

    def __init__(self, target_model: NNCFNetwork,
                 skipped_blocks: BUILDING_BLOCKS,
                 skip_dependencies: GROUPED_BLOCK_IDS,
                 node_names_per_block: Dict[int, List[str]],
                 ordinal_ids: ORDINAL_IDS):
        super().__init__()
        self._target_model = target_model
        self._tracing_context = self._target_model.get_tracing_context()  # type: TracingContext
        self._skipped_blocks = skipped_blocks
        self._skip_dependencies = skip_dependencies
        self._ordinal_ids = ordinal_ids
        self._node_names_per_block = node_names_per_block
        self._depth_indicator = 1
        self._is_search_space_obsolete = True
        self._cached_search_space = None

    def get_transformation_commands(self) -> List[TransformationCommand]:
        """
        :return: transformation commands for introducing the elasticity to NNCFNetwork
        """
        return []

    @property
    def depth_indicator(self) -> int:
        """
        :return: depth indicator value that restricts number of skipped blocks in the independent groups of blocks
        """
        return self._depth_indicator

    @depth_indicator.setter
    def depth_indicator(self, depth_indicator: int) -> None:
        """
        Sets depth indicator value that restricts number of skipped blocks in the independent groups of blocks

        :param depth_indicator: depth indicator value
        """
        self._depth_indicator = depth_indicator
        self._is_search_space_obsolete = True

    def get_search_space(self) -> ElasticDepthSearchSpace:
        """
        :return: search space that is produced by iterating over all elastic depth parameters
        """
        if not self._is_search_space_obsolete:
            return self._cached_search_space
        range_block_ids = range(0, len(self._skipped_blocks))
        # TODO(nlyalyus): can be a huge search space. no need to iterate and filter all of them. (ticket 69746)
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
        """
        Forms an elasticity configuration that describes currently activated Subnet - indexes of all skipped blocks.

        :return: list of blocks' indexes to skip
        """
        return self._tracing_context.active_block_indexes

    def get_random_config(self) -> ElasticDepthConfig:
        """
        Forms an elasticity configuration that describes a Subnet with randomly chosen elastic depth

        :return: list of blocks' indexes to skip
        """
        num_blocks = len(self._tracing_context.skipped_blocks)
        config = []
        for i in range(0, num_blocks):
            is_skipped = bool(random.getrandbits(1))  # random.randint(0, 1)
            if is_skipped:
                config.append(i)
        return config

    def get_minimum_config(self) -> ElasticDepthConfig:
        """
        Forms an elasticity configuration that describes a Subnet with minimum elastic depth, i.e. maximum number of
        blocks is skipped.

        :return: list of blocks' indexes to skip
        """
        return list(range(0, len(self._tracing_context.skipped_blocks)))

    def get_maximum_config(self) -> ElasticDepthConfig:
        """
        Forms an elasticity configuration that describes a Subnet with maximum elastic depth, i.e. minimum number of
        blocks is skipped.

        :return: list of blocks' indexes to skip
        """
        return []

    def activate_supernet(self) -> None:
        """
        Activates the Supernet - the original network without skipping any blocks.
        """
        self.activate_maximum_subnet()

    def set_config(self, config: ElasticDepthConfig):
        """
        Activates a Subnet that doesn't execute the blocks specified in the given elasticity configuration

        :param config: list of blocks' indexes to skip
        """
        config = self._remove_inconsistent_blocks(config)
        self._tracing_context.set_active_skipped_block(config)

    def resolve_conflicts_with_other_elasticities(self,
                                                  config: ElasticDepthConfig,
                                                  elasticity_handlers: ELASTICITY_HANDLERS_MAP) -> ElasticDepthConfig:
        """
        Resolves a conflict between the given elasticity config and active elasticity configs of the given handlers.
        For example, elastic width configuration may contradict to elastic depth one. When we activate some
        configuration in the Elastic Width Handler, i.e. define number of output channels for some layers, we
        change output shapes of the layers. Consequently, it affects the blocks that can be skipped by Elastic Depth
        Handler, because input and output shapes may not be identical now.

        :param config: elasticity configuration
        :param elasticity_handlers: map of elasticity dimension to elasticity handler
        :return: elasticity configuration without conflicts with other active configs of other elasticity handlers
        """
        result = config
        if ElasticityDim.WIDTH in elasticity_handlers:
            width_handler = elasticity_handlers[ElasticityDim.WIDTH]
            assert isinstance(width_handler, ElasticWidthHandler)
            blocks = [self._tracing_context.skipped_blocks[block_idx] for block_idx in config]
            pairs_of_nodes = [(block.start_node_name, block.end_node_name) for block in blocks]
            if pairs_of_nodes:
                indexes_of_pairs = width_handler.find_pairs_of_nodes_with_different_width(pairs_of_nodes)
                if indexes_of_pairs:
                    result = [element for idx, element in enumerate(config) if idx not in indexes_of_pairs]
                    nncf_logger.debug('The blocks with indexes {} are not skipped to avoid inconsistency with width'.
                                      format(indexes_of_pairs))
        return result

    def get_kwargs_for_flops_counting(self) -> Dict[str, Any]:
        """
        Provides arguments for counting flops of the currently activated subnet.

        :return: mapping of parameters to its values
        """
        op_addresses_to_skip = []
        active_block_indexes = self._tracing_context.active_block_indexes
        if active_block_indexes is not None:
            for idx in active_block_indexes:
                op_addresses_to_skip.extend(self._node_names_per_block[idx])
        return {'op_addresses_to_skip': op_addresses_to_skip}

    def _remove_inconsistent_blocks(self, config: ElasticDepthConfig) -> ElasticDepthConfig:
        return self._remove_blocks_skipped_non_progressively(config)

    def _remove_blocks_skipped_non_progressively(self, config: ElasticDepthConfig) -> ElasticDepthConfig:
        assert self._skip_dependencies is not None, 'Please include depth dependencies in conf. Pending automation.'
        block_indexes_to_remove = []
        for block_index in config:
            tmp_block_indexes_to_remove = self._get_block_indexes_to_remove(block_index, config)
            block_indexes_to_remove.extend(tmp_block_indexes_to_remove)

        for block_index in block_indexes_to_remove:
            config.remove(block_index)
            nncf_logger.debug('The block #{} is not skipped to not violate progressive shrinking'.format(block_index))
        return config

    def _get_block_indexes_to_remove(self, block_index: int, config: ElasticDepthConfig) -> ElasticDepthConfig:
        block_indexes_to_remove = []
        for group in self._skip_dependencies.values():
            found = False
            group_index = group.index(block_index) if block_index in group else None
            if group_index is not None:
                found = True
                if len(group) - group_index > self.depth_indicator:
                    nncf_logger.debug('The block with {} did not pass the depth_indicator test'.format(block_index))
                    block_indexes_to_remove.append(block_index)
                    break
                valid_block_indexes = [group[group_index]]
                for i in range(group_index + 1, len(group)):
                    if group[i] in config:
                        valid_block_indexes.append(group[i])
                    else:
                        nncf_logger.debug('The block #{} or #{} did not satisfy requirement of next static block'.
                                          format(block_index, valid_block_indexes))
                        for valid_block_index in valid_block_indexes:
                            block_indexes_to_remove.append(valid_block_index)
                        break
            if found:
                break
        return block_indexes_to_remove


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
    """
    Determines which modifications should be made to the original FP32 model in order to introduce elastic depth
    to the model.
    """
    _state_names = EDBuilderStateNames

    def __init__(self, elasticity_params: Optional[Dict[str, Any]] = None,
                 ignored_scopes: Optional[List[str]] = None,
                 target_scopes: Optional[List[str]] = None):
        super().__init__(ignored_scopes, target_scopes, elasticity_params)
        self._mode = ElasticDepthMode.from_str(self._elasticity_params.get('mode', ElasticDepthMode.AUTO.value))
        self._skipped_blocks = None  # type: Optional[BUILDING_BLOCKS]
        self._skip_dependencies = None  # type: Optional[GROUPED_BLOCK_IDS]
        self._ordinal_ids = None  # type: Optional[ORDINAL_IDS]
        self._max_block_size = self._elasticity_params.get('max_block_size', 50)
        self._min_block_size = self._elasticity_params.get('min_block_size', 6)
        self._allow_nested_blocks = self._elasticity_params.get('allow_nested_blocks', False)
        self._allow_linear_combination = self._elasticity_params.get('allow_linear_combination', False)

        if self._mode == ElasticDepthMode.MANUAL:
            skipped_blocks = self._elasticity_params.get('skipped_blocks', [])  # type: List[List[str,str]]
            self._skipped_blocks = [BuildingBlock(*b) for b in skipped_blocks]
            self._skip_dependencies = get_group_of_dependent_blocks(self._skipped_blocks)

    def build(self, target_model: NNCFNetwork) -> ElasticDepthHandler:
        """
        Creates modifications to the given NNCFNetwork for introducing elastic depth and creates a handler object that
        can manipulate this elasticity.

        :param target_model: a target NNCFNetwork for adding modifications
        :return: a handler object that can manipulate the elastic depth.
        """
        tracing_context = target_model.get_tracing_context()
        tracing_context.elastic_depth = True

        if self._mode == ElasticDepthMode.AUTO:
            if not self._skipped_blocks and not self._skip_dependencies:
                self._skipped_blocks, self._ordinal_ids, self._skip_dependencies = \
                    get_building_blocks(
                        target_model,
                        max_block_size=self._max_block_size,
                        min_block_size=self._min_block_size,
                        allow_nested_blocks=self._allow_nested_blocks,
                        allow_linear_combination=self._allow_linear_combination
                    )
                str_bs = [str(block) for block in self._skipped_blocks]
                nncf_logger.info('\n'.join(['\n\"Found building blocks:\": [', ',\n'.join(str_bs), ']']))
        else:
            graph = target_model.get_original_graph()
            ordinal_ids = []
            for block in self._skipped_blocks:
                start_node = graph.get_node_by_name(block.start_node_name)
                end_node = graph.get_node_by_name(block.end_node_name)
                ordinal_ids.append([start_node.node_id, end_node.node_id])
            self._ordinal_ids = ordinal_ids

        tracing_context.set_elastic_blocks(self._skipped_blocks, self._ordinal_ids)
        node_names_per_block = self._get_node_names_per_block(target_model, self._skipped_blocks)

        return ElasticDepthHandler(target_model, self._skipped_blocks, self._skip_dependencies,
                                   node_names_per_block, self._ordinal_ids)

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Initializes object from the state.

        :param state: Output of `get_state()` method.
        """
        skipped_blocks_from_state = state[self._state_names.SKIPPED_BLOCKS]
        skip_dependencies_from_state = state[self._state_names.SKIPPED_BLOCKS_DEPENDENCIES]
        skipped_blocks = [BuildingBlock.from_state(bb_state) for bb_state in skipped_blocks_from_state]
        ordinal_ids_from_state = state[self._state_names.ORDINAL_IDS]
        is_skipped_blocks_diff = self._skipped_blocks and self._skipped_blocks != skipped_blocks
        is_skip_dependencies_diff = self._skip_dependencies and self._skip_dependencies != skip_dependencies_from_state
        is_ordinal_ids_diff = self._ordinal_ids and self._ordinal_ids != ordinal_ids_from_state

        if is_skipped_blocks_diff or is_skip_dependencies_diff or is_ordinal_ids_diff:
            nncf_logger.warning('Elasticity parameters were provided in two places: on init and on loading '
                                'state. The one from state is taken by ignoring the ones from init.')
        self._mode = ElasticDepthMode.from_str(state[self._state_names.MODE])
        self._skipped_blocks = skipped_blocks
        self._skip_dependencies = skip_dependencies_from_state
        self._ordinal_ids = ordinal_ids_from_state

    def get_state(self) -> Dict[str, Any]:
        """
        Returns a dictionary with Python data structures (dict, list, tuple, str, int, float, True, False, None) that
        represents state of the object.

        :return: state of the object
        """
        skipped_blocks_from_state = []
        if self._skipped_blocks is not None:
            skipped_blocks_from_state = list(map(lambda x: x.get_state(), self._skipped_blocks))
        return {
            self._state_names.MODE: self._mode.value,
            self._state_names.SKIPPED_BLOCKS: skipped_blocks_from_state,
            self._state_names.SKIPPED_BLOCKS_DEPENDENCIES: self._skip_dependencies,
            self._state_names.ORDINAL_IDS: self._ordinal_ids
        }

    @staticmethod
    def _get_node_names_per_block(target_model: NNCFNetwork, skipped_blocks) -> Dict[int, List[NNCFNodeName]]:
        graph = target_model.get_original_graph()
        all_node_names_per_block = {}
        for idx, block in enumerate(skipped_blocks):
            simple_paths = graph.get_all_simple_paths(block.start_node_name, block.end_node_name)

            node_names_in_block = set()
            for node_keys_in_path in simple_paths:
                for node_key in node_keys_in_path:
                    node = graph.get_node_by_key(node_key)
                    node_names_in_block.add(node.node_name)

            node_names_in_block.remove(block.start_node_name)
            all_node_names_per_block[idx] = list(node_names_in_block)
        return all_node_names_per_block
