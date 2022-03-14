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
from itertools import combinations
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from nncf.common.graph import NNCFNodeName
from nncf.common.graph.transformations.commands import TransformationCommand
from nncf.common.utils.logger import logger as nncf_logger
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.base_handler import BaseElasticityParams
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.base_handler import ELASTICITY_BUILDERS
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.base_handler import ELASTICITY_HANDLERS_MAP
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.base_handler import ELASTICITY_PARAMS
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.base_handler import SingleElasticityBuilder
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.base_handler import SingleElasticityHandler
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elastic_width import ElasticWidthHandler
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elasticity_dim import ElasticityDim
from nncf.experimental.torch.search_building_blocks.search_blocks import BuildingBlock
from nncf.experimental.torch.search_building_blocks.search_blocks import BuildingBlocks
from nncf.experimental.torch.search_building_blocks.search_blocks import GroupedBlockIDs
from nncf.experimental.torch.search_building_blocks.search_blocks import OrdinalIDs
from nncf.experimental.torch.search_building_blocks.search_blocks import get_building_blocks
from nncf.experimental.torch.search_building_blocks.search_blocks import get_group_of_dependent_blocks
from nncf.torch.dynamic_graph.context import TracingContext
from nncf.torch.nncf_network import NNCFNetwork

BlockId = int
ElasticDepthConfig = List[BlockId]  # list of block indexes
ElasticDepthSearchSpace = List[ElasticDepthConfig]  # grouped list of block indexes


class ElasticDepthHandler(SingleElasticityHandler):
    """
    An interface for handling elastic depth dimension in the network, i.e. skip some layers in the model.
    """

    def __init__(self, target_model: NNCFNetwork,
                 skipped_blocks: BuildingBlocks,
                 skip_dependencies: GroupedBlockIDs,
                 node_names_per_block: Dict[int, List[NNCFNodeName]],
                 ordinal_ids: OrdinalIDs):
        """
        Constructor

        :param target_model: a target NNCFNetwork for adding modifications
        :param skipped_blocks: list of building blocks to be skipped.
        :param skip_dependencies: indexes of building blocks grouped by connectivity.
        Blocks that follow each other in the graph (i.e. they are connected by one edge) belong to the same group.
        :param node_names_per_block: mapping of block id and all node names inside the block.
        :param ordinal_ids: original ids of the start and end of the block in original graph
        """
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

    def activate_subnet_for_config(self, config: ElasticDepthConfig):
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

    def get_names_of_skipped_nodes(self) -> List[NNCFNodeName]:
        """
        :return: names of nodes that are currently skipped by Elastic Depth
        """
        names_of_skipped_nodes = []
        active_block_indexes = self._tracing_context.active_block_indexes
        if active_block_indexes is not None:
            for idx in active_block_indexes:
                names_of_skipped_nodes.extend(self._node_names_per_block[idx])
        return names_of_skipped_nodes

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


class EDBuilderStateNames:
    SKIPPED_BLOCKS = 'skipped_blocks'
    SKIPPED_BLOCKS_DEPENDENCIES = 'skipped_blocks_dependencies'
    OrdinalIds = 'ordinal_ids'


class EDParamsStateNames:
    MODE = 'mode'
    MAX_BLOCK_SIZE = 'max_block_size'
    MIN_BLOCK_SIZE = 'min_block_size'
    ALLOW_NESTED_BLOCKS = 'allow_nested_blocks'
    ALLOW_LINEAR_COMBINATION = 'allow_linear_combination'
    SKIPPED_BLOCKS = 'skipped_blocks'


@ELASTICITY_PARAMS.register(ElasticityDim.DEPTH)
class ElasticDepthParams(BaseElasticityParams):
    _state_names = EDParamsStateNames

    def __init__(self,
                 max_block_size: int,
                 min_block_size: int,
                 allow_nested_blocks: bool,
                 allow_linear_combination: bool,
                 skipped_blocks: Optional[List[List[NNCFNodeName]]] = None):
        """
        Constructor

        :param max_block_size: Defines minimal number of operations in the block. Option is available for the auto mode
         only. Default value is 50.
        :param min_block_size: Defines minimal number of operations in the skipping block. Option is available for the
         auto mode only. Default value is 6.
        :param allow_nested_blocks: If true, automatic block search will consider nested blocks: the ones that are part
         of bigger block. By default, nested blocks are declined during the search and bigger blocks are found only.
         False, by default.
        :param allow_linear_combination: If False, automatic block search will decline blocks that are a combination of
          other blocks, in another words, that consist entirely of operations of other blocks. False, by default.
        :param skipped_blocks: list of building blocks to be skipped. The block is defined by names of start and end
        nodes. If the parameter is not specified blocks to skip are found automatically.
        """
        self.max_block_size = max_block_size
        self.min_block_size = min_block_size
        self.allow_nested_blocks = allow_nested_blocks
        self.allow_linear_combination = allow_linear_combination
        self.skipped_blocks = skipped_blocks

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ElasticDepthParams':
        """
        Creates the object from its config.
        """
        kwargs = {
            cls._state_names.MAX_BLOCK_SIZE: config.get(cls._state_names.MAX_BLOCK_SIZE, 50),
            cls._state_names.MIN_BLOCK_SIZE: config.get(cls._state_names.MIN_BLOCK_SIZE, 6),
            cls._state_names.ALLOW_NESTED_BLOCKS: config.get(cls._state_names.ALLOW_NESTED_BLOCKS, False),
            cls._state_names.ALLOW_LINEAR_COMBINATION: config.get(cls._state_names.ALLOW_LINEAR_COMBINATION, False),
            cls._state_names.SKIPPED_BLOCKS: config.get(cls._state_names.SKIPPED_BLOCKS),
        }
        return cls(**kwargs)

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> 'ElasticDepthParams':
        """
        Creates the object from its state.

        :param state: Output of `get_state()` method.
        """
        return cls(**state)

    def get_state(self) -> Dict[str, Any]:
        """
        Returns the compression loss state.

        :return: The compression loss state.
        """
        return {
            self._state_names.MAX_BLOCK_SIZE: self.max_block_size,
            self._state_names.MIN_BLOCK_SIZE: self.min_block_size,
            self._state_names.ALLOW_NESTED_BLOCKS: self.allow_nested_blocks,
            self._state_names.ALLOW_LINEAR_COMBINATION: self.allow_linear_combination,
            self._state_names.SKIPPED_BLOCKS: self.skipped_blocks,
        }

    def __eq__(self, other: 'ElasticDepthParams') -> bool:
        return self.__dict__ == other.__dict__


@ELASTICITY_BUILDERS.register(ElasticityDim.DEPTH)
class ElasticDepthBuilder(SingleElasticityBuilder):
    """
    Determines which modifications should be made to the original FP32 model in order to introduce elastic depth
    to the model.
    """
    _state_names = EDBuilderStateNames

    def __init__(self,
                 params: ElasticDepthParams,
                 ignored_scopes: Optional[List[str]] = None,
                 target_scopes: Optional[List[str]] = None):
        """
        :param params: parameters to configure elastic depth.
        :param ignored_scopes: A list of strings to match against NNCFGraph node names
          and ignore matching nodes. Ignored nodes will not considered for skipping.
        :param target_scopes: A list of strings to match against NNCFGraph and define a set of nodes
          to be considered for skipping. When `ignored_scopes` is a "denylist",
          the `target_scopes` is an "allowlist"; otherwise, same effects apply as for `ignored_scopes`.
        """
        super().__init__(ignored_scopes, target_scopes)
        self._params = params
        self._skipped_blocks = None  # type: Optional[BuildingBlocks]
        self._skip_dependencies = None  # type: Optional[GroupedBlockIDs]
        if self._params.skipped_blocks is not None:
            self._skipped_blocks = [BuildingBlock(*b) for b in self._params.skipped_blocks]
            self._skip_dependencies = get_group_of_dependent_blocks(self._skipped_blocks)
        self._ordinal_ids = None  # type: Optional[OrdinalIDs]

    def build(self, target_model: NNCFNetwork) -> ElasticDepthHandler:
        """
        Creates modifications to the given NNCFNetwork for introducing elastic depth and creates a handler object that
        can manipulate this elasticity.

        :param target_model: a target NNCFNetwork for adding modifications
        :return: a handler object that can manipulate the elastic depth.
        """
        tracing_context = target_model.get_tracing_context()
        tracing_context.elastic_depth = True

        if self._skipped_blocks is None:
            self._skipped_blocks, self._ordinal_ids, self._skip_dependencies = \
                get_building_blocks(
                    target_model,
                    max_block_size=self._params.max_block_size,
                    min_block_size=self._params.min_block_size,
                    allow_nested_blocks=self._params.allow_nested_blocks,
                    allow_linear_combination=self._params.allow_linear_combination
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
        params_from_state = state[SingleElasticityBuilder._state_names.ELASTICITY_PARAMS]
        params = ElasticDepthParams.from_state(params_from_state)
        if self._params and self._params != params:
            nncf_logger.warning(
                'Different elasticity parameters were provided in two places: on init and on loading '
                'state. The one from state is taken by ignoring the ones from init.')
        self._params = params
        skipped_blocks_from_state = state[self._state_names.SKIPPED_BLOCKS]
        self._skip_dependencies = state[self._state_names.SKIPPED_BLOCKS_DEPENDENCIES]
        self._skipped_blocks = [BuildingBlock.from_state(bb_state) for bb_state in skipped_blocks_from_state]
        self._ordinal_ids = state[self._state_names.OrdinalIds]

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
            SingleElasticityBuilder._state_names.ELASTICITY_PARAMS: self._params.get_state(),
            self._state_names.SKIPPED_BLOCKS: skipped_blocks_from_state,
            self._state_names.SKIPPED_BLOCKS_DEPENDENCIES: self._skip_dependencies,
            self._state_names.OrdinalIds: self._ordinal_ids
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
