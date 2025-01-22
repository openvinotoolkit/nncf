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
from typing import Dict, List

import torch
from torch import nn

from nncf import NNCFConfig
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.logging import nncf_logger
from nncf.common.pruning.clusterization import Cluster
from nncf.common.pruning.clusterization import Clusterization
from nncf.common.pruning.mask_propagation import MaskPropagationAlgorithm
from nncf.common.pruning.utils import is_prunable_depthwise_conv
from nncf.common.utils.helpers import create_table
from nncf.config.extractors import extract_algo_specific_config
from nncf.config.schemata.defaults import PRUNE_BATCH_NORMS
from nncf.config.schemata.defaults import PRUNE_DOWNSAMPLE_CONVS
from nncf.config.schemata.defaults import PRUNE_FIRST_CONV
from nncf.torch.algo_selector import ZeroCompressionLoss
from nncf.torch.compression_method_api import PTCompressionAlgorithmBuilder
from nncf.torch.compression_method_api import PTCompressionAlgorithmController
from nncf.torch.graph.transformations.commands import PTInsertionCommand
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.graph.transformations.commands import TransformationPriority
from nncf.torch.graph.transformations.layout import PTTransformationLayout
from nncf.torch.module_operations import UpdateWeightAndBias
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.pruning.operations import PT_PRUNING_OPERATOR_METATYPES
from nncf.torch.pruning.structs import PrunedModuleInfo
from nncf.torch.pruning.tensor_processor import PTNNCFPruningTensorProcessor
from nncf.torch.pruning.utils import init_output_masks_in_graph
from nncf.torch.utils import get_model_device


class BasePruningAlgoBuilder(PTCompressionAlgorithmBuilder):
    def __init__(self, config, should_init: bool = True):
        super().__init__(config, should_init)
        params = self._algo_config.get("params", {})
        self._set_default_params_for_ranking_type(params)
        self._params = params

        self.prune_first = params.get("prune_first_conv", PRUNE_FIRST_CONV)
        self.prune_batch_norms = params.get("prune_batch_norms", PRUNE_BATCH_NORMS)
        self.prune_downsample_convs = params.get("prune_downsample_convs", PRUNE_DOWNSAMPLE_CONVS)

        self._prunable_types = self.get_op_types_of_pruned_modules()

        from nncf.common.pruning.node_selector import PruningNodeSelector

        self.pruning_node_selector = PruningNodeSelector(
            PT_PRUNING_OPERATOR_METATYPES,
            self._prunable_types,
            self.get_types_of_grouping_ops(),
            self.ignored_scopes,
            self.target_scopes,
            self.prune_first,
            self.prune_downsample_convs,
        )

        self.pruned_module_groups_info = []
        self._pruned_norms_operators = []

    @staticmethod
    def _set_default_params_for_ranking_type(params: Dict) -> None:
        """
        Setting default parameter values of pruning algorithm depends on the ranking type:
        for learned_ranking `all_weights` must be True (in case of False was set by the user, an Exception will be
        raised), `prune_first_conv`, `prune_downsample_convs` are recommended to be True (this
        params will be set to True by default (and remain unchanged if the user sets some value).
        :param params: dict with parameters of the algorithm from config
        """
        learned_ranking = "interlayer_ranking_type" in params and params["interlayer_ranking_type"] == "learned_ranking"
        if not learned_ranking:
            return
        nncf_logger.info(
            "LeGR pruning sets `prune_first_conv`, `prune_downsample_convs`, "
            "`all_weights` to True by default. Adjusting this is not recommended."
        )
        params.setdefault("prune_first_conv", True)
        params.setdefault("prune_downsample_convs", True)
        if params.get("all_weights") is False:
            raise Exception(
                "For LeGR pruning the `all_weights` config parameter be set to `true`."
                "Adjust the config accordingly if you want to proceed."
            )
        params.setdefault("all_weights", True)

    def _get_transformation_layout(self, target_model: NNCFNetwork) -> PTTransformationLayout:
        layout = PTTransformationLayout()
        commands = self._prune_weights(target_model)
        for command in commands:
            layout.register(command)
        return layout

    def _prune_weights(self, target_model: NNCFNetwork):
        target_model_graph = target_model.nncf.get_original_graph()
        groups_of_nodes_to_prune = self.pruning_node_selector.create_pruning_groups(target_model_graph)

        device = get_model_device(target_model)
        insertion_commands = []
        self.pruned_module_groups_info = Clusterization[PrunedModuleInfo](lambda x: x.node_name)

        for i, group in enumerate(groups_of_nodes_to_prune.get_all_clusters()):
            group_minfos = []
            for node in group.elements:
                node_name = node.node_name
                module = target_model.nncf.get_containing_module(node_name)
                module_scope = target_model_graph.get_scope_by_node_name(node_name)
                # Check that we need to prune weights in this op
                assert self._is_pruned_module(module)

                nncf_logger.debug(f"Will prune the weights for the operation: {node_name}")
                pruning_block = self.create_weight_pruning_operation(module, node_name)
                # Hook for weights and bias
                hook = UpdateWeightAndBias(pruning_block).to(device)
                insertion_commands.append(
                    PTInsertionCommand(
                        PTTargetPoint(TargetType.PRE_LAYER_OPERATION, target_node_name=node_name),
                        hook,
                        TransformationPriority.PRUNING_PRIORITY,
                    )
                )
                group_minfos.append(
                    PrunedModuleInfo(
                        node_name=node_name,
                        module_scope=module_scope,
                        module=module,
                        operand=pruning_block,
                        node_id=node.node_id,
                        is_depthwise=is_prunable_depthwise_conv(node),
                    )
                )

            cluster = Cluster[PrunedModuleInfo](i, group_minfos, [n.node_id for n in group.elements])
            self.pruned_module_groups_info.add_cluster(cluster)

        # Propagate masks to find norm layers to prune
        init_output_masks_in_graph(target_model_graph, self.pruned_module_groups_info.get_all_nodes())
        MaskPropagationAlgorithm(
            target_model_graph, PT_PRUNING_OPERATOR_METATYPES, PTNNCFPruningTensorProcessor
        ).mask_propagation()

        # Adding binary masks also for Batch/Group/Layer Norms to allow applying masks after propagation
        types_to_apply_mask = ["group_norm", "layer_norm"]
        if self.prune_batch_norms:
            types_to_apply_mask.append("batch_norm")

        all_norm_layers = target_model_graph.get_nodes_by_types(types_to_apply_mask)
        for node in all_norm_layers:
            if node.attributes["output_mask"] is None:
                # Skip elements that will not be pruned
                continue

            node_name = node.node_name
            module = target_model.nncf.get_containing_module(node_name)

            pruning_block = self.create_weight_pruning_operation(module, node_name)
            # Hook for weights and bias
            hook = UpdateWeightAndBias(pruning_block).to(device)
            insertion_commands.append(
                PTInsertionCommand(
                    PTTargetPoint(TargetType.PRE_LAYER_OPERATION, target_node_name=node_name),
                    hook,
                    TransformationPriority.PRUNING_PRIORITY,
                )
            )
            self._pruned_norms_operators.append((node, pruning_block, module))
        return insertion_commands

    def create_weight_pruning_operation(self, module, node_name):
        raise NotImplementedError

    def _is_pruned_module(self, module: nn.Module):
        """
        Return whether this module should be pruned or not.
        """
        raise NotImplementedError

    def get_op_types_of_pruned_modules(self):
        """
        Returns list of operation types that should be pruned.
        """
        raise NotImplementedError

    def get_types_of_grouping_ops(self):
        raise NotImplementedError

    def initialize(self, model: NNCFNetwork) -> None:
        pass


class BasePruningAlgoController(PTCompressionAlgorithmController):
    def __init__(
        self,
        target_model: NNCFNetwork,
        prunable_types: List[str],
        pruned_module_groups_info: Clusterization[PrunedModuleInfo],
        config: NNCFConfig,
    ):
        super().__init__(target_model)
        self._loss = ZeroCompressionLoss(get_model_device(target_model))
        self._prunable_types = prunable_types
        self.config = config
        self.pruning_config = extract_algo_specific_config(config, "filter_pruning")
        params = self.pruning_config.get("params", {})
        self.pruned_module_groups_info = pruned_module_groups_info
        self.prune_batch_norms = params.get("prune_batch_norms", PRUNE_BATCH_NORMS)
        self.prune_first = params.get("prune_first_conv", PRUNE_FIRST_CONV)
        self.prune_downsample_convs = params.get("prune_downsample_convs", PRUNE_DOWNSAMPLE_CONVS)
        self.prune_flops = False
        self.check_pruning_level(params)
        self._hooks = []

    def freeze(self):
        raise NotImplementedError

    def set_pruning_level(self, pruning_level):
        raise NotImplementedError

    def step(self, next_step):
        pass

    def check_pruning_level(self, params):
        """
        Check that set only one of pruning target params
        """
        pruning_target = params.get("pruning_target", None)
        pruning_flops_target = params.get("pruning_flops_target", None)
        if pruning_target and pruning_flops_target:
            raise ValueError("Only one parameter from 'pruning_target' and 'pruning_flops_target' can be set.")
        if pruning_flops_target:
            self.prune_flops = True

    def _clean_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def get_mask(self, minfo: PrunedModuleInfo) -> torch.Tensor:
        """
        Returns pruning mask for minfo.module.
        """
        raise NotImplementedError

    def pruning_level_for_mask(self, minfo: PrunedModuleInfo):
        mask = self.get_mask(minfo)
        pruning_level = 1 - mask.nonzero().size(0) / max(mask.view(-1).size(0), 1)
        return pruning_level

    def mask_shape(self, minfo: PrunedModuleInfo):
        mask = self.get_mask(minfo)
        return mask.shape

    def get_stats_for_pruned_modules(self):
        """
        Creates a table with layer pruning level statistics
        """
        header = ["Name", "Weight's shape", "Bias shape", "Layer PR"]
        rows_data = []
        for minfo in self.pruned_module_groups_info.get_all_nodes():
            rows_data.append(
                [
                    str(minfo.module_scope),
                    list(minfo.module.weight.size()),
                    list(minfo.module.bias.size()) if minfo.module.bias is not None else [],
                    self.pruning_level_for_mask(minfo),
                ]
            )

        return create_table(header, rows_data, max_col_widths=[33, 20, 6, 8])
