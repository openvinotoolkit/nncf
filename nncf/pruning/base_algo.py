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
from typing import List, Dict

from functools import partial, update_wrapper
from nncf.dynamic_graph.context import Scope

from nncf.pruning.export_helpers import IdentityMaskForwardOps, Elementwise
from texttable import Texttable
from torch import nn

from nncf.compression_method_api import CompressionAlgorithmBuilder, \
    CompressionAlgorithmController
from nncf.dynamic_graph.graph import InputAgnosticOperationExecutionContext, NNCFNode
from nncf.module_operations import UpdateWeight
from nncf.nncf_logger import logger as nncf_logger
from nncf.nncf_network import NNCFNetwork, InsertionPoint, InsertionCommand, InsertionType, OperationPriority
from nncf.pruning.filter_pruning.layers import apply_filter_binary_mask
from nncf.pruning.model_analysis import cluster_special_ops_in_model, NodesCluster, Clusterization
from nncf.pruning.utils import get_bn_for_module_scope, \
    get_first_pruned_modules, get_last_pruned_modules, is_conv_with_downsampling, is_grouped_conv, is_depthwise_conv, \
    get_previous_conv, get_sources_of_node


class PrunedModuleInfo:
    BN_MODULE_NAME = 'bn_module'
    # TODO: delete next bn + rewrite all places with it
    DEPTHWISE_BN_NAME = 'next_bn_module'

    def __init__(self, module_name: str, module: nn.Module, operand, related_modules: Dict):
        self.module_name = module_name
        self.module = module
        self.operand = operand
        self.related_modules = related_modules


class NodeInfo:
    def __init__(self, nncf_node: NNCFNode,  module: nn.Module, module_scope: Scope):
        self.node = nncf_node
        self.id = nncf_node.node_id
        self.module = module
        self.module_scope = module_scope


class BasePruningAlgoBuilder(CompressionAlgorithmBuilder):
    def __init__(self, config, should_init: bool = True):
        super().__init__(config, should_init)
        params = config.get('params', {})
        self._params = params

        self.prune_first = params.get('prune_first_conv', False)
        self.prune_last = params.get('prune_last_conv', False)
        self.prune_batch_norms = params.get('prune_batch_norms', False)
        self.prune_downsample_convs = params.get('prune_downsample_convs', False)

        # TODO: are we need this common list without groups?
        self._pruned_module_info = []
        self.pruned_module_groups = []

    def apply_to(self, target_model: NNCFNetwork) -> NNCFNetwork:
        insertion_commands = self._prune_weights(target_model)
        for command in insertion_commands:
            target_model.register_insertion_command(command)
        target_model.register_algorithm(self)
        return target_model

    def _get_modules_that_should_be_pruned_together(self, target_model: NNCFNetwork):
        """
        This function groups ALL modules with pruning types to groups that should be pruned together.
        1. Create clusters for special ops (eltwises) that should be pruned together
        2. Create groups of nodes that should be pruned together (taking into account clusters of special ops)
        3. Add remaining single nodes
        4. Unite clusters for Conv + Depthwise conv (should be pruned together too)
        5. Checks for groups (all nodes in group can prune or all group can't be pruned)
        Return groups of modules that
        :param target_model:
        :return:
        """
        graph = target_model._original_graph
        pruned_types = self.get_types_of_pruned_modules()
        all_modules_to_prune = target_model.get_nncf_modules()
        all_nodes_to_prune = graph.get_all_nodes_of_type(self.get_types_of_pruned_modules())  # NNCFNodes here
        assert len(all_nodes_to_prune) <= len(all_modules_to_prune)

        # 1. Clusters for special ops (eltwises)
        special_ops_clusterization = cluster_special_ops_in_model(target_model, Elementwise.get_all_op_aliases(),
                                                            IdentityMaskForwardOps.get_all_op_aliases())

        pruned_nodes_clusterization = Clusterization("id")

        # 2. Clusters for nodes that should be pruned together (taking into account clusters for eltwises)
        for i, cluster in enumerate(special_ops_clusterization.get_all_clusters()):
            all_pruned_inputs = set()
            for node in cluster.nodes:
                sources = get_sources_of_node(node, graph, pruned_types)
                for source_node in sources:
                    # TODO: Is this check is really needed here?
                    if source_node.op_exec_context.operator_name in pruned_types:
                        source_scope = source_node.op_exec_context.scope_in_model
                        source_module = target_model.get_module_by_scope(source_scope)
                        source_node_info = NodeInfo(source_node, source_module, source_scope)

                        all_pruned_inputs.add(source_node_info)
            cluster = NodesCluster(i, list(all_pruned_inputs), [n.id for n in all_pruned_inputs])
            pruned_nodes_clusterization.add_cluster(cluster, i)

        last_cluster_idx = len(special_ops_clusterization.get_all_clusters())

        # 3. Add remaining single nodes as separate clusters
        for node in all_nodes_to_prune:
            if not pruned_nodes_clusterization.is_not_in_any_cluster(node.node_id):
                scope = node.op_exec_context.scope_in_model
                module = target_model.get_module_by_scope(scope)
                node_info = NodeInfo(node, module, scope)

                cluster = NodesCluster(last_cluster_idx, [node_info], [node.node_id])
                pruned_nodes_clusterization.add_cluster(cluster, last_cluster_idx)

                last_cluster_idx += 1

        # 4. Unite clusters for Conv + Depthwise conv (should be pruned together too)
        for node in all_nodes_to_prune:
            scope = node.op_exec_context.scope_in_model
            module = target_model.get_module_by_scope(scope)
            cluster_id = pruned_nodes_clusterization.get_cluster_for_node(node.node_id).id

            if is_depthwise_conv(module):
                previous_conv = get_previous_conv(target_model, module, scope)
                if previous_conv:
                    previous_conv_cluster_id = pruned_nodes_clusterization.get_cluster_for_node(previous_conv.node_id).id
                    pruned_nodes_clusterization.union_clusters(cluster_id, previous_conv_cluster_id)

        # 5. Checks for groups (all nodes in group can prune or not).
        for cluster in pruned_nodes_clusterization.get_all_clusters():
            can_prune_nodes = [self._can_prune_module(target_model, node_info.module, node_info.module_scope) for node_info in cluster.nodes]
            if not all([can_prune[0] for can_prune in can_prune_nodes]):
                # TODO: beautiful informative logging here
                nncf_logger.info("Group of nodes {} can't be pruned".format(cluster.nodes))
                pruned_nodes_clusterization.delete_cluster(cluster.id)
        return pruned_nodes_clusterization

    def _can_prune_module(self, target_model: NNCFNetwork, module, module_scope):
        prune = True
        msg = None

        input_non_pruned_modules = get_first_pruned_modules(target_model,
                                                            self.get_types_of_pruned_modules() + ['linear'])
        output_non_pruned_modules = get_last_pruned_modules(target_model,
                                                            self.get_types_of_pruned_modules() + ['linear'])
        module_scope_str = str(module_scope)

        if not self._should_consider_scope(module_scope_str):
            msg = "Ignored adding Weight Pruner in scope: {}".format(module_scope_str)
            prune = False
        elif not self.prune_first and module in input_non_pruned_modules:
            msg = "Ignored adding Weight Pruner in scope: {} because"\
                             " this scope is one of the first convolutions".format(module_scope_str)
            prune = False
        elif not self.prune_last and module in output_non_pruned_modules:
            msg = "Ignored adding Weight Pruner in scope: {} because"\
                             " this scope is one of the last convolutions".format(module_scope_str)
            prune = False
        elif is_grouped_conv(module):
            if not is_depthwise_conv(module):
                msg = "Ignored adding Weight Pruner in scope: {} because" \
                      " this scope is grouped convolution".format(module_scope_str)
                prune = False
            #     previous_conv = get_previous_conv(target_model, module, module_scope)
            #     if previous_conv:
            #         depthwise_bn = get_bn_for_module_scope(target_model, module_scope)
            #         bn_for_depthwise[str(previous_conv.op_exec_context.scope_in_model)] = depthwise_bn
        elif not self.prune_downsample_convs and is_conv_with_downsampling(module):
            msg = "Ignored adding Weight Pruner in scope: {} because"\
                             " this scope is convolution with downsample".format(module_scope_str)
            prune = False
        return prune, msg

    def _prune_weights(self, target_model: NNCFNetwork):
        grops_of_modules_to_prune = self._get_modules_that_should_be_pruned_together(target_model)

        device = next(target_model.parameters()).device
        insertion_commands = []
        self.pruned_module_groups = Clusterization('module_name')

        for i, group in enumerate(grops_of_modules_to_prune.get_all_clusters()):
            group_minfos = []
            for node in group.nodes:
                module_scope, module = node.module_scope, node.module
                # Check that we need to prune weights in this op
                assert self._is_pruned_module(module)

                module_scope_str = str(module_scope)

                nncf_logger.info("Adding Weight Pruner in scope: {}".format(module_scope_str))
                operation = self.create_weight_pruning_operation(module)
                hook = UpdateWeight(operation).to(device)
                insertion_commands.append(
                    InsertionCommand(
                        InsertionPoint(
                            InputAgnosticOperationExecutionContext("", module_scope, 0),
                            InsertionType.NNCF_MODULE_PRE_OP
                        ),
                        hook,
                        OperationPriority.PRUNING_PRIORITY
                    )
                )

                related_modules = {}
                # TODO: discuss this param: it seems like we should ALWAYS prune BN
                if self.prune_batch_norms:
                    related_modules[PrunedModuleInfo.BN_MODULE_NAME] = get_bn_for_module_scope(target_model, module_scope)

                minfo = PrunedModuleInfo(module_scope_str, module, hook.operand, related_modules)
                self._pruned_module_info.append(minfo)
                group_minfos.append(minfo)
            cluster = NodesCluster(i, group_minfos, [n.id for n in group.nodes])
            self.pruned_module_groups.add_cluster(cluster, cluster.id)
        return insertion_commands

    def build_controller(self, target_model: NNCFNetwork) -> CompressionAlgorithmController:
        return BasePruningAlgoController(target_model, self._pruned_module_info, self.pruned_module_groups, self._params)

    def create_weight_pruning_operation(self, module):
        raise NotImplementedError

    def _is_pruned_module(self, module: nn.Module):
        """
        Return whether this module should be pruned or not.
        """
        raise NotImplementedError

    def get_types_of_pruned_modules(self):
        """
        Returns list of operation types that should be pruned.
        """
        raise NotImplementedError


class BasePruningAlgoController(CompressionAlgorithmController):
    def __init__(self, target_model: NNCFNetwork,
                 pruned_module_info: List[PrunedModuleInfo],
                 pruned_module_groups: Clusterization,
                 config):
        super().__init__(target_model)
        self.config = config
        params = self.config.get("params", {})
        self.pruned_module_info = pruned_module_info
        self.pruned_module_groups = pruned_module_groups
        self.prune_first = params.get('prune_first_conv', False)
        self.prune_last = params.get('prune_last_conv', False)
        self.prune_batch_norms = params.get('prune_batch_norms', False)
        self.zero_grad = params.get('zero_grad', True)
        self._hooks = []

    def freeze(self):
        raise NotImplementedError

    def set_pruning_rate(self, pruning_rate):
        raise NotImplementedError

    def zero_grads_for_pruned_modules(self):
        """
        This function registers a hook that will set the
        gradients for pruned filters to zero.
        """
        self._clean_hooks()

        def hook(grad, mask):
            mask = mask.to(grad.device)
            return apply_filter_binary_mask(mask, grad)

        for minfo in self.pruned_module_info[:1]:
            mask = minfo.operand.binary_filter_pruning_mask
            weight = minfo.module.weight
            partial_hook = update_wrapper(partial(hook, mask=mask), hook)
            self._hooks.append(weight.register_hook(partial_hook))
            if minfo.module.bias is not None:
                bias = minfo.module.bias
                partial_hook = update_wrapper(partial(hook, mask=mask), hook)
                self._hooks.append(bias.register_hook(partial_hook))

    def _clean_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def _get_mask(self, minfo: PrunedModuleInfo):
        """
        Returns pruning mask for minfo.module.
        """
        raise NotImplementedError

    @staticmethod
    def pruning_rate_for_weight(minfo: PrunedModuleInfo):
        """
        Calculates sparsity rate for all weight elements.
        """
        weight = minfo.module.weight
        pruning_rate = 1 - weight.nonzero().size(0) / weight.view(-1).size(0)
        return pruning_rate

    @staticmethod
    def pruning_rate_for_filters(minfo: PrunedModuleInfo):
        """
        Calculates sparsity rate for weight filter-wise.
        """
        weight = minfo.module.weight
        filters_sum = weight.view(weight.size(0), -1).sum(axis=1)
        pruning_rate = 1 - len(filters_sum.nonzero()) / filters_sum.size(0)
        return pruning_rate

    def pruning_rate_for_mask(self, minfo: PrunedModuleInfo):
        mask = self._get_mask(minfo)
        pruning_rate = 1 - mask.nonzero().size(0) / max(mask.view(-1).size(0), 1)
        return pruning_rate

    def mask_shape(self, minfo: PrunedModuleInfo):
        mask = self._get_mask(minfo)
        return mask.shape

    def statistics(self):
        stats = super().statistics()
        table = Texttable()
        header = ["Name", "Weight's Shape", "Mask Shape", "Mask zero %", "PR", "Filter PR"]
        data = [header]

        for minfo in self.pruned_module_info:
            drow = {h: 0 for h in header}
            drow["Name"] = minfo.module_name
            drow["Weight's Shape"] = list(minfo.module.weight.size())

            drow["Mask Shape"] = list(self.mask_shape(minfo))

            drow["Mask zero %"] = self.pruning_rate_for_mask(minfo) * 100

            drow["PR"] = self.pruning_rate_for_weight(minfo)

            drow["Filter PR"] = self.pruning_rate_for_filters(minfo)

            row = [drow[h] for h in header]
            data.append(row)
        table.add_rows(data)

        stats["pruning_statistic_by_module"] = table
        return self.add_algo_specific_stats(stats)

    @staticmethod
    def add_algo_specific_stats(stats):
        return stats

    def get_stats_for_pruned_modules(self):
        """
        Return dict with information about pruned modules. Keys in dict is module names, values is dicts with next keys:
         'w_shape': shape of module weight,
         'b_shape': shape of module bias,
         'params_count': total number of params in module
         'mask_pr': proportion of zero elements in filter pruning mask.
        """
        stats = {}
        for minfo in self.pruned_module_info:
            layer_info = {}
            layer_info["w_shape"] = list(minfo.module.weight.size())
            layer_info["b_shape"] = list(minfo.module.bias.size()) if minfo.module.bias is not None else []
            layer_info["params_count"] = sum(p.numel() for p in minfo.module.parameters() if p.requires_grad)

            layer_info["mask_pr"] = self.pruning_rate_for_mask(minfo)

            if PrunedModuleInfo.BN_MODULE_NAME in minfo.related_modules and \
                    minfo.related_modules[PrunedModuleInfo.BN_MODULE_NAME] is not None:
                bn_info = {}
                bn_module = minfo.related_modules[PrunedModuleInfo.BN_MODULE_NAME]
                bn_info['w_shape'] = bn_module.weight.size()

                bn_info["b_shape"] = bn_module.bias.size() if bn_module.bias is not None else []
                bn_info['params_count'] = sum(p.numel() for p in bn_module.parameters() if p.requires_grad)
                bn_info["mask_pr"] = self.pruning_rate_for_mask(minfo)
                stats[minfo.module_name + '/BatchNorm'] = bn_info

            stats[minfo.module_name] = layer_info

        return stats
