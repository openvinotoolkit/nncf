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
import networkx as nx
import torch

from nncf.dynamic_graph.graph import NNCFGraph
from nncf.dynamic_graph.operator_metatypes import NoopMetatype, HardTanhMetatype, TanhMetatype, RELUMetatype, \
    PRELUMetatype, ELUMetatype, GELUMetatype, SigmoidMetatype, SoftmaxMetatype, AvgPool2dMetatype, MaxPool2dMetatype, \
    DropoutMetatype, Conv1dMetatype, Conv2dMetatype, Conv3dMetatype, BatchNormMetatype, CatMetatype, AddMetatype, \
    SubMetatype, DivMetatype, MulMetatype, LinearMetatype, MatMulMetatype, MinMetatype, MaxMetatype, MeanMetatype
from nncf.nncf_logger import logger as nncf_logger
from nncf.nncf_network import NNCFNetwork
from nncf.pruning.export_utils import PruningOperationsMetatypeRegistry, identity_mask_propagation, get_input_masks, \
    fill_input_masks
from nncf.pruning.utils import get_sources_of_node

PRUNING_OPERATOR_METATYPES = PruningOperationsMetatypeRegistry("operator_metatypes")


# pylint: disable=protected-access
class DefaultMetaOp:
    subtypes = []
    additional_types = []

    def mask_propagation(self, model: NNCFNetwork, nx_node: dict, graph: NNCFGraph, nx_graph: nx.DiGraph):
        """
        Propagate mask through a node using masks of all inputs and pruning mask of current node (if any).
        Should set the following attributes:
        input_masks - list of masks of input nodes (None if there is no mask in some input)
        output_mask - resulting mask of nx_node operation
        accept_pruned_input - can this operation work with pruned input or not
        :param model: model to prune
        :param nx_node: node from networkx graph to propagate mask through it
        :param graph: graph of model to prune
        :param nx_graph: networkx graph
        """
        raise NotImplementedError

    def input_prune(self, model: NNCFNetwork, nx_node, graph: NNCFGraph, nx_graph: nx.DiGraph):
        """
        Prune nx_node by input_masks (if masks is not none and operation support it).
        """

    def output_prune(self, model: NNCFNetwork, nx_node, graph: NNCFGraph, nx_graph: nx.DiGraph):
        """
        Prune nx_node by output_mask (if mask is not none and operation support it).
        """

    @classmethod
    def get_all_op_aliases(cls):
        """
        :return: list of all aliases of types in metatype
        """
        op_types = []
        for subtype in cls.subtypes:
            op_types.extend(subtype.get_all_aliases())
        op_types = list(set(op_types)) + cls.additional_types
        return op_types


@PRUNING_OPERATOR_METATYPES.register('model_input')
class Input(DefaultMetaOp):
    subtypes = [NoopMetatype]

    def mask_propagation(self, model, nx_node, graph, nx_graph):
        nx_node['input_masks'] = []
        nx_node['output_mask'] = None
        nx_node['accept_pruned_input'] = False


@PRUNING_OPERATOR_METATYPES.register('identity_mask_propagation')
class IdentityMaskForwardOps(DefaultMetaOp):
    subtypes = [HardTanhMetatype, TanhMetatype, RELUMetatype, PRELUMetatype, ELUMetatype, GELUMetatype, SigmoidMetatype,
                SoftmaxMetatype, AvgPool2dMetatype, MaxPool2dMetatype, DropoutMetatype]
    additional_types = ["h_sigmoid", "h_swish"]

    def mask_propagation(self, model: NNCFNetwork, nx_node, graph: NNCFGraph, nx_graph: nx.DiGraph):
        identity_mask_propagation(nx_node, nx_graph)
        nx_node['accept_pruned_input'] = True


@PRUNING_OPERATOR_METATYPES.register('convolution')
class Convolution(DefaultMetaOp):
    subtypes = [Conv1dMetatype, Conv2dMetatype, Conv3dMetatype]

    def mask_propagation(self, model: NNCFNetwork, nx_node, graph: NNCFGraph, nx_graph: nx.DiGraph):
        output_mask = None
        accept_pruned_input = True
        is_depthwise = False
        input_masks = get_input_masks(nx_node, nx_graph)

        nncf_node = graph._nx_node_to_nncf_node(nx_node)
        node_module = model.get_module_by_scope(nncf_node.op_exec_context.scope_in_model)

        if node_module.pre_ops:
            output_mask = node_module.pre_ops['0'].op.binary_filter_pruning_mask

        # In case of group convs we can't prune by output filters
        if node_module.groups != 1:
            if node_module.weight.size(1) == 1:
                # Depthwise case
                is_depthwise = True
                output_mask = input_masks[0]
            else:
                accept_pruned_input = False
                output_mask = None

        nx_node['input_masks'] = input_masks
        nx_node['output_mask'] = output_mask
        nx_node['accept_pruned_input'] = accept_pruned_input
        nx_node['is_depthwise'] = is_depthwise

    def input_prune(self, model: NNCFNetwork, nx_node, graph: NNCFGraph, nx_graph: nx.DiGraph):
        input_mask = nx_node['input_masks'][0]
        if input_mask is None:
            return
        bool_mask = torch.tensor(input_mask, dtype=torch.bool)
        new_num_channels = int(torch.sum(input_mask))

        nncf_node = graph._nx_node_to_nncf_node(nx_node)
        node_module = model.get_module_by_scope(nncf_node.op_exec_context.scope_in_model)
        is_depthwise = nx_node['is_depthwise']
        old_num_clannels = int(node_module.weight.size(1))

        if is_depthwise:
            # In depthwise case prune output channels by input mask, here only fix for new number of input channels
            node_module.groups = new_num_channels
            node_module.in_channels = new_num_channels
        else:
            out_channels = node_module.weight.size(0)
            broadcasted_mask = bool_mask.repeat(out_channels).view(out_channels, bool_mask.size(0))
            new_weight_shape = list(node_module.weight.shape)
            new_weight_shape[1] = new_num_channels

            node_module.in_channels = new_num_channels
            node_module.weight = torch.nn.Parameter(node_module.weight[broadcasted_mask].view(new_weight_shape))

        nncf_logger.info('Pruned Convolution {} by input mask. Old input filters number: {}, new filters number:'
                         ' {}.'.format(nx_node['key'], old_num_clannels, new_num_channels))

    def output_prune(self, model: NNCFNetwork, nx_node, graph: NNCFGraph, nx_graph: nx.DiGraph):
        mask = nx_node['output_mask']
        if mask is None:
            return

        bool_mask = torch.tensor(mask, dtype=torch.bool)

        nncf_node = graph._nx_node_to_nncf_node(nx_node)
        node_module = model.get_module_by_scope(nncf_node.op_exec_context.scope_in_model)
        old_num_clannels = int(node_module.weight.size(0))

        node_module.out_channels = int(torch.sum(mask))
        node_module.weight = torch.nn.Parameter(node_module.weight[bool_mask])

        if node_module.bias is not None and not nx_node['is_depthwise']:
            node_module.bias = torch.nn.Parameter(node_module.bias[bool_mask])

        nncf_logger.info('Pruned Convolution {} by pruning mask. Old output filters number: {}, new filters number:'
                         ' {}.'.format(nx_node['key'], old_num_clannels, node_module.out_channels))


@PRUNING_OPERATOR_METATYPES.register('batch_norm')
class BatchNorm(DefaultMetaOp):
    subtypes = [BatchNormMetatype]

    def mask_propagation(self, model: NNCFNetwork, nx_node, graph: NNCFGraph, nx_graph: nx.DiGraph):
        identity_mask_propagation(nx_node, nx_graph)
        nx_node['accept_pruned_input'] = True

    def input_prune(self, model: NNCFNetwork, nx_node, graph: NNCFGraph, nx_graph: nx.DiGraph):
        input_mask = nx_node['input_masks'][0]
        if input_mask is None:
            return

        nncf_node = graph._nx_node_to_nncf_node(nx_node)
        node_module = model.get_module_by_scope(nncf_node.op_exec_context.scope_in_model)

        bool_mask = torch.tensor(input_mask, dtype=torch.bool)
        old_num_clannels = int(node_module.weight.size(0))
        new_num_channels = int(torch.sum(input_mask))

        node_module.num_features = new_num_channels
        node_module.weight = torch.nn.Parameter(node_module.weight[bool_mask])
        node_module.bias = torch.nn.Parameter(node_module.bias[bool_mask])
        node_module.running_mean = torch.nn.Parameter(node_module.running_mean[bool_mask], requires_grad=False)
        node_module.running_var = torch.nn.Parameter(node_module.running_var[bool_mask], requires_grad=False)

        nncf_logger.info('Pruned BatchNorm {} by input mask. Old num features: {}, new num features:'
                         ' {}.'.format(nx_node['key'], old_num_clannels, new_num_channels))


@PRUNING_OPERATOR_METATYPES.register('concat')
class Concat(DefaultMetaOp):
    subtypes = [CatMetatype]

    def all_inputs_from_convs(self, nx_node, nx_graph, graph):
        """
        Return whether all input sources of nx_node is convolutions or not
        :param nx_node: node to determine it's sources
        :param nx_graph:  networkx graph to work with
        :param graph:  NNCF graph to work with
        """
        inputs = [u for u, _ in nx_graph.in_edges(nx_node['key'])]
        input_masks = get_input_masks(nx_node, nx_graph)

        for i, inp in enumerate(inputs):
            # If input has mask ->  it went from convolution (source of this node is a convolution)
            if input_masks[i] is not None:
                continue
            nncf_input_node = graph._nx_node_to_nncf_node(nx_graph.nodes[inp])
            source_nodes = get_sources_of_node(nncf_input_node, graph, Convolution.get_all_op_aliases() +
                                               StopMaskForwardOps.get_all_op_aliases() +
                                               Input.get_all_op_aliases())
            sources_types = [node.op_exec_context.operator_name for node in source_nodes]
            if any([t in sources_types for t in StopMaskForwardOps.get_all_op_aliases()]):
                return False
        return True

    def check_concat(self, nx_node, nx_graph, graph):
        if self.all_inputs_from_convs(nx_node, nx_graph, graph):
            return True
        return False

    def mask_propagation(self, model: NNCFNetwork, nx_node, graph: NNCFGraph, nx_graph: nx.DiGraph):
        result_mask = None
        accept_pruned_input = False

        if self.check_concat(nx_node, nx_graph, graph):
            input_masks, filled_input_masks = fill_input_masks(nx_node, nx_graph)

            if all([mask is None for mask in input_masks]):
                result_mask = None
            else:
                result_mask = torch.cat(filled_input_masks)
                accept_pruned_input = True

        nx_node['input_masks'] = input_masks
        nx_node['output_mask'] = result_mask
        nx_node['accept_pruned_input'] = accept_pruned_input


@PRUNING_OPERATOR_METATYPES.register('elementwise')
class Elementwise(DefaultMetaOp):
    subtypes = [AddMetatype, SubMetatype, DivMetatype, MulMetatype]

    def mask_propagation(self, model: NNCFNetwork, nx_node, graph: NNCFGraph, nx_graph: nx.DiGraph):
        input_masks = get_input_masks(nx_node, nx_graph)

        nx_node['input_masks'] = input_masks
        nx_node['output_mask'] = None
        nx_node['accept_pruned_input'] = False


@PRUNING_OPERATOR_METATYPES.register('stop_propagation_ops')
class StopMaskForwardOps(DefaultMetaOp):
    subtypes = [MeanMetatype, MaxMetatype, MinMetatype, LinearMetatype, MatMulMetatype]

    def mask_propagation(self, model: NNCFNetwork, nx_node, graph: NNCFGraph, nx_graph: nx.DiGraph):
        input_masks = get_input_masks(nx_node, nx_graph)

        nx_node['input_masks'] = input_masks
        nx_node['output_mask'] = None
        nx_node['accept_pruned_input'] = False


class ModelPruner:
    CAN_PRUNE_ATTR = 'can_prune'

    def __init__(self, model: NNCFNetwork, graph: NNCFGraph, nx_graph: nx.DiGraph):
        self.model = model
        self.graph = graph
        self.nx_graph = nx_graph

    def node_propagate_can_prune_attr(self, nx_node_key):
        nx_node = self.nx_graph.nodes[nx_node_key]
        is_conv = self.graph.node_type_fn(nx_node) in Convolution().get_all_op_aliases()
        return not is_conv or (is_conv and nx_node['is_depthwise'])

    def node_accept_different_inputs(self, nx_node):
        """
        Return whether nx_node accept pruned and not pruned inputs as inputs at the same time.
        """
        node_type = self.graph.node_type_fn(nx_node)
        return node_type in Concat.get_all_op_aliases()

    @staticmethod
    def get_class_by_type_name(type_name):
        """
        Return class of metaop that corresponds to type_name type.
        """
        cls = PRUNING_OPERATOR_METATYPES.get_operator_metatype_by_op_name(type_name)
        if cls is None:
            nncf_logger.warning(
                "Layer {} is not pruneable - will not propagate pruned filters through it".format(type_name))
            cls = StopMaskForwardOps
        return cls

    def mask_propagation(self):
        """
        Mask propagation in graph:
        to propagate masks run method mask_propagation (of metaop of current node) on all nodes in topological order.
        """
        sorted_nodes = [self.nx_graph.nodes[node_name] for node_name in nx.topological_sort(self.nx_graph)]
        for node in sorted_nodes:
            node_type = self.graph.node_type_fn(node)
            cls = self.get_class_by_type_name(node_type)()
            cls.mask_propagation(self.model, node, self.graph, self.nx_graph)
        nncf_logger.info('Finished mask propagation in graph')

    def propagate_can_prune_attr_up(self):
        """
        Propagating can_prune attribute in reversed topological order.
        This attribute depends on accept_pruned_input and can_prune attributes of output nodes.
        Node can_prune is True if all outputs accept_pruned_input is True and all outputs
        (except convs because conv can be pruned by input and output independently).
        """
        for node_name in self.nx_graph.nodes:
            self.nx_graph.nodes[node_name][ModelPruner.CAN_PRUNE_ATTR] = True

        reversed_sorted_nodes = reversed([self.nx_graph.nodes[name] for name in nx.topological_sort(self.nx_graph)])
        for node in reversed_sorted_nodes:
            # Check all output nodes accept_pruned_input attribute
            out_edges = self.nx_graph.out_edges(node['key'])
            outputs_accept_pruned_input = all(self.nx_graph.nodes[key]['accept_pruned_input'] for _, key in out_edges)

            # Check all output nodes can_prune attribute
            outputs_will_be_pruned = all([self.nx_graph.nodes[key][ModelPruner.CAN_PRUNE_ATTR] for _, key in out_edges
                                          if self.node_propagate_can_prune_attr(key)])
            node[ModelPruner.CAN_PRUNE_ATTR] = outputs_accept_pruned_input and outputs_will_be_pruned

        nncf_logger.info('Propagated can_prune attribute up')

    def propagate_can_prune_attr_down(self):
        """
        Propagating can_prune attribute down to fix all branching cases with one pruned and one not pruned
        branches.
        """
        sorted_nodes = [self.nx_graph.nodes[name] for name in nx.topological_sort(self.nx_graph)]
        for node in sorted_nodes:
            # Propagate attribute only in not conv case
            if self.node_propagate_can_prune_attr(node['key']):
                in_edges = self.nx_graph.in_edges(node['key'])
                can_prune = all(self.nx_graph.nodes[key][ModelPruner.CAN_PRUNE_ATTR] for key, _ in in_edges)
                can_prune_any = any(self.nx_graph.nodes[key][ModelPruner.CAN_PRUNE_ATTR] for key, _ in in_edges)

                if (not self.node_accept_different_inputs(node) and not can_prune) or \
                        (self.node_accept_different_inputs(node) and not can_prune_any):
                    node[ModelPruner.CAN_PRUNE_ATTR] = can_prune

        nncf_logger.info('Propagated can_prune attribute down')

    def apply_mask(self):
        """
        Applying propagated masks for all nodes in topological order:
        if all inputs of node can_prune -> running input_prune method for this node
        if node[ModelPruner.CAN_PRUNE_ATTR] -> running output_prune method for this node
        """
        sorted_nodes = [self.nx_graph.nodes[name] for name in nx.topological_sort(self.nx_graph)]
        with torch.no_grad():
            for node in sorted_nodes:
                node_type = self.graph.node_type_fn(node)
                node_cls = self.get_class_by_type_name(node_type)()

                in_edges = self.nx_graph.in_edges(node['key'])
                can_prune_input = all(self.nx_graph.nodes[key][ModelPruner.CAN_PRUNE_ATTR] for key, _ in in_edges)
                if can_prune_input:
                    node_cls.input_prune(self.model, node, self.graph, self.nx_graph)

                if node[ModelPruner.CAN_PRUNE_ATTR]:
                    node_cls.output_prune(self.model, node, self.graph, self.nx_graph)
        nncf_logger.info('Finished mask applying step')

    def prune_model(self):
        """
        Model pruner work in three stages:
        1. Mask propagation: propagate pruning masks through the graph.
        2. Propagate can_prune attribute (up and then down) through the graph.This attribute shows can we really
        prune some node or not.
        3. Applying masks accordingly with can_prune attribute (only when can prune).
        :return:
        """
        nncf_logger.info('Start pruning model')
        self.mask_propagation()
        self.propagate_can_prune_attr_up()
        self.propagate_can_prune_attr_down()
        self.apply_mask()
        nncf_logger.info('Finished pruning model')
