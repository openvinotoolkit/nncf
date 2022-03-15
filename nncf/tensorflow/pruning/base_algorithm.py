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

from typing import Dict
from typing import List
from typing import Tuple

import tensorflow as tf

from nncf import NNCFConfig
from nncf.common.graph import NNCFNode
from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNodeName
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.common.pruning.clusterization import Cluster
from nncf.common.pruning.clusterization import Clusterization
from nncf.common.pruning.mask_propagation import MaskPropagationAlgorithm
from nncf.common.pruning.node_selector import PruningNodeSelector
from nncf.common.pruning.statistics import PrunedLayerSummary
from nncf.common.pruning.structs import PrunedLayerInfoBase
from nncf.common.pruning.utils import is_prunable_depthwise_conv
from nncf.common.pruning.utils import get_output_channels
from nncf.common.utils.logger import logger as nncf_logger
from nncf.common.compression import BaseCompressionAlgorithmController
from nncf.config.extractors import extract_algo_specific_config
from nncf.tensorflow.api.compression import TFCompressionAlgorithmBuilder
from nncf.tensorflow.graph.converter import TFModelConverterFactory
from nncf.tensorflow.graph.metatypes.keras_layers import TFBatchNormalizationLayerMetatype
from nncf.tensorflow.graph.model_transformer import TFModelTransformer
from nncf.tensorflow.graph.transformations.commands import TFLayerWeight
from nncf.tensorflow.graph.transformations.commands import TFInsertionCommand
from nncf.tensorflow.graph.transformations.layout import TFTransformationLayout
from nncf.tensorflow.graph.utils import get_layer_identifier
from nncf.tensorflow.graph.utils import get_nncf_operations
from nncf.tensorflow.tensor import TFNNCFTensor
from nncf.tensorflow.pruning.tensor_processor import TFNNCFPruningTensorProcessor
from nncf.tensorflow.pruning.operations import TFElementwisePruningOp
from nncf.tensorflow.pruning.operations import TFIdentityMaskForwardPruningOp
from nncf.tensorflow.pruning.operations import TF_PRUNING_OPERATOR_METATYPES
from nncf.tensorflow.pruning.utils import get_filter_axis
from nncf.tensorflow.pruning.utils import get_filters_num
from nncf.tensorflow.sparsity.magnitude.operation import BinaryMask
from nncf.tensorflow.sparsity.utils import strip_model_from_masks


class PrunedLayerInfo(PrunedLayerInfoBase):
    def __init__(self, node_name: NNCFNodeName, layer_name: str, node_id: int, is_depthwise: bool):
        super().__init__(node_name, node_id, is_depthwise)
        self.layer_name = layer_name


class BasePruningAlgoBuilder(TFCompressionAlgorithmBuilder):
    """
    Determines which modifications should be made to the original model in
    order to enable pruning during fine-tuning.
    """

    def __init__(self, config: NNCFConfig, should_init: bool = True):
        super().__init__(config, should_init)
        params = self._algo_config.get('params', {})
        self._params = params

        self._ignore_frozen_layers = True
        self._prune_first = params.get('prune_first_conv', False)
        self._prune_batch_norms = params.get('prune_batch_norms', True)
        self._prune_downsample_convs = params.get('prune_downsample_convs', False)

        self._prunable_types = self._get_op_types_of_pruned_layers()
        types_of_grouping_ops = self._get_types_of_grouping_ops()
        self._pruning_node_selector = PruningNodeSelector(TF_PRUNING_OPERATOR_METATYPES,
                                                          self._prunable_types,
                                                          types_of_grouping_ops,
                                                          self.ignored_scopes,
                                                          self.target_scopes,
                                                          self._prune_first,
                                                          self._prune_downsample_convs)

        self._pruned_layer_groups_info = None
        self._graph = None
        self._op_names = []

    def apply_to(self, model: tf.keras.Model) -> tf.keras.Model:
        """
        Adds pruning masks to the model.

        :param model: The original uncompressed model.
        :return: The model with pruning masks.
        """
        transformer = TFModelTransformer(model)
        transformation_layout = self.get_transformation_layout(model)
        return transformer.transform(transformation_layout)

    def get_transformation_layout(self, model: tf.keras.Model) -> TFTransformationLayout:
        """
        Computes necessary model transformations (pruning mask insertions) to enable pruning.

        :param model: The original uncompressed model.
        :return: The instance of the `TransformationLayout` class containing
            a list of pruning mask insertions.
        """
        converter = TFModelConverterFactory.create(model)
        self._graph = converter.convert()
        groups_of_nodes_to_prune = self._pruning_node_selector.create_pruning_groups(self._graph)

        transformations = TFTransformationLayout()
        shared_layers = set()

        self._pruned_layer_groups_info = Clusterization[PrunedLayerInfo](lambda x: x.layer_name)

        for i, group in enumerate(groups_of_nodes_to_prune.get_all_clusters()):
            group_minfos = []
            for node in group.elements:
                layer_name = get_layer_identifier(node)
                layer = model.get_layer(layer_name)
                group_minfos.append(PrunedLayerInfo(node.node_name, layer_name, node.node_id,
                                                    is_prunable_depthwise_conv(node)))

                # Add output_mask to elements to run mask_propagation
                # and detect spec_nodes that will be pruned.
                # It should be done for all elements of shared layer.
                node.data['output_mask'] = TFNNCFTensor(tf.ones(get_output_channels(node)))
                if layer_name in shared_layers:
                    continue
                if node.is_shared():
                    shared_layers.add(layer_name)
                # Check that we need to prune weights in this op
                assert self._is_pruned_layer(layer)
                nncf_logger.info('Adding Weight Pruner in: %s', layer_name)

                _, layer_info = converter.get_layer_info_for_node(node.node_name)
                for weight_def in node.metatype.weight_definitions:
                    transformations.register(
                        self._get_insertion_command_binary_mask(
                            layer_info.layer_name, weight_def.weight_attr_name)
                    )
                if node.metatype.bias_attr_name is not None and \
                        getattr(layer, node.metatype.bias_attr_name) is not None:
                    transformations.register(
                        self._get_insertion_command_binary_mask(
                            layer_info.layer_name, node.metatype.bias_attr_name)
                    )

            cluster = Cluster[PrunedLayerInfo](i, group_minfos, [n.node_id for n in group.elements])
            self._pruned_layer_groups_info.add_cluster(cluster)

        # Propagating masks across the graph to detect spec_nodes that will be pruned
        mask_propagator = MaskPropagationAlgorithm(self._graph, TF_PRUNING_OPERATOR_METATYPES,
                                                   TFNNCFPruningTensorProcessor)
        mask_propagator.mask_propagation()

        # Add masks for all spec modules, because prunable batchnorm layers can be determined
        # at the moment of mask propagation
        types_spec_layers = [TFBatchNormalizationLayerMetatype] \
            if self._prune_batch_norms else []

        spec_nodes = self._graph.get_nodes_by_metatypes(types_spec_layers)
        for spec_node in spec_nodes:
            layer_name = get_layer_identifier(spec_node)
            layer = model.get_layer(layer_name)
            if spec_node.data['output_mask'] is None:
                # Skip elements that will not be pruned
                continue
            if layer_name in shared_layers:
                continue
            if spec_node.is_shared():
                shared_layers.add(layer_name)
            nncf_logger.info('Adding Weight Pruner in: %s', layer_name)

            _, layer_info = converter.get_layer_info_for_node(spec_node.node_name)
            for weight_def in spec_node.metatype.weight_definitions:
                if spec_node.metatype is TFBatchNormalizationLayerMetatype \
                        and not layer.scale and weight_def.weight_attr_name == 'gamma':
                    nncf_logger.debug('Fused gamma parameter encountered in BatchNormalization layer. '
                                      'Do not add mask to it.')
                    continue

                transformations.register(
                    self._get_insertion_command_binary_mask(
                        layer_info.layer_name, weight_def.weight_attr_name)
                )
            transformations.register(
                self._get_insertion_command_binary_mask(
                    layer_info.layer_name, spec_node.metatype.bias_attr_name)
            )
        return transformations

    def initialize(self, model: tf.keras.Model) -> None:
        pass

    def _get_insertion_command_binary_mask(self, layer_name: str,
                                           attr_name: str) -> TFInsertionCommand:
        op_name = self._get_pruning_operation_name(layer_name, attr_name)
        self._op_names.append(op_name)

        return TFInsertionCommand(
            target_point=TFLayerWeight(layer_name, attr_name),
            callable_object=BinaryMask(op_name),
            priority=TransformationPriority.PRUNING_PRIORITY
        )

    @staticmethod
    def _get_bn_for_node(node: NNCFNode, bn_nodes: List[NNCFNode]) -> Tuple[bool, List[NNCFNode]]:
        is_finished = False
        propagating_ops = [op_name for meta_op in [TFIdentityMaskForwardPruningOp, TFElementwisePruningOp]
                           for op_name in meta_op.get_all_op_aliases()]
        if node.node_type == 'BatchNormalization':
            is_finished = True
            bn_nodes.append(node)
        elif node.node_type not in propagating_ops:
            is_finished = True
        return is_finished, bn_nodes

    def _get_related_batchnorms(self, layer_name: str, group: Cluster, graph: NNCFGraph) -> List[NNCFNode]:
        """
        Returns List of batchnorm elements related to the layer.
        Note: Single node per layer for shared bactchnorm layers
        """
        layer_nodes = [node_ for node_ in group.elements
                       if node_.layer_name == layer_name]
        bn_nodes = []
        bn_layer_names = []
        for layer_node in layer_nodes:
            for next_node in graph.get_next_nodes(layer_node):
                for bn_node in graph.traverse_graph(next_node, self._get_bn_for_node):
                    bn_layer_name = get_layer_identifier(bn_node)
                    if bn_layer_name not in bn_layer_names:
                        bn_layer_names.append(bn_layer_name)
                        bn_nodes.append(bn_node)
        return bn_nodes

    def _is_pruned_layer(self, layer: tf.keras.layers.Layer) -> bool:
        """
        Return whether this layer should be pruned or not.
        """
        raise NotImplementedError

    def _get_op_types_of_pruned_layers(self) -> List[str]:
        """
        Returns list of operation types that should be pruned.
        """
        raise NotImplementedError

    def _get_types_of_grouping_ops(self) -> List[str]:
        raise NotImplementedError

    def _get_pruning_operation_name(self, layer_name: str, weight_attr_name: str) -> str:
        return f'{layer_name}_{weight_attr_name}_pruning_binary_mask'


class BasePruningAlgoController(BaseCompressionAlgorithmController):
    """
    Serves as a handle to the additional modules, parameters and hooks inserted
    into the original uncompressed model to enable pruning.
    """

    def __init__(self,
                 target_model: tf.keras.Model,
                 op_names: List[str],
                 prunable_types: List[str],
                 pruned_layer_groups_info: Clusterization[PrunedLayerInfo],
                 config):
        super().__init__(target_model)
        self._op_names = op_names
        self._prunable_types = prunable_types
        self.config = config
        self.pruning_config = extract_algo_specific_config(config,
                                                           "filter_pruning")
        params = self.pruning_config.get('params', {})
        self.pruning_init = self.pruning_config.get('pruning_init', 0)
        self.pruning_level = self.pruning_init
        self._pruned_layer_groups_info = pruned_layer_groups_info
        self.prune_flops = False
        self._check_pruning_level(params)
        self._num_of_sparse_elements_by_node = None

    def freeze(self):
        raise NotImplementedError

    def set_pruning_level(self, pruning_level: float):
        raise NotImplementedError

    def step(self, next_step):
        pass

    def _check_pruning_level(self, params):
        """
        Check that set only one of pruning target params
        """
        pruning_target = params.get('pruning_target', None)
        pruning_flops_target = params.get('pruning_flops_target', None)
        if pruning_target and pruning_flops_target:
            raise ValueError('Only one parameter from \'pruning_target\' and \'pruning_flops_target\' can be set.')
        if pruning_flops_target:
            self.prune_flops = True

    def _calculate_num_of_sparse_elements_by_node(self) -> Dict[NNCFNodeName, int]:
        """Returns the number of sparse elements per node. Take into account names ('^') for the shared ops."""
        if self._num_of_sparse_elements_by_node is None:
            self._calculate_pruned_layers_summary()

        retval = {}
        for group in self._pruned_layer_groups_info.get_all_clusters():
            for node in group.elements:
                retval[node.node_name] = self._num_of_sparse_elements_by_node[node.layer_name]
        return retval

    def _calculate_pruned_layers_summary(self) -> List[PrunedLayerSummary]:
        pruning_levels = []
        mask_names = []
        weights_shapes = []
        mask_shapes = []
        self._num_of_sparse_elements_by_node = {}
        for wrapped_layer, weight_attr, op_name in get_nncf_operations(self._model, self._op_names):
            mask = wrapped_layer.ops_weights[op_name.name]['mask']
            mask_names.append(mask.name)
            weights_shapes.append(list(mask.shape))
            reduce_axes = list(range(len(mask.shape)))
            filter_axis = get_filter_axis(wrapped_layer, weight_attr)
            if filter_axis == -1:
                filter_axis = reduce_axes[filter_axis]
            reduce_axes.remove(filter_axis)
            filter_mask = tf.reduce_max(tf.cast(mask, tf.int32), axis=reduce_axes, keepdims=True)
            mask_shapes.append(list(filter_mask.shape))
            filters_number = get_filters_num(wrapped_layer)
            pruned_filters_number = filters_number - tf.reduce_sum(filter_mask)
            pruning_levels.append(pruned_filters_number / filters_number)
            pruned_filter_number = filters_number - tf.reduce_sum(filter_mask)
            self._num_of_sparse_elements_by_node[wrapped_layer.name] = pruned_filter_number.numpy()

        pruning_levels = tf.keras.backend.batch_get_value(pruning_levels)
        mask_pruning = list(zip(mask_names, weights_shapes, mask_shapes, pruning_levels))

        pruned_layers_summary = []
        for mask_name, weights_shape, mask_shape, pruning_level in mask_pruning:
            pruned_layers_summary.append(PrunedLayerSummary(mask_name, weights_shape, mask_shape, pruning_level))

        return pruned_layers_summary

    def strip_model(self, model: tf.keras.Model) -> tf.keras.Model:
        return strip_model_from_masks(model, self._op_names)
