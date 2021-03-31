"""
 Copyright (c) 2021 Intel Corporation
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

from beta.nncf.tensorflow.api.compression import TFCompressionAlgorithmController
from beta.nncf.tensorflow.api.compression import TFCompressionAlgorithmBuilder
from beta.nncf.tensorflow.graph.converter import convert_keras_model_to_nncf_graph
from beta.nncf.tensorflow.graph.model_transformer import TFModelTransformer
from beta.nncf.tensorflow.pruning.export_helpers import TFElementwise
from beta.nncf.tensorflow.pruning.export_helpers import TFIdentityMaskForwardOps
from beta.nncf.tensorflow.graph.transformations.commands import TFLayerWeight
from beta.nncf.tensorflow.graph.transformations.commands import TFInsertionCommand
from beta.nncf.tensorflow.graph.transformations.layout import TFTransformationLayout
from beta.nncf.tensorflow.graph.utils import get_layer_identifier
from beta.nncf.tensorflow.graph.utils import collect_wrapped_layers
from beta.nncf.tensorflow.layers.common import BIAS_ATTR_NAME
from beta.nncf.tensorflow.layers.common import LAYERS_WITH_WEIGHTS
from beta.nncf.tensorflow.layers.common import SPECIAL_LAYERS_WITH_WEIGHTS
from beta.nncf.tensorflow.layers.common import WEIGHT_ATTR_NAME
from beta.nncf.tensorflow.pruning.utils import get_filter_axis
from beta.nncf.tensorflow.pruning.utils import get_filters_num
from beta.nncf.tensorflow.pruning.utils import is_shared
from beta.nncf.tensorflow.sparsity.magnitude.operation import BinaryMask
from beta.nncf.tensorflow.sparsity.utils import convert_raw_to_printable
from beta.nncf.tensorflow.sparsity.utils import strip_model_from_masks
from beta.nncf.tensorflow.pruning.export_helpers import TF_PRUNING_OPERATOR_METATYPES
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.common.pruning.pruning_node_selector import PruningNodeSelector
from nncf.common.pruning.model_analysis import NodesCluster
from nncf.common.pruning.model_analysis import Clusterization
from nncf.common.utils.logger import logger as nncf_logger


class PrunedLayerInfo:
    BN_LAYER_NAME = 'bn_layer'

    def __init__(self, layer_name: str, related_layers: Dict[str, List[str]]):
        self.layer_name = layer_name
        self.related_layers = related_layers


class BasePruningAlgoBuilder(TFCompressionAlgorithmBuilder):
    """
    Determines which modifications should be made to the original model in
    order to enable pruning during fine-tuning.
    """

    def __init__(self, config):
        super().__init__(config)
        params = config.get('params', {})
        self._params = params
        self.ignored_scopes = self.config.get('ignored_scopes', [])
        self.target_scopes = self.config.get('target_scopes', [])

        self._ignore_frozen_layers = True
        self._prune_first = params.get('prune_first_conv', False)
        self._prune_last = params.get('prune_last_conv', False)
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
                                                          self._prune_last,
                                                          self._prune_downsample_convs)

        self._pruned_layer_groups_info = None

    def apply_to(self, model: tf.keras.Model) -> tf.keras.Model:
        """
        Adds pruning masks to the model.

        :param model: The original uncompressed model.
        :return: The model with pruning masks.
        """
        transformation_layout = self.get_transformation_layout(model)
        return TFModelTransformer(model, transformation_layout).transform()

    def get_transformation_layout(self, model: tf.keras.Model) -> TFTransformationLayout:
        """
        Computes necessary model transformations (pruning mask insertions) to enable pruning.

        :param model: The original uncompressed model.
        :return: The instance of the `TransformationLayout` class containing
            a list of pruning mask insertions.
        """
        graph = convert_keras_model_to_nncf_graph(model)
        groups_of_nodes_to_prune = self._pruning_node_selector.create_pruning_groups(graph)

        transformations = TFTransformationLayout()
        shared_layers = set()

        self._pruned_layer_groups_info = Clusterization('layer_name')

        for i, group in enumerate(groups_of_nodes_to_prune.get_all_clusters()):
            group_minfos = []
            for node in group.nodes:
                layer_name = get_layer_identifier(node)
                if layer_name in shared_layers:
                    continue
                if is_shared(node):
                    shared_layers.add(layer_name)
                layer = model.get_layer(layer_name)
                # Check that we need to prune weights in this op
                assert self._is_pruned_layer(layer)

                nncf_logger.info('Adding Weight Pruner in: %s', layer_name)
                for attr_name_key in [WEIGHT_ATTR_NAME, BIAS_ATTR_NAME]:
                    attr_name = LAYERS_WITH_WEIGHTS[node.node_type][attr_name_key]
                    if getattr(layer, attr_name) is not None:
                        transformations.register(
                            TFInsertionCommand(
                                target_point=TFLayerWeight(layer_name, attr_name),
                                callable_object=BinaryMask(),
                                priority=TransformationPriority.PRUNING_PRIORITY
                            ))

                related_layers = {}
                if self._prune_batch_norms:
                    bn_nodes = self._get_related_batchnorms(layer_name, group, graph)
                    for bn_node in bn_nodes:
                        bn_layer_name = get_layer_identifier(bn_node)
                        for attr_name_key in [WEIGHT_ATTR_NAME, BIAS_ATTR_NAME]:
                            attr_name = SPECIAL_LAYERS_WITH_WEIGHTS[bn_node.node_type][attr_name_key]
                            transformations.register(
                                TFInsertionCommand(
                                    target_point=TFLayerWeight(bn_layer_name, attr_name),
                                    callable_object=BinaryMask(),
                                    priority=TransformationPriority.PRUNING_PRIORITY
                                ))
                        if PrunedLayerInfo.BN_LAYER_NAME in related_layers:
                            related_layers[PrunedLayerInfo.BN_LAYER_NAME].append(bn_layer_name)
                        else:
                            related_layers[PrunedLayerInfo.BN_LAYER_NAME] = [bn_layer_name]

                minfo = PrunedLayerInfo(layer_name, related_layers)
                group_minfos.append(minfo)
            cluster = NodesCluster(i, group_minfos, [n.node_id for n in group.nodes])
            self._pruned_layer_groups_info.add_cluster(cluster)

        return transformations

    @staticmethod
    def _get_bn_for_node(node: NNCFNode, bn_nodes: List[NNCFNode]) -> Tuple[bool, List[NNCFNode]]:
        is_finished = False
        propagating_ops = [op_name for meta_op in [TFIdentityMaskForwardOps, TFElementwise]
                           for op_name in meta_op.get_all_op_aliases()]
        if node.node_type == 'BatchNormalization':
            is_finished = True
            bn_nodes.append(node)
        elif node.node_type not in propagating_ops:
            is_finished = True
        return is_finished, bn_nodes

    def _get_related_batchnorms(self, layer_name: str, group: NodesCluster, graph: NNCFGraph) -> List[NNCFNode]:
        """
        Returns List of batchnorm nodes related to the layer.
        Note: Single node per layer for shared bactchnorm layers
        """
        layer_nodes = [node_ for node_ in group.nodes
                       if get_layer_identifier(node_) == layer_name]
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


class BasePruningAlgoController(TFCompressionAlgorithmController):
    """
    Serves as a handle to the additional modules, parameters and hooks inserted
    into the original uncompressed model to enable pruning.
    """

    def __init__(self,
                 target_model: tf.keras.Model,
                 prunable_types: List[str],
                 pruned_layer_groups_info: Clusterization,
                 config):
        super().__init__(target_model)
        self._prunable_types = prunable_types
        self.config = config
        params = self.config.get('params', {})
        self.pruning_init = config.get('pruning_init', 0)
        self.pruning_rate = self.pruning_init
        self._pruned_layer_groups_info = pruned_layer_groups_info
        self.prune_flops = False
        self._check_pruning_rate(params)

    def freeze(self):
        raise NotImplementedError

    def set_pruning_rate(self, pruning_rate: float):
        raise NotImplementedError

    def step(self, next_step):
        pass

    def _check_pruning_rate(self, params):
        """
        Check that set only one of pruning target params
        """
        pruning_target = params.get('pruning_target', None)
        pruning_flops_target = params.get('pruning_flops_target', None)
        if pruning_target and pruning_flops_target:
            raise ValueError('Only one parameter from \'pruning_target\' and \'pruning_flops_target\' can be set.')
        if pruning_flops_target:
            raise Exception('Pruning by flops is not supported in NNCF TensorFlow yet.')

    def statistics(self, quickly_collected_only=False) -> Dict[str, object]:
        raw_pruning_statistics = self.raw_statistics()
        prefix = 'pruning'
        header = ['Name', 'Weight\'s Shape', 'Mask Shape', 'PR']
        return convert_raw_to_printable(raw_pruning_statistics, prefix, header)

    def raw_statistics(self) -> Dict[str, object]:
        raw_pruning_statistics = {}
        pruning_rates = []
        mask_names = []
        weights_shapes = []
        mask_shapes = []
        wrapped_layers = collect_wrapped_layers(self._model)
        for wrapped_layer in wrapped_layers:
            for weight_attr, ops in wrapped_layer.weights_attr_ops.items():
                for op_name, op in ops.items():
                    if isinstance(op, BinaryMask):
                        mask = wrapped_layer.ops_weights[op_name]['mask']
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
                        pruning_rates.append(pruned_filters_number / filters_number)

        raw_pruning_statistics.update({
            'pruning_rate': self.pruning_rate
        })

        pruning_rates = tf.keras.backend.batch_get_value(pruning_rates)

        mask_pruning = list(zip(mask_names, weights_shapes, mask_shapes, pruning_rates))
        raw_pruning_statistics['pruning_statistic_by_layer'] = []
        for mask_name, weights_shape, mask_shape, pruning_rate in mask_pruning:
            raw_pruning_statistics['pruning_statistic_by_layer'].append({
                'Name': mask_name,
                'Weight\'s Shape': weights_shape,
                'Mask Shape': mask_shape,
                'PR': pruning_rate
            })

        return raw_pruning_statistics

    def strip_model(self, model: tf.keras.Model) -> tf.keras.Model:
        return strip_model_from_masks(model)
