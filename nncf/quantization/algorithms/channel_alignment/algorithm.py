# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

import numpy as np
from tqdm import tqdm

from nncf import Dataset
from nncf.common.factory import EngineFactory
from nncf.common.factory import ModelTransformerFactory
from nncf.common.factory import NNCFGraphFactory
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.layer_attributes import ConvolutionLayerAttributes
from nncf.common.graph.model_transformer import ModelTransformer
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.logging import nncf_logger
from nncf.common.tensor_statistics.statistic_point import StatisticPoint
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.tensor_statistics.statistics import MinMaxTensorStatistic
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.quantization.algorithms.algorithm import Algorithm
from nncf.quantization.algorithms.channel_alignment.backend import ChannelAlignmentAlgoBackend
from nncf.quantization.algorithms.channel_alignment.backend import ConvParamsContainer
from nncf.quantization.algorithms.channel_alignment.backend import DimsDescriptor
from nncf.quantization.algorithms.fast_bias_correction.backend import ALGO_BACKENDS

TModel = TypeVar("TModel")
TTensor = TypeVar("TTensor")

FAST_BIAS_CORRECTION_THRESHOLD = 2


class ChannelAlignment(Algorithm):
    """
    Post-training FastBiasCorrection algorithm implementation.

    The main purpose of this algorithm to reduce quantization error
    via correction the bias of the Convolutions, FullyConnected, etc. layers.
    The algorithm pipeline is very simple:
        - we collects floating-point statistics from the corresponding model for the layers with bias;
        - then we gets the quantized model and try to reduce it's error by correction of the bias;
        - the shift calculates using the sub-graph that consists of the correction layer and
        weight quantizer-dequantizer pair or fake quantize node;
        - the floating-point statistics uses as input for
        the sub-graph and further quantization output calculation;
        - in the end we corrects the original bias by the difference (shift)
        between floating-point and quantized outputs.
    """

    def __init__(
        self,
        subset_size: int = 100,
        inplace_statistics: bool = True,
        backend_params: Optional[Dict[str, Any]] = None,
    ):
        """
        :param subset_size: Size of a subset for the statistics collection,
            defaults to 100.
        :param threshold: The magnitude threshold that regulates the application of the
            shift. Magnitude calculates as the maximum of the absolute ratio of the
            shift to the original bias value. If the calculated value is less than the
            threshold, the shift will apply to the bias, defaults to 2.
        :param apply_for_all_nodes: If True, then the bias correction be applied to all
            quantized nodes, if the node has no bias then a bias node will be inserted,
            and if False, then the bias correction will only be applied to quantized
            nodes that have a bias.
        :param inplace_statistics: Defines wheather to calculate quantizers statistics
            by backend graph operations or by default Python implementation, defaults
            to True.
        :param backend_params: Backend specific parameters.
        """
        super().__init__()
        self.subset_size = subset_size
        self.inplace_statistics = inplace_statistics
        self.backend_params = backend_params
        self.nncf_graph = None
        self._backend_entity = None
        self._nncf_grpah = None
        self._q = 1e-4

    @property
    def available_backends(self) -> Dict[str, BackendType]:
        return ALGO_BACKENDS.registry_dict

    def _set_backend_entity(self, model: TModel) -> None:
        """
        Creates a helper class with a backed-specific logic of the algorithm.

        :param model: Backend-specific input model.
        """
        model_backend = get_backend(model)
        if model_backend == BackendType.OPENVINO:
            from nncf.quantization.algorithms.channel_alignment.openvino_backend import OVChannelAlignmentAlgoBackend

            self._backend_entity = OVChannelAlignmentAlgoBackend()

    def _apply(
        self,
        model: TModel,
        statistic_points: Optional[StatisticPointsContainer] = None,
        dataset: Optional[Dataset] = None,
    ) -> TModel:
        self._set_backend_entity(model)

        nncf_graph = NNCFGraphFactory.create(model) if self.nncf_graph is None else self.nncf_graph
        model_transformer = ModelTransformerFactory.create(model)
        transformation_layout = TransformationLayout()

        def filter_func(point: StatisticPoint) -> bool:
            return ChannelAlignment in point.algorithm_to_tensor_collectors and point.target_point == target_point

        for conv_in, add_in, conv_out in tqdm(self._get_node_pairs(nncf_graph), desc="Channel allignment"):
            target_point, node_in = self._get_target_point_and_node_in(conv_in, add_in)
            tensor_collectors = list(
                statistic_points.get_algo_statistics_for_node(node_in.node_name, filter_func, ChannelAlignment)
            )
            assert len(tensor_collectors) == 1
            stat: MinMaxTensorStatistic = tensor_collectors[0].get_statistics()

            conv_in_cont = ConvParamsContainer(conv_in, model, nncf_graph, self._backend_entity)
            conv_out_cont = ConvParamsContainer(conv_out, model, nncf_graph, self._backend_entity)
            dims_descriptor: DimsDescriptor = self._backend_entity.get_dims_descriptor(conv_in)
            if conv_in_cont.has_bias() and conv_out_cont.has_bias():
                amean = (stat.max_values + stat.min_values) * 0.5
                conv_in_cont.bias, conv_out_cont.bias = self._align_means(
                    conv_in_cont.bias, conv_out_cont.bias, conv_out_cont.weight, amean, dims_descriptor
                )

            ascale = stat.max_values - stat.min_values
            eps = np.finfo(ascale.dtype).eps
            if (ascale > eps).any():
                conv_in_cont.weight, conv_out_cont.weight, conv_in_cont.bias = self._align_scales(
                    conv_in_cont.weight,
                    conv_out_cont.weight,
                    conv_in_cont.bias,
                    ascale,
                    dims_descriptor,
                    eps,
                )

            for container in [conv_in_cont, conv_out_cont]:
                if not np.equal(container.weight, container.original_weight).all():
                    transformation_layout.register(
                        self._backend_entity.create_weights_update_command(
                            container.op, container.weight, container.weight_port_id
                        )
                    )

                if not np.equal(container.bias, container.original_bias).all():
                    transformation_layout.register(
                        self._backend_entity.create_bias_update_command(container.op, container.bias, nncf_graph)
                    )

        transformed_model = model_transformer.transform(transformation_layout)
        return transformed_model

    @staticmethod
    def _align_means(bias_in_value, bias_out_value, conv_out_value, amean, dims_descriptor: DimsDescriptor):
        updated_add_in_value = bias_in_value - amean.reshape(bias_in_value.shape)

        weight_dims = len(conv_out_value.shape)
        updated_conv_out_value = conv_out_value
        if weight_dims > 2:
            axes = list(range(weight_dims))
            axes.remove(dims_descriptor.conv_weight_in_channels_dim)
            axes.remove(dims_descriptor.conv_weight_out_channels_dim)
            updated_conv_out_value = np.sum(conv_out_value, axis=tuple(axes))
        updated_conv_out_value = np.transpose(
            updated_conv_out_value,
            (dims_descriptor.conv_weight_out_channels_dim, dims_descriptor.conv_weight_in_channels_dim),
        )
        shift = updated_conv_out_value.dot(
            amean.reshape(updated_conv_out_value.shape[dims_descriptor.conv_weight_in_channels_dim])
        )

        updated_add_out_value = bias_out_value + shift.reshape(bias_out_value.shape)
        return updated_add_in_value, updated_add_out_value

    @staticmethod
    def _align_scales(conv_in_value, conv_out_value, bias_in_value, ascale, dims_descr: DimsDescriptor, eps):
        # scale producer convolution weights
        conv_in_shape = conv_in_value.shape
        if conv_in_shape[dims_descr.conv_weight_out_channels_dim] == ascale.shape[dims_descr.bias_channels_dim]:
            positive_scales_mask = ascale > eps
            scale_factor = ascale / np.median(ascale[positive_scales_mask])
            scale_factor[~positive_scales_mask] = 1
            scale_factor = np.clip(scale_factor, 1e-2, 1e2)

            scale_in_shape = np.ones(len(conv_in_shape), dtype=int)
            scale_in_shape[dims_descr.conv_weight_out_channels_dim] = scale_factor.shape[dims_descr.bias_channels_dim]
            conv_in_value = conv_in_value / scale_factor.reshape(scale_in_shape)

            if bias_in_value is not None:
                bias_in_value = bias_in_value / scale_factor.reshape(bias_in_value.shape)

            scale_out_shape = np.ones(len(conv_out_value.shape), dtype=int)
            scale_out_shape[dims_descr.conv_weight_in_channels_dim] = scale_factor.shape[dims_descr.bias_channels_dim]
            conv_out_value = conv_out_value * scale_factor.reshape(scale_out_shape)
        return conv_in_value, conv_out_value, bias_in_value

    def _check_consumer_conv_node(self, conv_node: NNCFNode):
        if conv_node is None:
            return False
        attrs: ConvolutionLayerAttributes = self._backend_entity.get_conv_layer_attributes(conv_node)
        # Check groups amount == 1
        if attrs.groups != 1:
            return False
        # Check node has no padding
        if any(attrs.padding_values):
            return False
        # Check node has valid stride
        if any(elem != 1 for elem in attrs.stride):
            return False
        # Check Node has vaild dilation
        if any(elem != 1 for elem in attrs.dilations):
            return False
        return True

    def _check_producer_node(self, conv_node, add_node, nncf_graph):
        # Check node exists
        if conv_node is None:
            return False
        # Check node is conv
        if not self._backend_entity.is_node_conv_or_matmul_operation(conv_node):
            return False
        # Check conv has only one consumer node
        if len(nncf_graph.get_next_nodes(conv_node)) > 1:
            return False
        # Check add has only one consumer node
        if add_node is not None and len(nncf_graph.get_next_nodes(add_node)) > 1:
            return False
        return True

    def _get_node_pairs(self, nncf_graph: NNCFGraph):
        # Return conv pairs that correspond to
        # Conv -> Add -> Conv pattern
        def get_previous_node(node, port_id):
            if node is None:
                return None

            input_edges = nncf_graph.get_input_edges(node)
            input_edges = [edge for edge in input_edges if edge.input_port_id == port_id]
            if len(input_edges) > 1:
                return None
            return input_edges[0].from_node

        pairs = []
        for conv_out in self._backend_entity.get_conv_nodes(nncf_graph):
            if not self._check_consumer_conv_node(conv_out):
                continue

            conv_in = get_previous_node(conv_out, 0)
            if conv_in is None:
                continue

            add_in = None
            if self._backend_entity.is_node_add_operation(conv_in):
                add_in = conv_in
                conv_in = get_previous_node(add_in, 0)

            if not self._check_producer_node(conv_in, add_in, nncf_graph):
                continue

            pairs.append((conv_in, add_in, conv_out))
        return pairs

    def _get_target_point_and_node_in(self, conv_in, add_in):
        node_in = conv_in if add_in is None else add_in
        input_port_id, _ = self._backend_entity.get_activation_port_ids_for_node(node_in)
        return (
            self._backend_entity.target_point(TargetType.POST_LAYER_OPERATION, node_in.node_name, input_port_id),
            node_in,
        )

    def get_statistic_points(self, model: TModel) -> StatisticPointsContainer:
        self._set_backend_entity(model)
        self.nncf_graph = NNCFGraphFactory.create(model)

        statistic_container = StatisticPointsContainer()
        for conv_in, add_in, _ in self._get_node_pairs(self.nncf_graph):
            target_point, node_in = self._get_target_point_and_node_in(conv_in, add_in)
            channel_axis = conv_in.metatype.output_channel_axis
            reduction_shape = list(range(len(self.nncf_graph.get_output_edges(node_in)[0].tensor_shape)))
            reduction_shape.remove(channel_axis)

            statistic_collector = self._backend_entity.get_statistic_collector(
                tuple(reduction_shape), self._q, self.subset_size, self.inplace_statistics
            )
            statistic_container.add_statistic_point(
                StatisticPoint(
                    target_point=target_point,
                    tensor_collector=statistic_collector,
                    algorithm=ChannelAlignment,
                )
            )

        return statistic_container
