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

from typing import List, Optional, Tuple, TypeVar

import numpy as np

from nncf import Dataset
from nncf.common.factory import CommandCreatorFactory
from nncf.common.factory import ModelTransformerFactory
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.patterns import GraphPattern
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.graph.utils import get_reduction_axes
from nncf.common.logging import nncf_logger
from nncf.common.logging.track_progress import track
from nncf.common.tensor_statistics.statistic_point import StatisticPoint
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.quantization.algorithms.algorithm import Algorithm
from nncf.quantization.algorithms.channel_alignment.backend import ChannelAlignmentAlgoBackend
from nncf.quantization.algorithms.channel_alignment.backend import LayoutDescriptor

TModel = TypeVar("TModel")


class ChannelAlignment(Algorithm):
    """
    Post-training ChannelAlignment algorithm implementation.

    The main purpose of this algorithm to reduce quantization error
    via correction the parameters of the Convolutions, FullyConnected and their biases.
    Algorithm consists of following steps:
        - algorithm is searching for convolution -> convolution pairs in the target model.
        - minimal and maximal activations quantiles of first convolutions are collected on the target subset.
        - algorithm calculates median of collected values, it is used then to adjust
            convolution layers biases and weights.
        - biases of matched subgraphs convolutions are adjusted, so mean points of first
            convolution activations quantile medians are translated to zero.
        - weights of matched subgraph convolutions are adjusted, so all first convolutions activations
            which were between median of low quantile and median of high quantile are translated to [-1, 1] range.
    In case processed network has one or more convolution -> convolution pairs, activations of the first convolution
    become more quantization friendly as, in most cases activations mean is equal to zero and
    most activations values are in range [-1, 1].
    """

    def __init__(
        self,
        subset_size: int = 100,
        inplace_statistics: bool = True,
    ):
        """
        :param subset_size: Size of a subset for the statistics collection,
            defaults to 100.
        :param inplace_statistics: Defines wheather to calculate quantizers statistics
            by backend graph operations or by default Python implementation, defaults
            to True.
        """
        super().__init__()
        self.subset_size = subset_size
        self.inplace_statistics = inplace_statistics
        self._backend_entity = None
        self._quantile = 1e-4
        self._algorithm_key = f"CA_{hash(self)}"

    @property
    def available_backends(self) -> List[BackendType]:
        return [BackendType.OPENVINO]

    def _set_backend_entity(self, model: TModel) -> None:
        """
        Creates a helper class with a backed-specific logic of the algorithm.

        :param model: Backend-specific input model.
        """
        model_backend = get_backend(model)
        if model_backend == BackendType.OPENVINO:
            from nncf.quantization.algorithms.channel_alignment.openvino_backend import OVChannelAlignmentAlgoBackend

            self._backend_entity = OVChannelAlignmentAlgoBackend()

    def apply(
        self,
        model: TModel,
        graph: NNCFGraph,
        statistic_points: Optional[StatisticPointsContainer] = None,
        dataset: Optional[Dataset] = None,
    ) -> TModel:
        self._set_backend_entity(model)
        model_transformer = ModelTransformerFactory.create(model)
        transformation_layout = TransformationLayout()

        def filter_func(point: StatisticPoint) -> bool:
            return self._algorithm_key in point.algorithm_to_tensor_collectors and point.target_point == target_point

        for conv_in, add_in, conv_out in track(self._get_node_pairs(graph), description="Channel alignment"):
            target_point, node_in = self._get_target_point_and_node_in(conv_in, add_in)
            tensor_collectors = list(
                statistic_points.get_algo_statistics_for_node(node_in.node_name, filter_func, self._algorithm_key)
            )
            assert len(tensor_collectors) == 1
            stat = tensor_collectors[0].get_statistics()
            if stat.min_values is None or stat.max_values is None:
                nncf_logger.debug(
                    f"Skipping channel alignment for pairs {conv_in.node_name}, {conv_out.node_name} "
                    "because statistics were not collected for this pair."
                )
                continue

            conv_in_cont = ConvParamsContainer(conv_in, model, graph, self._backend_entity)
            conv_out_cont = ConvParamsContainer(conv_out, model, graph, self._backend_entity)
            if (
                conv_in_cont.dims.conv_weight_out_channels_dim is None
                or conv_out_cont.dims.conv_weight_out_channels_dim is None
            ):
                nncf_logger.debug(
                    f"Skipping channel alignment for pairs {conv_in.node_name}, {conv_out.node_name} "
                    " because one of the node is 1D MatMul, 1D Matmuls are not supported by CA algortihm yet."
                )
                continue

            amean = (stat.max_values + stat.min_values) * 0.5
            conv_in_cont.bias, conv_out_cont.bias = self._align_means(
                conv_in_cont.bias,
                conv_out_cont.bias,
                conv_out_cont.weight,
                amean,
                conv_out_cont.dims,
            )

            ascale = (stat.max_values - stat.min_values).astype(np.float32)
            eps = np.finfo(ascale.dtype).eps
            if (ascale > eps).any():
                conv_in_cont.weight, conv_out_cont.weight, conv_in_cont.bias = self._align_scales(
                    conv_in_cont.weight,
                    conv_out_cont.weight,
                    conv_in_cont.bias,
                    ascale,
                    conv_in_cont.dims,
                    conv_out_cont.dims,
                    eps,
                )

            command_creator = CommandCreatorFactory.create(model)
            for container in [conv_in_cont, conv_out_cont]:
                if container.stated_weight.is_modified():
                    transformation_layout.register(
                        command_creator.create_command_to_update_weight(
                            container.op, container.weight, container.weight_port_id
                        )
                    )

                if container.stated_bias.is_modified():
                    if container.bias_op_exist():
                        command = command_creator.create_command_to_update_bias(container.op, container.bias, graph)
                    else:
                        command = command_creator.create_command_to_insert_bias(container.op, container.bias)
                    transformation_layout.register(command)

        transformed_model = model_transformer.transform(transformation_layout)
        return transformed_model

    @staticmethod
    def _align_means(
        bias_in_value: np.ndarray,
        bias_out_value: np.ndarray,
        conv_out_value: np.ndarray,
        amean: np.ndarray,
        conv_out_descr: LayoutDescriptor,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Function which calculates new add_in_value and add_out_value
        in ChannelAlignment pattern, so output activations of the second convolution bias
        are the same, but the first convolution bias is shifted with minus by amean value.

        :param bias_in_value: Bias of the first convolution in the ChannelAlignment pattern.
        :param bias_out_value: Bias of the second convolution in the ChannelAlignment pattern.
        :param amean: Mean value to shift first and second convolutions biases.
        :param conv_out_descr: The second convolution weights layout descriptor.
        """
        updated_add_in_value = bias_in_value - amean.reshape(bias_in_value.shape)

        weight_dims = conv_out_value.ndim
        updated_conv_out_value = conv_out_value
        if weight_dims > 2:
            axes = list(range(weight_dims))
            axes.remove(conv_out_descr.conv_weight_in_channels_dim)
            axes.remove(conv_out_descr.conv_weight_out_channels_dim)
            updated_conv_out_value = np.sum(conv_out_value, axis=tuple(axes))

        out_channel_dim, in_channel_dim = 0, 1
        if conv_out_descr.conv_weight_out_channels_dim > conv_out_descr.conv_weight_in_channels_dim:
            out_channel_dim, in_channel_dim = in_channel_dim, out_channel_dim

        updated_conv_out_value = np.transpose(
            updated_conv_out_value,
            (out_channel_dim, in_channel_dim),
        )
        shift = updated_conv_out_value.dot(amean.reshape(updated_conv_out_value.shape[1]))

        updated_add_out_value = bias_out_value + shift.reshape(bias_out_value.shape)
        return updated_add_in_value, updated_add_out_value

    @staticmethod
    def _align_scales(
        conv_in_value: np.ndarray,
        conv_out_value: np.ndarray,
        bias_in_value: Optional[np.ndarray],
        ascale: np.ndarray,
        conv_in_descr: LayoutDescriptor,
        conv_out_descr: LayoutDescriptor,
        eps: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Function which calculates new conv_in_value, conv_out_value and bias_in_value
        in ChannelAlignment pattern, so output activations of conv_out are the same,
        but activations of conv_in are scale times smaller. Negative scales are skipped,
        too small (<1e-2) and too big (>1e2) scales are clamped.

        :param conv_in_value: Weights of the first convolution in the ChannelAlignment pattern.
        :param conv_out_value: Weights of the second convolution in the ChannelAlignment pattern.
        :param bias_in_value: Bias of the first convolution in the ChannelAlignment pattern. Could be None.
        :param ascale: Scale value to apply to convolutions weights.
        :param conv_in_descr: The first convolution weights layout descriptor.
        :param conv_out_descr: The second convolution weights layout descriptor.
        :param eps: Minimal significant value > 0 for convolution weights and biases precision.
        """
        conv_in_shape = conv_in_value.shape
        # TODO(dlyakhov) support group convolutions with groups number not in [1, out_channels]
        if conv_in_shape[conv_in_descr.conv_weight_out_channels_dim] != ascale.shape[conv_in_descr.bias_channels_dim]:
            return conv_in_value, conv_out_value, bias_in_value

        positive_scales_mask = ascale > eps
        scale_factor = ascale / np.median(ascale[positive_scales_mask])
        scale_factor[~positive_scales_mask] = 1
        scale_factor = np.clip(scale_factor, 1e-2, 1e2)

        scale_in_shape = np.ones(len(conv_in_shape), dtype=int)
        scale_in_shape[conv_in_descr.conv_weight_out_channels_dim] = scale_factor.shape[conv_in_descr.bias_channels_dim]
        updated_conv_in_value = conv_in_value / scale_factor.reshape(scale_in_shape)

        updated_bias_in_value = bias_in_value / scale_factor.reshape(bias_in_value.shape)

        scale_out_shape = np.ones(len(conv_out_value.shape), dtype=int)
        scale_out_shape[conv_out_descr.conv_weight_in_channels_dim] = scale_factor.shape[
            conv_in_descr.bias_channels_dim
        ]
        updated_conv_out_value = conv_out_value * scale_factor.reshape(scale_out_shape)
        return updated_conv_in_value, updated_conv_out_value, updated_bias_in_value

    def _check_consumer_conv_node(self, conv_node: NNCFNode) -> bool:
        attrs = self._backend_entity.get_conv_layer_attributes(conv_node)
        if attrs is None:
            return False
        # Check groups amount == 1
        if attrs.groups != 1:
            return False
        # Check node has no padding
        if any(attrs.padding_values):
            return False
        # Check node has valid stride
        if any(elem != 1 for elem in attrs.stride):
            return False
        # Check Node has valid dilation
        if any(elem != 1 for elem in attrs.dilations):
            return False
        return True

    def _check_producer_conv_node(self, conv_node: NNCFNode):
        return conv_node.layer_attributes is not None

    def _get_target_patterns(self) -> GraphPattern:
        input_attrs = {
            GraphPattern.LABEL_ATTR: "INPUT",
            GraphPattern.METATYPE_ATTR: GraphPattern.NON_PATTERN_NODE_TYPE,
        }
        producer_attrs = {
            GraphPattern.LABEL_ATTR: "CONV_PRODUCER",
            GraphPattern.NODE_TYPE_ATTR: self._backend_entity.get_conv_metatypes()
            + self._backend_entity.get_linear_metatypes(),
        }
        bias_attrs = {
            GraphPattern.LABEL_ATTR: "BIAS_PRODUCER",
            GraphPattern.NODE_TYPE_ATTR: self._backend_entity.get_add_metatypes(),
        }
        bias_const_attrs = {
            GraphPattern.LABEL_ATTR: "BIAS_CONSTANT",
            GraphPattern.METATYPE_ATTR: GraphPattern.NON_PATTERN_NODE_TYPE,
        }
        consumer_attrs = {
            GraphPattern.LABEL_ATTR: "CONV_CONSUMER",
            GraphPattern.NODE_TYPE_ATTR: self._backend_entity.get_conv_metatypes(),
        }
        conv_const_attrs = {
            GraphPattern.LABEL_ATTR: "CONV_CONSTANT",
            GraphPattern.METATYPE_ATTR: GraphPattern.NON_PATTERN_NODE_TYPE,
        }

        use_constant = True

        def get_conv_conv_pattern() -> GraphPattern:
            conv_conv = GraphPattern()
            if use_constant:
                input_node = conv_conv.add_node(**input_attrs)
                producer_constant = conv_conv.add_node(**conv_const_attrs)
                consumer_constant = conv_conv.add_node(**conv_const_attrs)

            pattern_conv_producer = conv_conv.add_node(**producer_attrs)
            pattern_conv_consumer = conv_conv.add_node(**consumer_attrs)

            if use_constant:
                conv_conv.add_edge(input_node, pattern_conv_producer)
                conv_conv.add_edge(producer_constant, pattern_conv_producer)
                conv_conv.add_edge(consumer_constant, pattern_conv_consumer)

            conv_conv.add_edge(pattern_conv_producer, pattern_conv_consumer)
            return conv_conv

        def get_conv_add_conv_pattern() -> GraphPattern:
            conv_bias_conv = GraphPattern()
            if use_constant:
                input_node = conv_bias_conv.add_node(**input_attrs)
                producer_constant = conv_bias_conv.add_node(**conv_const_attrs)
                bias_producer_const = conv_bias_conv.add_node(**bias_const_attrs)
                consumer_constant = conv_bias_conv.add_node(**conv_const_attrs)

            pattern_conv_producer = conv_bias_conv.add_node(**producer_attrs)
            pattern_bias_producer = conv_bias_conv.add_node(**bias_attrs)
            pattern_conv_consumer = conv_bias_conv.add_node(**consumer_attrs)

            if use_constant:
                conv_bias_conv.add_edge(input_node, pattern_conv_producer)
                conv_bias_conv.add_edge(producer_constant, pattern_conv_producer)
                conv_bias_conv.add_edge(consumer_constant, pattern_conv_consumer)
                conv_bias_conv.add_edge(bias_producer_const, pattern_bias_producer)

            conv_bias_conv.add_edge(pattern_conv_producer, pattern_bias_producer)
            conv_bias_conv.add_edge(pattern_bias_producer, pattern_conv_consumer)
            return conv_bias_conv

        pattern = get_conv_conv_pattern()
        pattern.add_pattern_alternative(get_conv_add_conv_pattern())
        return pattern

    def _get_node_pairs(self, nncf_graph: NNCFGraph) -> List[Tuple[NNCFNode, Optional[NNCFNode], NNCFNode]]:
        pairs = []
        patterns = self._get_target_patterns()
        for subgraph in nncf_graph.find_matching_subgraphs(patterns):
            if len(subgraph) == 2:
                add_in = None
                conv_in, conv_out = subgraph
            else:
                conv_in, add_in, conv_out = subgraph

            if not self._check_producer_conv_node(conv_in):
                continue

            if not self._check_consumer_conv_node(conv_out):
                continue

            pairs.append((conv_in, add_in, conv_out))
        return pairs

    def _get_target_point_and_node_in(self, conv_in, add_in) -> Tuple[TargetPoint, NNCFNode]:
        node_in = conv_in if add_in is None else add_in
        input_port_id, _ = self._backend_entity.get_activation_port_ids_for_node(node_in)
        return (
            self._backend_entity.target_point(TargetType.POST_LAYER_OPERATION, node_in.node_name, input_port_id),
            node_in,
        )

    def get_statistic_points(self, model: TModel, graph: NNCFGraph) -> StatisticPointsContainer:
        self._set_backend_entity(model)

        statistic_container = StatisticPointsContainer()
        for conv_in, add_in, _ in self._get_node_pairs(graph):
            target_point, node_in = self._get_target_point_and_node_in(conv_in, add_in)

            channel_axis = conv_in.metatype.output_channel_axis
            activation_shape = list(range(len(graph.get_output_edges(node_in)[0].tensor_shape)))
            reduction_axes = get_reduction_axes([0, channel_axis], activation_shape)

            statistic_collector = self._backend_entity.get_statistic_collector(
                reduction_axes, self._quantile, self.subset_size, self.inplace_statistics
            )
            statistic_container.add_statistic_point(
                StatisticPoint(
                    target_point=target_point,
                    tensor_collector=statistic_collector,
                    algorithm=self._algorithm_key,
                )
            )

        return statistic_container


class StatedTensor:
    """
    Tensor wrapper with additional method is_modified which is true if
    given tensor was modified at least once after the initialization.
    """

    def __init__(self, value: np.ndarray):
        """
        :param value: Tensor to wrap.
        """
        self._value = value
        self._mod_times = 0

    @property
    def val(self):
        return self._value

    @val.setter
    def val(self, value):
        if self._value is None and value is None:
            return
        self._mod_times += 1
        self._value = value

    def is_modified(self) -> bool:
        """
        :return: True if wrapped tensor was changed at least once after the
            initialization else False.
        """
        return self._mod_times > 0


class ConvParamsContainer:
    """
    Convolution container class which is incapsulating common convolutional parameters collection.
    """

    def __init__(
        self, conv_op: NNCFNode, model: TModel, nncf_graph: NNCFGraph, backend_entity: ChannelAlignmentAlgoBackend
    ):
        """
        :param conv_op: NNCF conv node.
        :param model: Backend-specific model instance.
        :param nncf_graph: NNCFGraph of given backend-specific model.
        :param backend_entity: Current backend entity to retrieve parameters from given conv node
        """
        _, self._weights_port_id = backend_entity.get_weights_port_ids_for_node(conv_op)
        self.stated_weight = StatedTensor(backend_entity.get_weight_value(conv_op, model, self._weights_port_id))
        self._bias_op_exist = False
        if backend_entity.is_node_with_bias(conv_op, nncf_graph):
            bias = backend_entity.get_bias_value(conv_op, model, nncf_graph)
            self._bias_op_exist = True
        else:
            bias = backend_entity.create_bias_tensor(conv_op, nncf_graph, 0)
        self.stated_bias = StatedTensor(bias)
        self._op = conv_op
        self._dims = backend_entity.get_dims_descriptor(conv_op)

    @property
    def weight(self):
        return self.stated_weight.val

    @weight.setter
    def weight(self, value):
        self.stated_weight.val = value

    @property
    def bias(self):
        return self.stated_bias.val

    @bias.setter
    def bias(self, value):
        self.stated_bias.val = value

    @property
    def op(self):
        return self._op

    @property
    def weight_port_id(self):
        return self._weights_port_id

    @property
    def dims(self) -> LayoutDescriptor:
        return self._dims

    def bias_op_exist(self) -> bool:
        return self._bias_op_exist
