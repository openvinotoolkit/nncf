# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import dataclasses
from copy import deepcopy
from typing import List, Optional, OrderedDict, Tuple, TypeVar

import nncf
import nncf.tensor.functions as fns
from nncf import Dataset
from nncf.common.factory import ModelTransformerFactory
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.logging import nncf_logger
from nncf.common.quantization.initialization.range import RangeInitCollectorParams
from nncf.common.quantization.quantizer_setup import SingleConfigQuantizationPoint
from nncf.common.quantization.quantizer_setup import SingleConfigQuantizerSetup
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import QuantizerGroup
from nncf.common.tensor_statistics.collectors import TensorStatisticCollectorBase
from nncf.common.tensor_statistics.statistic_point import StatisticPoint
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.experimental.common.quantization.algorithms.quantizer.quantizer import NNCFQuantizer
from nncf.experimental.common.tensor_statistics.statistics import MinMaxTensorStatistic
from nncf.quantization.advanced_parameters import changes_asdict
from nncf.quantization.algorithms.algorithm import Algorithm
from nncf.quantization.fake_quantize import calculate_quantizer_parameters
from nncf.quantization.fake_quantize import get_quantizer_narrow_range
from nncf.quantization.range_estimator import RangeEstimatorParameters
from nncf.quantization.range_estimator import RangeEstimatorParametersSet

TModel = TypeVar("TModel")


class MinMaxRangeEstimator(Algorithm):
    def __init__(
        self,
        quantizer: NNCFQuantizer,
        subset_size: int = 300,
        inplace_statistics: bool = True,
        batchwise_statistics: bool = False,
        activations_range_estimator_params: Optional[RangeEstimatorParameters] = None,
        weights_range_estimator_params: Optional[RangeEstimatorParameters] = None,
    ):
        """
        :param subset_size: Size of a subset to calculate activations statistics used
            for quantization, defaults to 300.
        :param inplace_statistics: Defines wheather to calculate quantizers statistics
            by backend graph operations or by default Python implementation, defaults
            to True.
        :param batchwise_statistics: Determines whether quantizer statistics should be calculated
            for each item of the batch or for the entire batch, default is False.
        :param activations_range_estimator_params: Quantization range estimation
            parameters for activation.
        :param weights_range_estimator_params: Quantization range estimation parameters
            for weights.
        """
        self._quantizer = quantizer
        self._subset_size = subset_size
        self._inplace_statistics = inplace_statistics
        self._batchwise_statistics = batchwise_statistics
        self._activations_range_estimator_params = activations_range_estimator_params
        self._weights_range_estimator_params = weights_range_estimator_params

        self._range_estimator_params = {
            QuantizerGroup.WEIGHTS: self._weights_range_estimator_params,
            QuantizerGroup.ACTIVATIONS: self._activations_range_estimator_params,
        }
        # Calculates global quantizer constraints
        self._reset_cache()
        self._algorithm_key = f"MMQ_{hash(self)}"

    def _reset_cache(self) -> None:
        """
        Marks cache by noninitialized values. Needs to be called when the new quantizer setup is needed.
        """
        self._quantization_target_points_to_qconfig: OrderedDict[TargetPoint, QuantizerConfig] = None
        self._unified_scale_groups = None

    def _init_cache(self) -> None:
        """
        Initializes cache.
        """
        self._quantization_target_points_to_qconfig: OrderedDict[TargetPoint, QuantizerConfig] = (
            collections.OrderedDict()
        )
        self._unified_scale_groups = []

    @property
    def available_backends(self) -> List[BackendType]:
        return [BackendType.TORCH_FX]

    def _set_backend_entity(self, model: TModel) -> None:
        """
        Creates a helper class with a backed-specific logic of the algorithm

        :param model: backend-specific input model
        """
        model_backend = get_backend(model)
        if model_backend == BackendType.TORCH_FX:
            from nncf.experimental.common.quantization.algorithms.range_estimator.torch_fx_backend import (
                FXRangeEstimatorAlgoBackend,
            )

            self._backend_entity = FXRangeEstimatorAlgoBackend()
        else:
            raise nncf.UnsupportedBackendError(
                "Cannot return backend-specific entity because {} is not supported!".format(model_backend.value)
            )

    def _get_range_estimator_parameters(
        self, target_point: TargetPoint, quantizer_config: QuantizerConfig
    ) -> RangeEstimatorParameters:
        """
        Returns range estimator parameters.

        :param target_point: Quantizer target point.
        :param quantizer_config: Quantizer config.
        :return: Range estimator parameters.
        """
        quantizer_group = QuantizerGroup.ACTIVATIONS
        if target_point.is_weight_target_point():
            quantizer_group = QuantizerGroup.WEIGHTS

        if quantizer_group == QuantizerGroup.WEIGHTS or (
            quantizer_group == QuantizerGroup.ACTIVATIONS and quantizer_config.per_channel
        ):
            params = RangeEstimatorParametersSet.MINMAX
        else:
            params = RangeEstimatorParametersSet.MEAN_MINMAX

        user_params = self._range_estimator_params[quantizer_group]
        if user_params is None:
            return deepcopy(params)

        min_changes = changes_asdict(user_params.min)
        min_statistic_collector = dataclasses.replace(params.min, **min_changes)

        max_changes = changes_asdict(user_params.max)
        max_statistic_collector = dataclasses.replace(params.max, **max_changes)

        return RangeEstimatorParameters(min_statistic_collector, max_statistic_collector)

    def _get_stat_collector(
        self,
        graph: NNCFGraph,
        target_point: TargetPoint,
        qconfig: QuantizerConfig,
        batchwise_statistics: bool,
    ) -> TensorStatisticCollectorBase:
        """
        Creates and returns a statistic collector based on the quantizer's configuration.

        :param graph: NNCFGraph instance.
        :param target_point: Target point indicates where statistics should be collected.
        :param qconfig: Configuration of a quantizer layer,
        defining the configuration of created statistic collector.
        :param batchwise_statistics: Determines whether quantizer statistics should be calculated
            for each item of the batch or for the entire batch.
        :return: Statistic Collector.
        """
        is_weight = target_point.is_weight_target_point()
        node = graph.get_node_by_name(target_point.target_node_name)
        shape = self._backend_entity.get_target_point_shape(graph, node, target_point)
        range_estimator_params = self._get_range_estimator_parameters(target_point, qconfig)

        channel_axes = ()
        if qconfig.per_channel:
            channel_axes = (
                self._backend_entity.get_weight_quantization_axes(node, target_point, len(shape)) if is_weight else (1,)
            )

        # Weight statistics is constant, so only one collection is enough.
        num_samples = self._subset_size if not is_weight else 1

        batchwise_statistics = batchwise_statistics and not is_weight

        collector_params = RangeInitCollectorParams(
            is_weights=is_weight, scheme=qconfig.mode, per_channel=qconfig.per_channel
        )
        reduction_axes, aggregation_axes = None, None
        if shape is not None:
            reduction_axes, aggregation_axes = collector_params.get_reduction_aggregation_axes(
                shape, channel_axes, batchwise_statistics
            )

        return self._backend_entity.get_statistic_collector(
            range_estimator_params,
            collector_params.use_abs_max,
            reduction_axes,
            aggregation_axes,
            self._inplace_statistics,
            num_samples=num_samples,
        )

    def _add_weight_quantization_target_point(
        self, quantization_point: SingleConfigQuantizationPoint, nncf_graph: NNCFGraph
    ) -> None:
        """
        Adds weight quantization target point to the set of existing points.

        :param quantization_point: SingleConfigQuantizationPoint for the needed layer.
        :param nncf_graph: The built NNCFGraph of the model.
        """
        weight_quantization_target_points = self._get_weight_quantization_target_points(quantization_point, nncf_graph)
        for weight_quantization_target_point in weight_quantization_target_points:
            self._quantization_target_points_to_qconfig[weight_quantization_target_point] = quantization_point.qconfig

    def _add_activation_quantization_target_point(
        self, quantization_point: SingleConfigQuantizationPoint, nncf_graph: NNCFGraph
    ) -> None:
        """
        Adds activation quantization target point to the set of existing points.

        :param quantization_point: SingleConfigQuantizationPoint for the needed layer.
        :param nncf_graph: NNCFGraph instance for working with the graph and nodes.
        """
        activation_quantization_target_point = self._get_activation_quantization_target_point(
            quantization_point, nncf_graph
        )
        self._quantization_target_points_to_qconfig[activation_quantization_target_point] = quantization_point.qconfig

    def _get_weight_quantization_target_points(
        self, quantization_point: SingleConfigQuantizationPoint, nncf_graph: NNCFGraph
    ) -> List[SingleConfigQuantizationPoint]:
        """
        Returns weight quantization target points to the set of existing points.

        :param quantization_point: SingleConfigQuantizationPoint for the needed layer.
        :param nncf_graph: NNCFGraph instance for working with the graph and nodes.
        :return: List of SingleConfigQuantizationPoints for the needed layer.
        """
        weight_quantization_target_points = []
        node_name = quantization_point.insertion_point.target_node_name
        node = nncf_graph.get_node_by_name(node_name)
        weights_port_ids = self._backend_entity.get_weight_tensor_port_ids(node, nncf_graph)
        for port_id in weights_port_ids:
            weight_quantization_target_points.append(
                self._backend_entity.target_point(TargetType.OPERATION_WITH_WEIGHTS, node_name, port_id)
            )
        return weight_quantization_target_points

    def _get_activation_quantization_target_point(
        self, quantization_point: SingleConfigQuantizationPoint, nncf_graph: NNCFGraph
    ) -> SingleConfigQuantizationPoint:
        """
        Returns activation quantization target point to the set of existing points.

        :param quantization_point: SingleConfigQuantizationPoint for the needed layer.
        :param nncf_graph: NNCFGraph instance for working with the graph and nodes.
        :return: SingleConfigQuantizationPoint for the needed layer.
        """
        node_name = quantization_point.insertion_point.target_node_name
        # If Quantization of node's input
        if quantization_point.insertion_point.input_port_id is not None:
            input_port_id = quantization_point.insertion_point.input_port_id
            activation_quantization_target_point = self._backend_entity.target_point(
                TargetType.PRE_LAYER_OPERATION, node_name, input_port_id
            )
        # If quantization of node's output or Model Input node
        else:
            # NOTE: Assumes that the operation has output edges only from one output port because
            # we haven't encountered a model with operations that have multiple output edges with different
            # output port IDs. Currently, such models are not supported. Usually, `output_port_id = 0` is used.
            # However, there are operations, such as LSTMSequence, where the `output_port_id` changes from case
            # to case. Therefore, the code below is required to dynamically determine the `output_port_id` where
            # the quantize operation should be inserted."
            node = nncf_graph.get_node_by_name(node_name)
            unique_output_port_ids = set(e.output_port_id for e in nncf_graph.get_output_edges(node))
            if len(unique_output_port_ids) > 1:
                nncf_logger.warning(
                    f"Cannot determine the output_port_id for the operation: {node_name}, "
                    "output_port_id = 0 will be used."
                )
                output_port_id = 0
            else:
                output_port_id = next(iter(unique_output_port_ids))

            activation_quantization_target_point = self._backend_entity.target_point(
                TargetType.POST_LAYER_OPERATION, node_name, output_port_id
            )
        return activation_quantization_target_point

    def _find_quantization_target_points(
        self, model: TModel, nncf_graph: NNCFGraph
    ) -> Tuple[OrderedDict[TargetPoint, QuantizerConfig], List[List[TargetPoint]]]:
        """
        Initializes a cache, finds quantization target points and them puts in the cache.

        :param model: Backend-specific model, for which Quantization Target Points are being seek.
        :param nncf_graph: NNCFGraph instance.
        :return: Mapping of quantization target points with associated quantization configuration,
        along with target points for scale unification.
        """
        quantizer_setup = self._quantizer.get_quantization_setup(model, nncf_graph)
        self._unified_scale_groups = self._collect_unified_groups(quantizer_setup, nncf_graph)
        quantization_points = list(quantizer_setup.quantization_points.values())
        quantization_points = self._topological_sort_quantization_points(quantization_points, nncf_graph)
        for quantization_point in quantization_points:
            if quantization_point.is_weight_quantization_point():
                self._add_weight_quantization_target_point(quantization_point, nncf_graph)
            elif quantization_point.is_activation_quantization_point():
                self._add_activation_quantization_target_point(quantization_point, nncf_graph)
            else:
                raise nncf.InternalError("Incorrect quantization point")
        return self._quantization_target_points_to_qconfig, self._unified_scale_groups

    def _get_quantization_target_points(
        self, model: TModel, nncf_graph: NNCFGraph
    ) -> Tuple[OrderedDict[TargetPoint, QuantizerConfig], List[List[TargetPoint]]]:
        """
        Returns Quantization Target Points.
        Returns a cache with target points if exists. Otherwise, initiates a procedure of finding them.

        :param model: Backend-specific model, for which Quantization Target Points are being seek.
        :param nncf_graph: NNCFGraph instance.
        :return: Mapping of quantization target points with associated quantization configuration,
        along with target points for scale unification.
        """
        if self._quantization_target_points_to_qconfig is not None:
            return self._quantization_target_points_to_qconfig, self._unified_scale_groups
        self._init_cache()
        return self._find_quantization_target_points(model, nncf_graph)

    def _collect_unified_groups(
        self, quantizer_setup: SingleConfigQuantizerSetup, nncf_graph: NNCFGraph
    ) -> List[List[TargetPoint]]:
        """
        Collects the group of quantizers for unification.

        :param quantizer_setup: SingleConfigQuantizerSetup instance.
        :param nncf_graph: NNCFGraph instance.
        :return: List with the groups of the TargetPoints.
        """
        unified_scale_groups = []
        for quantizer_ids in quantizer_setup.unified_scale_groups.values():
            unified_scale_group = []
            for quantizer_id in quantizer_ids:
                quantization_point = quantizer_setup.quantization_points[quantizer_id]

                # Only activation quantizers can be unified
                if quantization_point.is_activation_quantization_point():
                    activation_target_point = self._get_activation_quantization_target_point(
                        quantization_point, nncf_graph
                    )
                    unified_scale_group.append(activation_target_point)
                else:
                    weight_target_points = self._get_weight_quantization_target_points(quantization_point, nncf_graph)
                    for weight_target_point in weight_target_points:
                        unified_scale_group.append(weight_target_point)
            unified_scale_groups.append(unified_scale_group)
        return unified_scale_groups

    def _topological_sort_quantization_points(
        self, quantization_points: List[SingleConfigQuantizationPoint], nncf_graph: NNCFGraph
    ) -> List[SingleConfigQuantizationPoint]:
        """
        Sorts quantization_points based on the topological order of nodes obtained form nncf_graph.

        :param quantization_points: Quantization points.
        :param nncf_graph: Instance of NNCFgraph used to get topological sort.
        :return: Sorted quantization_points.
        """
        node_names_to_pos = {node.node_name: i for i, node in enumerate(nncf_graph.topological_sort())}
        quantization_points.sort(key=lambda point: node_names_to_pos[point.insertion_point.target_node_name])
        return quantization_points

    def apply(
        self,
        model: TModel,
        graph: NNCFGraph,
        statistic_points: Optional[StatisticPointsContainer] = None,
        dataset: Optional[Dataset] = None,
    ) -> TModel:
        transformation_layout = TransformationLayout()
        model_transformer = ModelTransformerFactory.create(model)
        quantization_target_points, unified_scale_groups = self._get_quantization_target_points(model, graph)
        weight_layer_names = set()

        def filter_func(point: StatisticPoint) -> bool:
            return (
                self._algorithm_key in point.algorithm_to_tensor_collectors
                and point.target_point == quantization_target_point
            )

        unified_ops_list = set()
        for unified_scale_group in unified_scale_groups:
            group_statistics = []
            for quantization_target_point in unified_scale_group:
                target_node_name = quantization_target_point.target_node_name
                for tensor_collector in statistic_points.get_algo_statistics_for_node(
                    target_node_name, filter_func, self._algorithm_key
                ):
                    statistics = tensor_collector.get_statistics()
                    if statistics.min_values is None or statistics.max_values is None:
                        raise nncf.InternalError(f"Statistics were not collected for the node {target_node_name}")
                    group_statistics.append(statistics)

            unified_values = self._unify_statistics(group_statistics)
            qconfigs = [quantization_target_points[qtp] for qtp in unified_scale_group]
            if any(qconfigs[0] != qconfig for qconfig in qconfigs[1:]):
                raise nncf.InternalError(f"QConfigs for unified scale group {unified_scale_group} are not equal")
            qconfig = qconfigs[0]
            q_group = QuantizerGroup.ACTIVATIONS
            narrow_range = get_quantizer_narrow_range(qconfig, q_group)
            parameters = calculate_quantizer_parameters(unified_values, qconfig, q_group, narrow_range)
            commands = self._backend_entity.create_unified_scales_quantizers_insertion_commands(
                graph, unified_scale_group, qconfig, parameters
            )
            for command in commands:
                transformation_layout.register(command)
            unified_ops_list.update(unified_scale_group)

        for quantization_target_point, qconfig in quantization_target_points.items():
            if quantization_target_point in unified_ops_list:
                continue
            target_node_name = quantization_target_point.target_node_name
            for tensor_collector in statistic_points.get_algo_statistics_for_node(
                target_node_name, filter_func, self._algorithm_key
            ):
                if quantization_target_point.is_weight_target_point():
                    weights_name = self._backend_entity.get_weight_name(graph, quantization_target_point)
                    if not self._backend_entity.should_quantize_weight(weights_name, weight_layer_names):
                        continue
                    weight_layer_names.add(weights_name)
                    quant_group = QuantizerGroup.WEIGHTS
                else:
                    quant_group = QuantizerGroup.ACTIVATIONS

                half_range = False
                narrow_range = get_quantizer_narrow_range(qconfig, quant_group)
                statistics = tensor_collector.get_statistics()
                if statistics.min_values is None or statistics.max_values is None:
                    raise nncf.InternalError(f"Statistics were not collected for the node {target_node_name}")
                parameters = calculate_quantizer_parameters(statistics, qconfig, quant_group, narrow_range, half_range)
                command = self._backend_entity.create_quantizer_insertion_command(
                    graph, quantization_target_point, qconfig, parameters
                )
                transformation_layout.register(command)
        if not transformation_layout.transformations:
            nncf_logger.info("The model has no operations to apply quantization.")
        quantized_model = model_transformer.transform(transformation_layout)
        return quantized_model

    def get_statistic_points(self, model: TModel, graph: NNCFGraph) -> StatisticPointsContainer:
        self._set_backend_entity(model)
        self._reset_cache()
        quantization_target_points, _ = self._get_quantization_target_points(model, graph)
        output = StatisticPointsContainer()
        for quantization_target_point, qconfig in quantization_target_points.items():
            nncf_logger.debug(
                f"Adding target point {quantization_target_point.target_node_name}"
                f" with type {quantization_target_point.type} for statistics collection"
            )
            stat_collector = self._get_stat_collector(
                graph, quantization_target_point, qconfig, self._batchwise_statistics
            )
            output.add_statistic_point(
                StatisticPoint(
                    target_point=quantization_target_point,
                    tensor_collector=stat_collector,
                    algorithm=self._algorithm_key,
                )
            )
        return output

    @staticmethod
    def _unify_statistics(statistics: List[MinMaxTensorStatistic]) -> MinMaxTensorStatistic:
        """
        Returns backend-specific unified statistics.

        :param statistics: List of MinMaxTensorStatistic instances.
        :return: Unified MinMaxTensorStatistic value.
        """

        max_values, min_values = [], []
        for statistic in statistics:
            max_values.append(statistic.max_values.flatten())
            min_values.append(statistic.min_values.flatten())
        max_values = fns.max(fns.stack(max_values), axis=0)
        min_values = fns.min(fns.stack(min_values), axis=0)
        return MinMaxTensorStatistic(min_values=min_values, max_values=max_values)
