"""
 Copyright (c) 2023 Intel Corporation
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

from copy import deepcopy
from typing import Dict, TypeVar, Optional, OrderedDict, Set, List
import collections

from nncf import Dataset
from nncf.scopes import IgnoredScope
from nncf.scopes import get_ignored_node_names_from_ignored_scope
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.parameters import TargetDevice
from nncf.parameters import ModelType
from nncf.common.hardware.config import get_hw_config_type
from nncf.common.insertion_point_graph import InsertionPointGraph
from nncf.common.quantization.quantizer_propagation.solver import QuantizerPropagationSolver
from nncf.common.quantization.quantizer_setup import SingleConfigQuantizationPoint
from nncf.common.quantization.quantizer_setup import SingleConfigQuantizerSetup
from nncf.common.quantization.structs import QuantizableWeightedLayerNode
from nncf.common.quantization.structs import QuantizationMode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import QuantizationPreset
from nncf.common.quantization.structs import QuantizerGroup
from nncf.common.quantization.structs import QuantizationConstraints
from nncf.common.quantization.config_assignment import assign_qconfig_lists_to_modules
from nncf.common.tensor_statistics.collectors import TensorStatisticCollectorBase
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.common.logging import nncf_logger
from nncf.common.graph.patterns import GraphPattern
from nncf.common.graph.patterns.manager import PatternsManager
from nncf.common.graph.operator_metatypes import OperatorMetatype

from nncf.quantization.algorithms.algorithm import Algorithm
from nncf.quantization.algorithms.algorithm import AlgorithmParameters
from nncf.quantization.algorithms.min_max.backend import ALGO_BACKENDS
from nncf.quantization.algorithms.definitions import RangeType
from nncf.quantization.algorithms.definitions import Granularity
from nncf.quantization.algorithms.definitions import OverflowFix
from nncf.quantization.passes import transform_to_inference_graph
from nncf.quantization.fake_quantize import get_quantizer_narrow_range
from nncf.quantization.fake_quantize import calculate_quantizer_parameters
from nncf.common.factory import NNCFGraphFactory
from nncf.common.tensor_statistics.statistic_point import StatisticPoint
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.factory import ModelTransformerFactory

TModel = TypeVar('TModel')


def _filter_target_points_by_metatypes(quantization_target_points: Set[TargetPoint],
                                       metatypes: List[OperatorMetatype],
                                       nncf_graph: NNCFGraph) -> Set[TargetPoint]:
    """Returns TargetPoints which are suited to a node having metatype specified in 'metatypes'.

    :param quantization_target_points: TargetPoints to be filtered.
    :param metatypes: Metatypes that pass filtering.
    :param nncf_graph: Instance of NNCFgraph used to get node by name.
    :return: Filtered TargetPoints.
    """
    output = set()
    for quantization_target_point in quantization_target_points:
        node = nncf_graph.get_node_by_name(quantization_target_point.target_node_name)
        if node.metatype in metatypes:
            output.add(quantization_target_point)
    return output


class MinMaxQuantizationParameters(AlgorithmParameters):
    """
    Base class of MinMaxQuantization algorithm parameters.
    """

    DEFAULT_QCONFIG = QuantizerConfig(num_bits=8,
                                      mode=QuantizationMode.SYMMETRIC,
                                      signedness_to_force=None,
                                      per_channel=False)

    def __init__(self,
                 number_samples: int = 300,
                 preset: QuantizationPreset = QuantizationPreset.PERFORMANCE,
                 weight_bits: Optional[int] = None,
                 weight_granularity: Optional[Granularity] = None,
                 signed_weights: Optional[bool] = None,
                 activation_bits: Optional[int] = None,
                 activation_granularity: Optional[Granularity] = None,
                 signed_activations: Optional[bool] = None,
                 target_device: TargetDevice = TargetDevice.ANY,
                 range_type: Optional[RangeType] = None,
                 quantize_outputs: bool = False,
                 ignored_scopes: Optional[IgnoredScope] = None,
                 model_type: Optional[ModelType] = None,
                 overflow_fix: OverflowFix = OverflowFix.FIRST_LAYER,
                 inplace_statistics: bool = True,
                 ):
        """
        :param number_samples: Number of samples for the statistics collection.
        :param preset: Preset parameter for Quantization.
            Defines the mode: symmetric or asymmetric of the activation quantizers.
        :param weight_bits: Bitwidth for the weight quantizers.
        :param weight_granularity: Type of quantization granularity for weight quantizers.
            Could be per-channel or per-tensor.
        :param signed_weights: Defines whether the datatype of the weight quantizers should be forced.
            True if the quantizer *must* be signed, False if *must* be unsigned,
            None if the signed/unsigned attribute should be determined based on the incoming activation
            statistics during range initialization.
        :param activation_bits: Bitwidth for the activation quantizers.
        :param activation_granularity: Type of quantization granularity for activation quantizers.
            Could be per-channel or per-tensor.
        :param signed_activations: Defines whether the datatype of the activation quantizers
            should be forced. True if the quantizer *must* be signed, False if *must* be unsigned,
            None if the signed/unsigned attribute should be determined based on the incoming activation
            statistics during range initialization.
        :param target_device: Target device for the settings of the quantization pipeline.
        :param range_type: Type of statistics range calculation.
        :param quantize_outputs: Boolean value that says whether quantize outputs or not.
        :param ignored_scopes: Desrciptor of the layers which input must not be quantized.
        :param overflow_fix: This option controls whether to apply the overflow issue fix for the 8-bit quantization.
        :param inplace_statistics: Appliclable only for OpenVINO backend.
            Will be available for ONNX backend in future. Defines wheather to calculate quantizers statistics
            by backend graph operations or by default Python implementation.
            Statistics computated inplace tend to be calculated faster and with lower memory stamp.
        """
        self.number_samples = number_samples
        self.target_device = target_device
        self.range_type = range_type
        self.quantize_outputs = quantize_outputs
        self.inplace_statistics = inplace_statistics
        self.ignored_scopes = IgnoredScope() if ignored_scopes is None else ignored_scopes
        self.global_quantizer_constraints = {}
        if weight_granularity is not None:
            weight_granularity = weight_granularity == Granularity.PERCHANNEL
        if activation_granularity is not None:
            activation_granularity = activation_granularity == Granularity.PERCHANNEL
        q_weight_constraints = self._get_quantizer_constraints(QuantizerGroup.WEIGHTS, preset, weight_bits,
                                                               weight_granularity, signed_weights)
        q_activation_constraints = self._get_quantizer_constraints(QuantizerGroup.ACTIVATIONS, preset, activation_bits,
                                                                   activation_granularity,
                                                                   signed_activations)
        self.global_quantizer_constraints[QuantizerGroup.WEIGHTS] = q_weight_constraints
        self.global_quantizer_constraints[QuantizerGroup.ACTIVATIONS] = q_activation_constraints
        self.model_type = model_type
        self.overflow_fix = overflow_fix

    def _get_quantizer_constraints(self, group: QuantizerGroup, preset: QuantizationPreset, num_bits: Optional[int],
                                   per_channel: Optional[bool],
                                   signedness_to_force: Optional[bool]) -> QuantizationConstraints:
        """
        Returns QuantizationConstraints for the provided quantizer group.

        :param group: Quantizer group.
        :param preset: Quantization preset.
        :param num_bits: Bitwidth of quantizers.
        :param per_channel: Per-channel ot per-tensor granularity.
        :param signedness_to_force: True if the quantizer *must* be signed, False if *must* be unsigned,
            None if the signed/unsigned attribute should be determined based on the incoming activation
            statistics during range initialization.
        :return: QuantizationConstraints.
        """
        constraints = {'mode': preset.get_params_configured_by_preset(group)['mode']}
        if num_bits is not None:
            constraints['num_bits'] = num_bits
        if per_channel is not None:
            constraints['per_channel'] = per_channel
        if signedness_to_force is not None:
            constraints['signedness_to_force'] = signedness_to_force
        return QuantizationConstraints(**constraints)


class MinMaxQuantization(Algorithm):
    """
    Post-training MinMaxQuantization algorithm.

    The algorithm modifies the model by inserting additional nodes, which emulates the quantization of the data flow.
    The algorithm calibrates the parameters of the inserted nodes by collecting the statistics in the insertion points.
    The modified model is returned after the work of the algorithm, which can be perfomed via the original framework.
    It is expected that the inference of the obtained model in the int8 mode would be faster than the original model.
    """

    def __init__(self, parameters: MinMaxQuantizationParameters):
        self.nncf_graph = None
        # It prevents the duplicate weight quantizers from being added.
        # It can happen when you have layers that share the identical weight tensor.
        self._quantization_target_points_to_qconfig = \
            collections.OrderedDict()  # type: OrderedDict[TargetPoint, QuantizerConfig]
        self._parameters = parameters
        self._unified_scale_groups = []

    @property
    def available_backends(self) -> Dict[str, BackendType]:
        return ALGO_BACKENDS.registry_dict

    def _set_backend_entity(self, model: TModel) -> None:
        """
        Creates a helper class with a backed-specific logic of the algorithm

        :param model: backend-specific input model
        """
        model_backend = get_backend(model)
        if model_backend == BackendType.ONNX:
            from nncf.quantization.algorithms.min_max.onnx_backend import \
                ONNXMinMaxAlgoBackend
            self._backend_entity = ONNXMinMaxAlgoBackend()
        elif model_backend == BackendType.OPENVINO:
            from nncf.experimental.openvino_native.quantization.algorithms.min_max.openvino_backend import \
                OVMinMaxAlgoBackend
            self._backend_entity = OVMinMaxAlgoBackend()
        elif model_backend == BackendType.TORCH:
            from nncf.quantization.algorithms.min_max.torch_backend import \
                PTMinMaxAlgoBackend
            self._backend_entity = PTMinMaxAlgoBackend()
        else:
            raise RuntimeError('Cannot return backend-specific entity '
                               'because {} is not supported!'.format(model_backend))

    def _get_stat_collector(self, nncf_graph: NNCFGraph,
                            target_point: TargetPoint,
                            quantizer_config: QuantizerConfig) -> TensorStatisticCollectorBase:
        """
        Creates and returns statistic collector instance based on the quantizer's configuration.

        :param quantizer_config: QuantizerConfig instance for the current layer.
        :return: One of the TensorStatisticCollectorBase instances
        """

        def is_default_parameters(range_type: RangeType,
                                  quantizer_config: QuantizerConfig) -> bool:
            return range_type is None or quantizer_config.per_channel

        range_type = self._parameters.range_type
        if is_default_parameters(range_type, quantizer_config):
            if quantizer_config.per_channel:
                range_type = RangeType.MINMAX
            else:
                range_type = RangeType.MEAN_MINMAX

        # TODO: allow to use different range type estimators
        # for weight quantizers
        if target_point.is_weight_target_point():
            range_type = RangeType.MINMAX

        if range_type == RangeType.MINMAX:
            return self._backend_entity.minmax_statistic_collector(nncf_graph, target_point, quantizer_config,
                                                                   num_samples=self._parameters.number_samples,
                                                                   inplace=self._parameters.inplace_statistics)
        if range_type == RangeType.MEAN_MINMAX:
            return self._backend_entity.mean_minmax_statistic_collector(nncf_graph, target_point, quantizer_config,
                                                                        use_per_sample_stats=False,
                                                                        num_samples=self._parameters.number_samples,
                                                                        inplace=self._parameters.inplace_statistics)
        raise RuntimeError('This range type is not supported!')

    def _get_default_qconfig(self, constraints: QuantizationConstraints = None) -> QuantizerConfig:
        """
        Returns default quantizer configuration, based on the provided constraints.

        :param constraints: Quantization constraints.
        :return: Quantizer config.
        """
        qconfig = deepcopy(self._parameters.DEFAULT_QCONFIG)
        if constraints is not None:
            qconfig = constraints.apply_constraints_to(qconfig)
        return qconfig

    def _get_quantizer_setup(self, nncf_graph: NNCFGraph, pattern: GraphPattern) -> SingleConfigQuantizerSetup:
        """
        Returns SingleConfigQuantizerSetup instance based on the input NNCFGraph.

        :param nncf_graph: NNCFGraph instance.
        :param pattern: GraphPattern instance.
        :return: SingleConfigQuantizerSetup for the current NNCFGraph entity.
        """
        hw_config_type = get_hw_config_type(self._parameters.target_device.value)
        hw_config_path = self._backend_entity.hw_config.get_path_to_hw_config(hw_config_type)
        hw_config = self._backend_entity.hw_config.from_json(hw_config_path)
        model_type = self._parameters.model_type
        ignored_scopes = self._parameters.ignored_scopes

        ignored_names = get_ignored_node_names_from_ignored_scope(ignored_scopes, nncf_graph)
        model_type_ignore_scope = self._backend_entity.get_model_type_ignore_scope(model_type,
                                                                                   self._parameters.target_device)
        ignored_names = set(ignored_names + get_ignored_node_names_from_ignored_scope(model_type_ignore_scope,
                                                                                      nncf_graph, strict=False))

        weight_nodes = self._backend_entity.get_weight_nodes(nncf_graph)

        default_weight_qconfig = self._get_default_qconfig(
            self._parameters.global_quantizer_constraints[QuantizerGroup.WEIGHTS])
        weighted_node_and_qconf_lists = assign_qconfig_lists_to_modules(nodes_with_weights=weight_nodes,
                                                                        default_weight_qconfig=default_weight_qconfig,
                                                                        global_weight_constraints=
                                                                        self._parameters.global_quantizer_constraints[
                                                                            QuantizerGroup.WEIGHTS],
                                                                        scope_overrides_dict=None,
                                                                        hw_config=hw_config)
        quantizable_layer_nodes = [QuantizableWeightedLayerNode(node, qconf_list) for node, qconf_list
                                   in weighted_node_and_qconf_lists.items()]
        inference_nncf_graph = transform_to_inference_graph(deepcopy(nncf_graph),
                                                            self._backend_entity.shapeof_metatypes,
                                                            self._backend_entity.read_variable_metatypes)
        ip_graph = InsertionPointGraph(inference_nncf_graph)
        ip_graph = ip_graph.get_ip_graph_with_merged_hw_optimized_operations(pattern)
        post_processing_types = self._backend_entity.post_processing_metatypes
        solver = QuantizerPropagationSolver(activation_ignored_scopes=ignored_names,
                                            weight_ignored_scopes=ignored_names,
                                            hw_config=hw_config,
                                            default_trait_to_metatype_map=self._backend_entity.quant_trait_op_dict,
                                            default_qconfig_list=[self._get_default_qconfig(
                                                self._parameters.global_quantizer_constraints[
                                                    QuantizerGroup.ACTIVATIONS])],
                                            quantizable_layer_nodes=quantizable_layer_nodes,
                                            quantize_outputs=self._parameters.quantize_outputs,
                                            global_constraints=self._parameters.global_quantizer_constraints,
                                            post_processing_marker_metatypes=post_processing_types)

        quantization_proposal = solver.run_on_ip_graph(ip_graph)
        multi_config_setup = quantization_proposal.quantizer_setup
        single_config_setup = multi_config_setup.select_first_qconfig_for_each_point()
        finalized_proposal = quantization_proposal.finalize(single_config_setup)
        final_setup = solver.get_final_quantizer_setup(finalized_proposal)
        return final_setup

    def _add_weight_quantization_target_point(self, quantization_point: SingleConfigQuantizationPoint,
                                              nncf_graph: NNCFGraph) -> None:
        """
        Adds weight quantization target point to the set of existing points.

        :param quantization_point: SingleConfigQuantizationPoint for the needed layer.
        :param model: Model in the original framework.
        :param nncf_graph: The built NNCFGraph of the model.
        """
        node_name = quantization_point.insertion_point.target_node_name
        node = nncf_graph.get_node_by_name(node_name)
        weights_port_ids = self._backend_entity.get_weight_tensor_port_ids(node)
        for port_id in weights_port_ids:
            weight_quantization_target_point = self._backend_entity.target_point(TargetType.OPERATION_WITH_WEIGHTS,
                                                                                 node_name,
                                                                                 port_id)
            self._quantization_target_points_to_qconfig[weight_quantization_target_point] = quantization_point.qconfig

    def _add_activation_quantization_target_point(self,
                                                  quantization_point: SingleConfigQuantizationPoint) -> None:
        """
        Adds activation quantization target point to the set of existing points.

        :param nncf_graph: NNCFGraph instance for working with the graph and nodes.
        :param quantization_point: SingleConfigQuantizationPoint for the needed layer.
        """
        activation_quantization_target_point = self._get_activation_quantization_target_point(quantization_point)
        self._quantization_target_points_to_qconfig[activation_quantization_target_point] = quantization_point.qconfig

    def _get_activation_quantization_target_point(
            self,
            quantization_point: SingleConfigQuantizationPoint) -> SingleConfigQuantizationPoint:
        """
        Returns activation quantization target point to the set of existing points.

        :param nncf_graph: NNCFGraph instance for working with the graph and nodes.
        :param quantization_point: SingleConfigQuantizationPoint for the needed layer.
        :return: SingleConfigQuantizationPoint for the needed layer.
        """
        node_name = quantization_point.insertion_point.target_node_name
        # If Quantization of node's input
        if quantization_point.insertion_point.input_port_id is not None:
            input_port_id = quantization_point.insertion_point.input_port_id
            activation_quantization_target_point = self._backend_entity.target_point(TargetType.PRE_LAYER_OPERATION,
                                                                                     node_name,
                                                                                     input_port_id)
        # If quantization of node's output or Model Input node
        else:
            output_port_id = 0
            activation_quantization_target_point = self._backend_entity.target_point(TargetType.POST_LAYER_OPERATION,
                                                                                     node_name,
                                                                                     output_port_id)
        return activation_quantization_target_point

    def _get_quantization_target_points(self, model: TModel) -> OrderedDict[TargetPoint, QuantizerConfig]:
        """
        Returns Quantization Target Points.
        In the Compression Pipeline logic NNCF assumes that the compression pipeline works only on the single model.
        So for the optimization purpose if Quantization Target Points were computed before the function returns them,
        otherwise builds NNCFGraph from the 'model',
        finds the quantization setup and processes it to the Set of Quantization Target Points.

        :param model: Backend-specific model, for which Quantization Target Points are being seek.
        :param nncf_graph: NNCFGraph instance.
        :return: Set of Quantization Target Points.
        """
        nncf_graph = NNCFGraphFactory.create(model) if self.nncf_graph is None else self.nncf_graph

        if self._quantization_target_points_to_qconfig:
            return self._quantization_target_points_to_qconfig, self._unified_scale_groups
        backend = get_backend(model)
        device = self._parameters.target_device
        pattern = PatternsManager.get_full_pattern_graph(backend, device)
        quantizer_setup = self._get_quantizer_setup(nncf_graph, pattern)
        self._apply_model_type_pass(self._parameters.model_type, quantizer_setup, nncf_graph)
        self._unified_scale_groups = self._collect_unified_groups(quantizer_setup)
        quantization_points = list(quantizer_setup.quantization_points.values())
        quantization_points = self._topological_sort_quantization_points(quantization_points, nncf_graph)
        for quantization_point in quantization_points:
            if quantization_point.is_weight_quantization_point():
                self._add_weight_quantization_target_point(quantization_point, nncf_graph)
            elif quantization_point.is_activation_quantization_point():
                self._add_activation_quantization_target_point(quantization_point)
            else:
                raise RuntimeError('Incorrect quantization point')
        return self._quantization_target_points_to_qconfig, self._unified_scale_groups

    def _collect_unified_groups(self, quantizer_setup: SingleConfigQuantizerSetup) -> List[List[TargetPoint]]:
        """
        Collects the group of quantizers for unification.

        :param quantizer_setup: SingleConfigQuantizerSetup instance.
        :return: List with the groups of the TargetPoints.
        """
        unified_scale_groups = []
        for quantizer_ids in quantizer_setup.unified_scale_groups.values():
            unified_scale_group = []
            for quantizer_id in quantizer_ids:
                quantization_point = quantizer_setup.quantization_points[quantizer_id]

                # Only activation quantizers can be unified
                if quantization_point.is_activation_quantization_point():
                    activation_target_point = self._get_activation_quantization_target_point(quantization_point)
                    unified_scale_group.append(activation_target_point)
                else:
                    raise RuntimeError('Only activation quantizers can be unified.')
            unified_scale_groups.append(unified_scale_group)
        return unified_scale_groups

    def _get_graph_pattern(self, model: TModel) -> GraphPattern:
        """
        Returns full graph pattern for quantizer setup calculation.

        :param model: Backend-specific model.
        :return: GraphPattern instance.
        """
        backend = get_backend(model)
        device = self._parameters.target_device
        return PatternsManager.get_full_pattern_graph(backend, device)

    def _topological_sort_quantization_points(self, quantization_points: List[SingleConfigQuantizationPoint],
                                              nncf_graph: NNCFGraph) -> List[SingleConfigQuantizationPoint]:
        """
        Sorts quantization_points based on the topological order of nodes obtained form nncf_graph.

        :param quantization_points: Quantization points.
        :param nncf_graph: Instance of NNCFgraph used to get topological sort.
        :return: Sorted quantization_points.
        """
        node_names_to_pos = {node.node_name: i for i, node in enumerate(nncf_graph.topological_sort())}
        quantization_points.sort(key=lambda point: node_names_to_pos[point.insertion_point.target_node_name])
        return quantization_points

    def _get_first_quantized_convolutions(self, quantization_points: List[TargetPoint],
                                          starting_node: NNCFNode,
                                          nncf_graph: NNCFGraph) -> List[TargetPoint]:
        """
        Returns target points connected to a first visited node with Convolution metatype,
        which are included in quantization_points. A traversal of nncf_graph is started from starting_node.

        :param quantization_points: Quantization target points.
        :param starting_node: Node from which traversal is started.
        :param nncf_graph: Instance of NNCFGraph to traverse.
        :return: First visited target points.
        """
        target_node_names_to_qp = collections.defaultdict(list)
        for q_p in quantization_points:
            target_node_names_to_qp[q_p.target_node_name].append(q_p)
        queue = collections.deque(nncf_graph.get_next_nodes(starting_node))
        while queue:
            node = queue.popleft()
            node_name = node.node_name
            if node_name in target_node_names_to_qp:
                return target_node_names_to_qp[node_name]
            queue.extend(nncf_graph.get_next_nodes(node))
        return []

    def _get_quantization_points_overflow_fix(self, overflow_fix: OverflowFix,
                                              quantization_target_points: OrderedDict[TargetPoint, QuantizerConfig],
                                              nncf_graph: NNCFGraph) -> Set[TargetPoint]:
        """
        Returns quantization target points, for whom overflow_fix should be applied.

        :param overflow_fix: OverflowFix parameter.
        :param quantization_target_points: Quantization target points.
        :param nncf_graph: Instance of NNCFGraph to traverse.
        :return: quantization target points to apply
        """
        weight_quantization_points = set(filter(lambda point: point.is_weight_target_point(),
                                                quantization_target_points.keys()))
        output = set()
        if overflow_fix == OverflowFix.ENABLE:
            output = _filter_target_points_by_metatypes(weight_quantization_points,
                                                        self._backend_entity.overflow_fix_metatypes,
                                                        nncf_graph)
        if overflow_fix == OverflowFix.FIRST_LAYER:
            weight_quantization_points = _filter_target_points_by_metatypes(weight_quantization_points,
                                                                            self._backend_entity.conv_metatype,
                                                                            nncf_graph)
            for input_node in nncf_graph.get_input_nodes():
                nodes = self._get_first_quantized_convolutions(weight_quantization_points, input_node, nncf_graph)
                output.update(nodes)
        return output

    def _apply(self,
               model: TModel,
               statistic_points: Optional[StatisticPointsContainer] = None,
               dataset: Optional[Dataset] = None) -> TModel:
        transformation_layout = TransformationLayout()
        nncf_graph = NNCFGraphFactory.create(model) if self.nncf_graph is None else self.nncf_graph
        model_transformer = ModelTransformerFactory.create(model)
        quantization_target_points, unified_scale_groups = self._get_quantization_target_points(model)
        quantization_points_overflow_fix = self._get_quantization_points_overflow_fix(self._parameters.overflow_fix,
                                                                                      quantization_target_points,
                                                                                      nncf_graph)
        weight_layer_names = set()

        def filter_func(point: StatisticPoint) -> bool:
            return MinMaxQuantization in point.algorithm_to_tensor_collectors and \
                   point.target_point == quantization_target_point

        unified_ops_list = set()
        for unified_scale_group in unified_scale_groups:
            group_statistics = []
            for quantization_target_point in unified_scale_group:
                target_node_name = quantization_target_point.target_node_name
                for tensor_collector in statistic_points.get_algo_statistics_for_node(
                    target_node_name,
                    filter_func,
                    MinMaxQuantization):
                    group_statistics.append(tensor_collector.get_statistics())

            unified_values = self._backend_entity.unify_statistics(group_statistics)
            for quantization_target_point in unified_scale_group:
                qconfig = quantization_target_points[quantization_target_point]
                q_group = QuantizerGroup.ACTIVATIONS
                narrow_range = get_quantizer_narrow_range(qconfig, q_group)
                parameters = calculate_quantizer_parameters(unified_values, qconfig, q_group, narrow_range)
                command = self._backend_entity.create_activation_quantizer_insertion_command(
                    nncf_graph, quantization_target_point, qconfig, parameters)
                transformation_layout.register(command)
                unified_ops_list.add(quantization_target_point)

        for quantization_target_point, qconfig in quantization_target_points.items():
            if quantization_target_point in unified_ops_list:
                continue
            target_node_name = quantization_target_point.target_node_name
            for tensor_collector in statistic_points.get_algo_statistics_for_node(
                    target_node_name,
                    filter_func,
                    MinMaxQuantization):
                if quantization_target_point.is_weight_target_point():
                    # If the nodes share one weight tensor, we should have only one quantizer on that
                    weights_name = self._backend_entity.get_weight_name(nncf_graph, quantization_target_point)
                    if weights_name in weight_layer_names:
                        continue
                    weight_layer_names.add(weights_name)
                    quant_group = QuantizerGroup.WEIGHTS
                else:
                    quant_group = QuantizerGroup.ACTIVATIONS

                half_range = quantization_target_point in quantization_points_overflow_fix
                narrow_range = get_quantizer_narrow_range(qconfig, quant_group)
                statistics = tensor_collector.get_statistics()
                parameters = calculate_quantizer_parameters(statistics, qconfig, quant_group, narrow_range, half_range)
                if quantization_target_point.is_weight_target_point():
                    command = self._backend_entity.create_weight_quantizer_insertion_command(
                        nncf_graph, quantization_target_point, qconfig, parameters)
                else:
                    command = self._backend_entity.create_activation_quantizer_insertion_command(
                        nncf_graph, quantization_target_point, qconfig, parameters)

                transformation_layout.register(command)

        quantized_model = model_transformer.transform(transformation_layout)
        return quantized_model

    def get_statistic_points(self, model: TModel) -> StatisticPointsContainer:
        self._set_backend_entity(model)
        nncf_graph = NNCFGraphFactory.create(model) if self.nncf_graph is None else self.nncf_graph

        quantization_target_points, _ = self._get_quantization_target_points(model)
        output = StatisticPointsContainer()
        for quantization_target_point, qconfig in quantization_target_points.items():
            nncf_logger.debug(f'Adding target point {quantization_target_point.target_node_name}'
                              f' with type {quantization_target_point.type} for statistics collection')
            stat_collector = self._get_stat_collector(nncf_graph, quantization_target_point,
                                                      qconfig)
            output.add_statistic_point(StatisticPoint(target_point=quantization_target_point,
                                                      tensor_collector=stat_collector,
                                                      algorithm=MinMaxQuantization))
        return output

    def _apply_model_type_pass(self, model_type: Optional[ModelType], quantizer_setup: SingleConfigQuantizerSetup,
                               nncf_graph: NNCFGraph) -> None:
        """
        Applies changes in-place into quantizer setup based on model_type and device parameters.

        :param model_type: Model type parameter.
        :param quantizer_setup: Quantizer setup which considered to update.
        :param nncf_graph: Instance of NNCFGraph.
        :return: None
        """
        if model_type == ModelType.TRANSFORMER:
            for quantization_point in quantizer_setup.quantization_points.values():
                if quantization_point.is_activation_quantization_point():
                    for node_name in quantization_point.directly_quantized_operator_node_names:
                        node = nncf_graph.get_node_by_name(node_name)
                        mat_mul_metatype = self._backend_entity.mat_mul_metatype
                        if node.metatype != mat_mul_metatype:
                            continue
                        if quantization_point.qconfig.mode != QuantizationMode.SYMMETRIC and \
                                node.layer_attributes is None:
                            quantization_point.qconfig.mode = QuantizationMode.SYMMETRIC
                            nncf_logger.debug(f'Update quantization mode for the node {node_name}'
                                              f' to the symmetric due to ModelType parameter.')
