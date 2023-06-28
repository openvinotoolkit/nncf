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

import collections
import dataclasses
from copy import deepcopy
from typing import Any, Dict, List, Optional, OrderedDict, Set, TypeVar

from nncf import Dataset
from nncf.common.factory import ModelTransformerFactory
from nncf.common.factory import NNCFGraphFactory
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.patterns import GraphPattern
from nncf.common.graph.patterns.manager import PatternsManager
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.hardware.config import get_hw_config_type
from nncf.common.insertion_point_graph import InsertionPointGraph
from nncf.common.logging import nncf_logger
from nncf.common.quantization.config_assignment import assign_qconfig_lists_to_modules
from nncf.common.quantization.quantizer_propagation.solver import QuantizerPropagationSolver
from nncf.common.quantization.quantizer_setup import SingleConfigQuantizationPoint
from nncf.common.quantization.quantizer_setup import SingleConfigQuantizerSetup
from nncf.common.quantization.structs import QuantizableWeightedLayerNode
from nncf.common.quantization.structs import QuantizationConstraints
from nncf.common.quantization.structs import QuantizationMode
from nncf.common.quantization.structs import QuantizationPreset
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import QuantizerGroup
from nncf.common.tensor_statistics.collectors import TensorStatisticCollectorBase
from nncf.common.tensor_statistics.statistic_point import StatisticPoint
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.parameters import ModelType
from nncf.parameters import TargetDevice
from nncf.quantization.advanced_parameters import OverflowFix
from nncf.quantization.advanced_parameters import QuantizationParameters
from nncf.quantization.advanced_parameters import changes_asdict
from nncf.quantization.algorithms.algorithm import Algorithm
from nncf.quantization.algorithms.min_max.backend import ALGO_BACKENDS
from nncf.quantization.fake_quantize import calculate_quantizer_parameters
from nncf.quantization.fake_quantize import get_quantizer_narrow_range
from nncf.quantization.passes import transform_to_inference_graph
from nncf.quantization.range_estimator import RangeEstimatorParameters
from nncf.quantization.range_estimator import RangeEstimatorParametersSet
from nncf.scopes import IgnoredScope
from nncf.scopes import get_ignored_node_names_from_ignored_scope

TModel = TypeVar("TModel")

DEFAULT_QCONFIG = QuantizerConfig(
    num_bits=8, mode=QuantizationMode.SYMMETRIC, signedness_to_force=None, per_channel=False
)


def _filter_target_points_by_metatypes(
    quantization_target_points: Set[TargetPoint], metatypes: List[OperatorMetatype], nncf_graph: NNCFGraph
) -> Set[TargetPoint]:
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


class MinMaxQuantization(Algorithm):
    """
    Post-training MinMaxQuantization algorithm.

    The algorithm modifies the model by inserting additional nodes, which emulates the quantization of the data flow.
    The algorithm calibrates the parameters of the inserted nodes by collecting the statistics in the insertion points.
    The modified model is returned after the work of the algorithm, which can be performed via the original framework.
    It is expected that the inference of the obtained model in the int8 mode would be faster than the original model.
    """

    def __init__(
        self,
        preset: QuantizationPreset = QuantizationPreset.PERFORMANCE,
        target_device: TargetDevice = TargetDevice.ANY,
        subset_size: int = 300,
        model_type: Optional[ModelType] = None,
        ignored_scope: Optional[IgnoredScope] = None,
        overflow_fix: OverflowFix = OverflowFix.FIRST_LAYER,
        quantize_outputs: bool = False,
        inplace_statistics: bool = True,
        activations_quantization_params: Optional[QuantizationParameters] = None,
        weights_quantization_params: Optional[QuantizationParameters] = None,
        activations_range_estimator_params: Optional[RangeEstimatorParameters] = None,
        weights_range_estimator_params: Optional[RangeEstimatorParameters] = None,
        backend_params: Optional[Dict[str, Any]] = None,
    ):
        """
        :param preset: A preset that controls the quantization mode,
            defaults to QuantizationPreset.PERFORMANCE.
        :param target_device: A target device the specificity of which will be taken
            into account while compressing in order to obtain the best performance
            for this type of device, defaults to TargetDevice.ANY.
        :param subset_size: Size of a subset to calculate activations statistics used
            for quantization, defaults to 300.
        :param model_type: Model type is needed to specify additional patterns
            in the model. Supported only `transformer` now.
        :param ignored_scope: An ignored scope that defined the list of model control
            flow graph nodes to be ignored during quantization.
        :param overflow_fix: This option controls whether to apply the overflow issue
            fix for the 8-bit quantization, defaults to OverflowFix.FIRST_LAYER.
        :param quantize_outputs: Whether to insert additional quantizers right before
            each of the model outputs.
        :param inplace_statistics: Defines wheather to calculate quantizers statistics
            by backend graph operations or by default Python implementation, defaults
            to True.
        :param activations_quantization_params: Quantization parameters for model
            activations.
        :param weights_quantization_params: Quantization parameters for model weights.
        :param activations_range_estimator_params: Quantization range estimation
            parameters for activation.
        :param weights_range_estimator_params: Quantization range estimation parameters
            for weights.
        :param backend_params: Backend specific parameters.
        """
        self._target_device = target_device
        self._subset_size = subset_size
        self._model_type = model_type
        self._ignored_scope = IgnoredScope() if ignored_scope is None else ignored_scope
        self._overflow_fix = overflow_fix
        self._quantize_outputs = quantize_outputs
        self._inplace_statistics = inplace_statistics
        self._backend_params = backend_params

        self._quantization_params = {
            QuantizerGroup.WEIGHTS: weights_quantization_params,
            QuantizerGroup.ACTIVATIONS: activations_quantization_params,
        }

        self._range_estimator_params = {
            QuantizerGroup.WEIGHTS: weights_range_estimator_params,
            QuantizerGroup.ACTIVATIONS: activations_range_estimator_params,
        }

        # Calculates global quantizer constraints
        self._global_quantizer_constraints = {}
        for quantizer_group in QuantizerGroup:
            self._global_quantizer_constraints[quantizer_group] = self._get_quantizer_constraints(
                quantizer_group, preset, self._quantization_params[quantizer_group]
            )

        # It prevents the duplicate weight quantizers from being added.
        # It can happen when you have layers that share the identical weight tensor.
        self._quantization_target_points_to_qconfig = (
            collections.OrderedDict()
        )  # type: OrderedDict[TargetPoint, QuantizerConfig]
        self._unified_scale_groups = []

    @property
    def available_backends(self) -> Dict[str, BackendType]:
        return ALGO_BACKENDS.registry_dict

    def _get_quantizer_constraints(
        self, group: QuantizerGroup, preset: QuantizationPreset, quantization_params: Optional[QuantizationParameters]
    ) -> QuantizationConstraints:
        """
        Returns QuantizationConstraints for the provided quantizer group.

        :param group: Quantizer group.
        :param preset: Quantization preset.
        :param quantization_params: Quantization parameters.
        :return: QuantizationConstraints.
        """
        constraints = {"mode": preset.get_params_configured_by_preset(group)["mode"]}
        if quantization_params is None:
            return QuantizationConstraints(**constraints)

        if quantization_params.mode is not None:
            constraints["mode"] = quantization_params.mode
        if quantization_params.num_bits is not None:
            constraints["num_bits"] = quantization_params.num_bits
        if quantization_params.per_channel is not None:
            constraints["per_channel"] = quantization_params.per_channel
        if quantization_params.signedness_to_force is not None:
            constraints["signedness_to_force"] = quantization_params.signedness_to_force

        return QuantizationConstraints(**constraints)

    def _set_backend_entity(self, model: TModel) -> None:
        """
        Creates a helper class with a backed-specific logic of the algorithm

        :param model: backend-specific input model
        """
        model_backend = get_backend(model)
        if model_backend == BackendType.ONNX:
            from nncf.quantization.algorithms.min_max.onnx_backend import ONNXMinMaxAlgoBackend

            self._backend_entity = ONNXMinMaxAlgoBackend()
        elif model_backend == BackendType.OPENVINO:
            from nncf.quantization.algorithms.min_max.openvino_backend import OVMinMaxAlgoBackend

            self._backend_entity = OVMinMaxAlgoBackend()
        elif model_backend == BackendType.TORCH:
            from nncf.quantization.algorithms.min_max.torch_backend import PTMinMaxAlgoBackend

            self._backend_entity = PTMinMaxAlgoBackend()
        else:
            raise RuntimeError(
                "Cannot return backend-specific entity because {} is not supported!".format(model_backend)
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
        self, nncf_graph: NNCFGraph, target_point: TargetPoint, quantizer_config: QuantizerConfig
    ) -> TensorStatisticCollectorBase:
        """
        Creates and returns statistic collector instance based on the quantizer's configuration.

        :param quantizer_config: QuantizerConfig instance for the current layer.
        :return: One of the TensorStatisticCollectorBase instances
        """
        range_estimator_params = self._get_range_estimator_parameters(target_point, quantizer_config)

        return self._backend_entity.get_statistic_collector(
            range_estimator_params,
            nncf_graph,
            target_point,
            quantizer_config,
            inplace=self._inplace_statistics,
            num_samples=self._subset_size,
        )

    def _get_default_qconfig(self, constraints: QuantizationConstraints = None) -> QuantizerConfig:
        """
        Returns default quantizer configuration, based on the provided constraints.

        :param constraints: Quantization constraints.
        :return: Quantizer config.
        """
        qconfig = deepcopy(DEFAULT_QCONFIG)
        if constraints is not None:
            qconfig = constraints.apply_constraints_to(qconfig)
        return qconfig

    def _get_ignored_names(self, nncf_graph: NNCFGraph, ignored_patterns: GraphPattern) -> Set[str]:
        """
        Returns all node names that are ignored for quantization:
        Firstly, the ignored names are obtained from user-defined ignored the scope.
        Secondly, the ignored names are updated from model_type parameter.
        Lastly, the ignored names are updated from ignored_patterns.

        :param nncf_graph: NNCFGraph instance.
        :param ignored_patterns: Ignored patterns.
        :return: Node names are ignored for quantization.
        """
        model_type = self._model_type
        device = self._target_device
        ignored_names = set()

        ignored_names.update(
            get_ignored_node_names_from_ignored_scope(
                self._ignored_scope, nncf_graph, strict=self._ignored_scope.validate
            )
        )

        model_type_ignore_scope = self._backend_entity.get_ignored_scope(model_type, device)

        ignored_names.update(
            get_ignored_node_names_from_ignored_scope(model_type_ignore_scope, nncf_graph, strict=False)
        )

        ignored_scope = self._get_ignored_scope(nncf_graph, ignored_patterns)

        ignored_names.update(
            get_ignored_node_names_from_ignored_scope(ignored_scope, nncf_graph, strict=self._ignored_scope.validate)
        )

        return ignored_names

    def _get_ignored_scope(self, nncf_graph: NNCFGraph, ignored_patterns: GraphPattern) -> IgnoredScope:
        """
        Returns IgnoredScope with node names matched ignored_patterns.

        :param nncf_graph: NNCFGraph instance.
        :param ignored_patterns: Ignored patterns.
        :return: IgnoredScope with all node names matched ignored_patterns.
        """
        nncf_node_names = [nncf_node.node_name for nncf_node in nncf_graph.find_matching_nodes(ignored_patterns)]
        return IgnoredScope(names=nncf_node_names)

    def _get_quantizer_setup(
        self, nncf_graph: NNCFGraph, hw_patterns: GraphPattern, ignored_patterns: GraphPattern
    ) -> SingleConfigQuantizerSetup:
        """
        Returns SingleConfigQuantizerSetup instance based on the input NNCFGraph.

        :param nncf_graph: NNCFGraph instance.
        :param hw_patterns: Hardware patterns.
        :param ignored_patterns: Ignored patterns.
        :return: SingleConfigQuantizerSetup for the current NNCFGraph entity.
        """
        hw_config_type = get_hw_config_type(self._target_device.value)
        hw_config_path = self._backend_entity.hw_config.get_path_to_hw_config(hw_config_type)
        hw_config = self._backend_entity.hw_config.from_json(hw_config_path)

        ignored_names = self._get_ignored_names(nncf_graph, ignored_patterns)
        weight_nodes = self._backend_entity.get_weight_nodes(nncf_graph)

        default_weight_qconfig = self._get_default_qconfig(self._global_quantizer_constraints[QuantizerGroup.WEIGHTS])
        weighted_node_and_qconf_lists = assign_qconfig_lists_to_modules(
            nodes_with_weights=weight_nodes,
            default_weight_qconfig=default_weight_qconfig,
            global_weight_constraints=self._global_quantizer_constraints[QuantizerGroup.WEIGHTS],
            scope_overrides_dict=None,
            hw_config=hw_config,
        )
        quantizable_layer_nodes = [
            QuantizableWeightedLayerNode(node, qconf_list) for node, qconf_list in weighted_node_and_qconf_lists.items()
        ]

        inference_nncf_graph = transform_to_inference_graph(
            deepcopy(nncf_graph), self._backend_entity.shapeof_metatypes, self._backend_entity.read_variable_metatypes
        )
        ip_graph = InsertionPointGraph(inference_nncf_graph)
        ip_graph = ip_graph.get_ip_graph_with_merged_hw_optimized_operations(hw_patterns)
        post_processing_types = self._backend_entity.post_processing_metatypes
        solver = QuantizerPropagationSolver(
            activation_ignored_scopes=ignored_names,
            weight_ignored_scopes=ignored_names,
            hw_config=hw_config,
            default_trait_to_metatype_map=self._backend_entity.quant_trait_op_dict,
            default_qconfig_list=[
                self._get_default_qconfig(self._global_quantizer_constraints[QuantizerGroup.ACTIVATIONS])
            ],
            quantizable_layer_nodes=quantizable_layer_nodes,
            quantize_outputs=self._quantize_outputs,
            global_constraints=self._global_quantizer_constraints,
            post_processing_marker_metatypes=post_processing_types,
            scales_unification_map=self._backend_entity.scales_unification_map,
        )

        quantization_proposal = solver.run_on_ip_graph(ip_graph)
        multi_config_setup = quantization_proposal.quantizer_setup
        single_config_setup = multi_config_setup.select_first_qconfig_for_each_point()
        finalized_proposal = quantization_proposal.finalize(single_config_setup)
        final_setup = solver.get_final_quantizer_setup(finalized_proposal)
        return final_setup

    def _add_weight_quantization_target_point(
        self, quantization_point: SingleConfigQuantizationPoint, nncf_graph: NNCFGraph
    ) -> None:
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
            weight_quantization_target_point = self._backend_entity.target_point(
                TargetType.OPERATION_WITH_WEIGHTS, node_name, port_id
            )
            self._quantization_target_points_to_qconfig[weight_quantization_target_point] = quantization_point.qconfig

    def _add_activation_quantization_target_point(self, quantization_point: SingleConfigQuantizationPoint) -> None:
        """
        Adds activation quantization target point to the set of existing points.

        :param nncf_graph: NNCFGraph instance for working with the graph and nodes.
        :param quantization_point: SingleConfigQuantizationPoint for the needed layer.
        """
        activation_quantization_target_point = self._get_activation_quantization_target_point(quantization_point)
        self._quantization_target_points_to_qconfig[activation_quantization_target_point] = quantization_point.qconfig

    def _get_activation_quantization_target_point(
        self, quantization_point: SingleConfigQuantizationPoint
    ) -> SingleConfigQuantizationPoint:
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
            activation_quantization_target_point = self._backend_entity.target_point(
                TargetType.PRE_LAYER_OPERATION, node_name, input_port_id
            )
        # If quantization of node's output or Model Input node
        else:
            output_port_id = 0
            activation_quantization_target_point = self._backend_entity.target_point(
                TargetType.POST_LAYER_OPERATION, node_name, output_port_id
            )
        return activation_quantization_target_point

    def _get_quantization_target_points(
        self, model: TModel, nncf_graph: NNCFGraph
    ) -> OrderedDict[TargetPoint, QuantizerConfig]:
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
        if self._quantization_target_points_to_qconfig:
            return self._quantization_target_points_to_qconfig, self._unified_scale_groups
        backend = get_backend(model)
        device = self._target_device
        model_type = self._model_type
        ignored_patterns = PatternsManager.get_full_ignored_pattern_graph(
            backend=backend, device=device, model_type=model_type
        )
        hw_patterns = PatternsManager.get_full_hw_pattern_graph(backend=backend, device=device, model_type=model_type)
        quantizer_setup = self._get_quantizer_setup(nncf_graph, hw_patterns, ignored_patterns)
        self._apply_model_type_pass(self._model_type, quantizer_setup, nncf_graph)
        self._unified_scale_groups = self._collect_unified_groups(quantizer_setup)
        quantization_points = list(quantizer_setup.quantization_points.values())
        quantization_points = self._topological_sort_quantization_points(quantization_points, nncf_graph)
        for quantization_point in quantization_points:
            if quantization_point.is_weight_quantization_point():
                self._add_weight_quantization_target_point(quantization_point, nncf_graph)
            elif quantization_point.is_activation_quantization_point():
                self._add_activation_quantization_target_point(quantization_point)
            else:
                raise RuntimeError("Incorrect quantization point")
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
                    raise RuntimeError("Only activation quantizers can be unified.")
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

    def _get_first_quantized_convolutions(
        self, quantization_points: List[TargetPoint], starting_node: NNCFNode, nncf_graph: NNCFGraph
    ) -> List[TargetPoint]:
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

        first_convs = []
        visited = set()
        while queue:
            node = queue.popleft()
            node_name = node.node_name
            if node_name in visited:
                continue
            visited.add(node_name)
            if node_name in target_node_names_to_qp:
                first_convs.extend(target_node_names_to_qp[node_name])
            else:
                queue.extend(nncf_graph.get_next_nodes(node))
        return first_convs

    def _get_quantization_points_overflow_fix(
        self,
        overflow_fix: OverflowFix,
        quantization_target_points: OrderedDict[TargetPoint, QuantizerConfig],
        nncf_graph: NNCFGraph,
    ) -> Set[TargetPoint]:
        """
        Returns quantization target points, for whom overflow_fix should be applied.

        :param overflow_fix: OverflowFix parameter.
        :param quantization_target_points: Quantization target points.
        :param nncf_graph: Instance of NNCFGraph to traverse.
        :return: quantization target points to apply
        """
        weight_quantization_points = set(
            filter(lambda point: point.is_weight_target_point(), quantization_target_points.keys())
        )
        output = set()
        if overflow_fix == OverflowFix.ENABLE:
            output = _filter_target_points_by_metatypes(
                weight_quantization_points, self._backend_entity.overflow_fix_metatypes, nncf_graph
            )
        if overflow_fix == OverflowFix.FIRST_LAYER:
            weight_quantization_points = _filter_target_points_by_metatypes(
                weight_quantization_points, self._backend_entity.conv_metatype, nncf_graph
            )
            for input_node in nncf_graph.get_input_nodes():
                nodes = self._get_first_quantized_convolutions(weight_quantization_points, input_node, nncf_graph)
                output.update(nodes)
        return output

    def _apply(
        self,
        model: TModel,
        statistic_points: Optional[StatisticPointsContainer] = None,
        dataset: Optional[Dataset] = None,
    ) -> TModel:
        transformation_layout = TransformationLayout()
        nncf_graph = NNCFGraphFactory.create(model)
        model_transformer = ModelTransformerFactory.create(model)
        quantization_target_points, unified_scale_groups = self._get_quantization_target_points(model, nncf_graph)
        quantization_points_overflow_fix = self._get_quantization_points_overflow_fix(
            self._overflow_fix, quantization_target_points, nncf_graph
        )
        weight_layer_names = set()

        def filter_func(point: StatisticPoint) -> bool:
            return (
                MinMaxQuantization in point.algorithm_to_tensor_collectors
                and point.target_point == quantization_target_point
            )

        unified_ops_list = set()
        for unified_scale_group in unified_scale_groups:
            group_statistics = []
            for quantization_target_point in unified_scale_group:
                target_node_name = quantization_target_point.target_node_name
                for tensor_collector in statistic_points.get_algo_statistics_for_node(
                    target_node_name, filter_func, MinMaxQuantization
                ):
                    group_statistics.append(tensor_collector.get_statistics())

            unified_values = self._backend_entity.unify_statistics(group_statistics)
            for quantization_target_point in unified_scale_group:
                qconfig = quantization_target_points[quantization_target_point]
                q_group = QuantizerGroup.ACTIVATIONS
                narrow_range = get_quantizer_narrow_range(qconfig, q_group)
                parameters = calculate_quantizer_parameters(unified_values, qconfig, q_group, narrow_range)
                command = self._backend_entity.create_activation_quantizer_insertion_command(
                    nncf_graph, quantization_target_point, qconfig, parameters
                )
                transformation_layout.register(command)
                unified_ops_list.add(quantization_target_point)

        for quantization_target_point, qconfig in quantization_target_points.items():
            if quantization_target_point in unified_ops_list:
                continue
            target_node_name = quantization_target_point.target_node_name
            for tensor_collector in statistic_points.get_algo_statistics_for_node(
                target_node_name, filter_func, MinMaxQuantization
            ):
                if quantization_target_point.is_weight_target_point():
                    weights_name = self._backend_entity.get_weight_name(nncf_graph, quantization_target_point)
                    if not self._backend_entity.should_quantize_weight(weights_name, weight_layer_names):
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
                        nncf_graph, quantization_target_point, qconfig, parameters
                    )
                else:
                    command = self._backend_entity.create_activation_quantizer_insertion_command(
                        nncf_graph, quantization_target_point, qconfig, parameters
                    )

                transformation_layout.register(command)

        quantized_model = model_transformer.transform(transformation_layout)
        return quantized_model

    def get_statistic_points(self, model: TModel) -> StatisticPointsContainer:
        self._set_backend_entity(model)
        nncf_graph = NNCFGraphFactory.create(model)

        quantization_target_points, _ = self._get_quantization_target_points(model, nncf_graph)
        output = StatisticPointsContainer()
        for quantization_target_point, qconfig in quantization_target_points.items():
            nncf_logger.debug(
                f"Adding target point {quantization_target_point.target_node_name}"
                f" with type {quantization_target_point.type} for statistics collection"
            )
            stat_collector = self._get_stat_collector(nncf_graph, quantization_target_point, qconfig)
            output.add_statistic_point(
                StatisticPoint(
                    target_point=quantization_target_point,
                    tensor_collector=stat_collector,
                    algorithm=MinMaxQuantization,
                )
            )
        return output

    def _apply_model_type_pass(
        self, model_type: Optional[ModelType], quantizer_setup: SingleConfigQuantizerSetup, nncf_graph: NNCFGraph
    ) -> None:
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
                        if (
                            quantization_point.qconfig.mode != QuantizationMode.SYMMETRIC
                            and node.layer_attributes is None
                        ):
                            quantization_point.qconfig.mode = QuantizationMode.SYMMETRIC
                            nncf_logger.debug(
                                f"Update quantization mode for the node {node_name}"
                                f" to the symmetric due to ModelType parameter."
                            )
