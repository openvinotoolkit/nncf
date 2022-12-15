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

from copy import deepcopy
from typing import Dict, Tuple, List, Set, TypeVar, Union, Optional

from nncf import Dataset
from nncf.common.graph.definitions import NNCFGraphNodeType
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.hardware.config import HWConfigType
from nncf.common.hardware.config import HW_CONFIG_TYPE_TARGET_DEVICE_MAP
from nncf.common.insertion_point_graph import InsertionPointGraph
from nncf.common.quantization.quantizer_propagation.solver import QuantizerPropagationSolver
from nncf.common.quantization.quantizer_setup import SingleConfigQuantizationPoint
from nncf.common.quantization.quantizer_setup import SingleConfigQuantizerSetup
from nncf.common.quantization.structs import QuantizableWeightedLayerNode
from nncf.common.quantization.structs import QuantizationMode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import QuantizationPreset
from nncf.common.quantization.structs import QuantizerGroup
from nncf.common.tensor_statistics.collectors import TensorStatisticCollectorBase
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.common.logging import nncf_logger

from nncf.quantization.algorithms.algorithm import Algorithm
from nncf.quantization.algorithms.algorithm import AlgorithmParameters
from nncf.quantization.algorithms.min_max.backend import ALGO_BACKENDS
from nncf.quantization.algorithms.definitions import RangeType
from nncf.quantization.algorithms.definitions import Granularity
from nncf.common.factory import NNCFGraphFactory
from nncf.common.tensor_statistics.statistic_point import StatisticPoint
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer

TModel = TypeVar('TModel')


class MinMaxQuantizationParameters(AlgorithmParameters):
    """
    Base class of MinMaxQuantization algorithm parameters.
    """

    DEFAULT_QCONFIG = QuantizerConfig(num_bits=8,
                                      mode=QuantizationMode.SYMMETRIC,
                                      signedness_to_force=None,
                                      per_channel=False)

    def __init__(self,
                 preset: QuantizationPreset = QuantizationPreset.PERFORMANCE,
                 weight_bits: int = 8,
                 weight_granularity: Granularity = Granularity.PERCHANNEL,
                 activation_bits: int = 8,
                 activation_granularity: Granularity = Granularity.PERTENSOR,
                 number_samples: int = 100,
                 target_device: HWConfigType = HWConfigType.CPU,
                 range_type: RangeType = RangeType.MEAN_MINMAX,
                 quantize_outputs: bool = False,
                 ignored_scopes: List[str] = None,
                 ):
        """
        :param weight_quantizer_config: QuantizerConfig instance for the weights.
        :param activation_quantizer_config: QuantizerConfig instance for the activations.
        :param number_samples: Number of samples for the statistics collection.
        :param target_device: Target device for the settings of the quantization pipeline.
        :param range_type: Range types for the quantizer's parameters.
        :param quantize_outputs: Boolean value that says whether quantize outputs or not.
        :param ignored_scopes: List of the layers that should not quantized during the process.
        """
        weight_mode, activation_mode = self._determine_weight_activation_modes(preset)
        self.weight_quantizer_config = self._determine_quantizer_config(weight_bits, weight_granularity, weight_mode)
        self.activation_quantizer_config = self._determine_quantizer_config(activation_bits, activation_granularity,
                                                                            activation_mode)
        self.number_samples = number_samples
        self.target_device = HWConfigType(HW_CONFIG_TYPE_TARGET_DEVICE_MAP[target_device.value])
        self.range_type = range_type
        self.quantize_outputs = quantize_outputs
        self.ignored_scopes = [] if ignored_scopes is None else ignored_scopes

    def to_json(self) -> Dict[str, Union[str, float, int]]:
        """
        Serialize all MinMaxQuantization parameters to JSON.
        """

    def _determine_weight_activation_modes(self,
                                           preset: QuantizationPreset) -> Tuple[QuantizationMode, QuantizationMode]:
        weight_mode = QuantizationPreset.get_params_configured_by_preset(preset, QuantizerGroup.WEIGHTS)['mode']
        activation_mode = QuantizationPreset.get_params_configured_by_preset(preset, QuantizerGroup.ACTIVATIONS)['mode']
        return weight_mode, activation_mode

    def _determine_quantizer_config(self, number_bits: int,
                                    granularity: Granularity, mode: QuantizationMode) -> QuantizerConfig:
        return QuantizerConfig(num_bits=number_bits, mode=mode,
                               per_channel=granularity == Granularity.PERCHANNEL)


class MinMaxQuantization(Algorithm):
    """
    Post-training MinMaxQuantization algorithm implementation.

    The main purpose of this algorithm to insert FakeQuantizes
    (or pairs of Quantizer-Dequantizer operations) into the model.
    This operation is very expressive and allows mapping values from arbitrary input and output ranges.
    This algorithm projects (discretize) the input values to the low-precision data type
    using affine transformation (with clamp and rounding) and then re-project discrete values back to the
    original range and data type.
    It can be considered as an emulation of the quantization/dequantization process which happens at runtime.

    :param weight_quantizer_config: QuantizerConfig instance for the weights.
    :param activation_quantizer_config: QuantizerConfig instance for the activations.
    :param number_samples: The number of the samples for the statistics collection.
    :param target_device: Target device for the settings of the quantization pipeline.
    :param range_type: Range types for the quantizer's parameters.
    :param quantize_outputs: Boolean value that says whether quantize outputs or not.
    :param ignored_scopes: List of the layers that should not quantized during the process.
    :param nncf_graph: NNCFGraph class for the algorithm.
    :param _quantization_target_points: Set of the unique target points.
    """

    def __init__(self, parameters: MinMaxQuantizationParameters):
        self.nncf_graph = None
        # It prevents the duplicate weight quantizers from being added.
        # It can happen when you have layers that share the identical weight tensor.
        self._quantization_target_points = set()  # type: Set[TargetPoint]
        self._parameters = parameters

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
        else:
            raise RuntimeError('Cannot return backend-specific entity'
                               'because {} is not supported!'.format(model_backend))

    def _get_stat_collector(self, quantizer_config: QuantizerConfig) -> TensorStatisticCollectorBase:
        """
        Creates and returns statistic collector instance based on the quantizer's configuration.

        :param quantizer_config: QuantizerConfig instance for the current layer.
        :return: One of the TensorStatisticCollectorBase instances
        """
        is_symmetric = quantizer_config.mode == QuantizationMode.SYMMETRIC
        axes = (0, 2, 3) if quantizer_config.per_channel else None
        if self._parameters.range_type == RangeType.MINMAX:
            return self._backend_entity.minmax_statistic_collector(use_abs_max=is_symmetric,
                                                                   reduction_shape=axes,
                                                                   num_samples=self._parameters.number_samples)
        if self._parameters.range_type == RangeType.MEAN_MINMAX:
            return self._backend_entity.mean_minmax_statistic_collector(use_per_sample_stats=False,
                                                                        use_abs_max=is_symmetric,
                                                                        reduction_shape=axes,
                                                                        num_samples=self._parameters.number_samples)
        raise RuntimeError('This range type is not supported!')

    def _get_quantizer_setup(self, nncf_graph: NNCFGraph) -> SingleConfigQuantizerSetup:
        """
        Returns SingleConfigQuantizerSetup instance based on the input NNCFGraph.

        :param nncf_graph: NNCFGraph instance.
        :return: SingleConfigQuantizerSetup for the current NNCFGraph entity.
        """
        ip_graph = InsertionPointGraph(nncf_graph)
        weight_nodes = nncf_graph.get_nodes_by_metatypes(self._backend_entity.layers_with_weights_metatypes)
        quantizable_layer_nodes = [QuantizableWeightedLayerNode(weight_node, [QuantizerConfig()])
                                   for weight_node in weight_nodes]
        pattern = self._backend_entity.hw_fused_patterns.get_full_pattern_graph()
        ip_graph = ip_graph.get_ip_graph_with_merged_hw_optimized_operations(pattern, quantizable_layer_nodes)

        hw_config_type = self._parameters.target_device
        hw_config_path = self._backend_entity.hw_config.get_path_to_hw_config(hw_config_type)
        hw_config = self._backend_entity.hw_config.from_json(hw_config_path)
        default_config = deepcopy(self._parameters.DEFAULT_QCONFIG)

        post_processing_types = self._backend_entity.post_processing_metatypes
        solver = QuantizerPropagationSolver(ignored_scopes=self._parameters.ignored_scopes,
                                            hw_config=hw_config,
                                            default_trait_to_metatype_map=self._backend_entity.quant_trait_op_dict,
                                            default_qconfig_list=[default_config],
                                            quantizable_layer_nodes=quantizable_layer_nodes,
                                            quantize_outputs=self._parameters.quantize_outputs,
                                            post_processing_marker_metatypes=post_processing_types)

        quantization_proposal = solver.run_on_ip_graph(ip_graph)
        multi_config_setup = quantization_proposal.quantizer_setup
        single_config_setup = multi_config_setup.select_first_qconfig_for_each_point()
        finalized_proposal = quantization_proposal.finalize(single_config_setup)
        final_setup = solver.get_final_quantizer_setup(finalized_proposal)
        return final_setup

    def _add_weight_quantization_target_point(self, quantization_point: SingleConfigQuantizationPoint,
                                              model: TModel,
                                              nncf_graph: NNCFGraph) -> None:
        """
        Adds weight quantization target point to the set of existing points.

        :param quantization_point: SingleConfigQuantizationPoint for the needed layer.
        :param model: Model in the original framework.
        :param nncf_graph: The built NNCFGraph of the model.
        """
        node_name = quantization_point.insertion_point.target_node_name
        node = nncf_graph.get_node_by_name(node_name)
        port_id = self._backend_entity.get_weight_tensor_port_id(node)
        weight_quantization_target_point = self._backend_entity.target_point(TargetType.OPERATION_WITH_WEIGHTS,
                                                                             node_name,
                                                                             port_id)
        self._quantization_target_points.add(weight_quantization_target_point)

    def _add_activation_quantization_target_point(self,
                                                  quantization_point: SingleConfigQuantizationPoint) -> None:
        """
        Adds activation quantization target point to the set of existing points.

        :param nncf_graph: NNCFGraph instance for working with the graph and nodes.
        :param quantization_point: SingleConfigQuantizationPoint for the needed layer.
        """
        node_name = quantization_point.insertion_point.target_node_name
        # If quantization of Model Input node
        if NNCFGraphNodeType.INPUT_NODE in node_name:
            # There is only onde node - input_node
            output_port_id = 0
            activation_quantization_target_point = self._backend_entity.target_point(TargetType.POST_LAYER_OPERATION,
                                                                                     node_name,
                                                                                     output_port_id)
        # If not Model Input node
        # If Quantization of node's input
        elif quantization_point.insertion_point.input_port_id is not None:
            input_port_id = quantization_point.insertion_point.input_port_id
            activation_quantization_target_point = self._backend_entity.target_point(TargetType.PRE_LAYER_OPERATION,
                                                                                     node_name,
                                                                                     input_port_id)
        # If quantization of node's output
        else:
            output_port_id = 0
            activation_quantization_target_point = self._backend_entity.target_point(TargetType.POST_LAYER_OPERATION,
                                                                                     node_name,
                                                                                     output_port_id)
        self._quantization_target_points.add(activation_quantization_target_point)

    def _get_quantization_target_points(self, model: TModel) -> Set[TargetPoint]:
        """
        Returns Quantization Target Points.
        In the Compression Pipeline logic NNCF assumes that the compression pipeline works only on the single model.
        So for the optimization purpose if Quantization Target Points were computed before the function returns them,
        otherwise builds NNCFGraph from the 'model',
        finds the quantization setup and processes it to the Set of Quantization Target Points.

        :param model: Backend-specific model, for which Quantization Target Points are being seek.
        :return: Set of Quantization Target Points.
        """
        nncf_graph = NNCFGraphFactory.create(model) if self.nncf_graph is None else self.nncf_graph

        if self._quantization_target_points:
            return self._quantization_target_points
        quantizer_setup = self._get_quantizer_setup(nncf_graph)
        for quantization_point in quantizer_setup.quantization_points.values():
            if quantization_point.is_weight_quantization_point():
                self._add_weight_quantization_target_point(quantization_point, model, nncf_graph)
            elif quantization_point.is_activation_quantization_point():
                self._add_activation_quantization_target_point(quantization_point)
            else:
                raise RuntimeError('Incorrect quantization point')
        self._quantization_target_points = sorted(self._quantization_target_points)
        return self._quantization_target_points

    def _apply(self,
               model: TModel,
               statistic_points: Optional[StatisticPointsContainer] = None,
               dataset: Optional[Dataset] = None) -> TModel:
        transformation_layout, transformation_commands = TransformationLayout(), []
        nncf_graph = NNCFGraphFactory.create(model) if self.nncf_graph is None else self.nncf_graph
        model_transformer = self._backend_entity.model_transformer(model)

        quantization_target_points = self._get_quantization_target_points(model)
        weight_quantizer_config = self._backend_entity.get_weight_config(self._parameters.weight_quantizer_config,
                                                                         model)
        weight_tensor_names = set()

        for quantization_target_point in quantization_target_points:
            target_node_name = quantization_target_point.target_node_name
            node = nncf_graph.get_node_by_name(target_node_name)
            if quantization_target_point.type == TargetType.OPERATION_WITH_WEIGHTS:
                weight_tensor_name, weight_tensor = self._backend_entity.get_weight_tensor(
                    model, quantization_target_point)
                # If the nodes share one weight tensor, we should have only one quantizer on that
                if weight_tensor_name in weight_tensor_names:
                    continue
                weight_tensor_names.add(weight_tensor_name)
                command = self._backend_entity.create_weight_quantizer_insertion_command(
                    quantization_target_point, weight_quantizer_config, weight_tensor, node)
                transformation_commands.append(command)
            elif quantization_target_point.type in [TargetType.PRE_LAYER_OPERATION, TargetType.POST_LAYER_OPERATION]:
                def filter_func(point):
                    return MinMaxQuantization in point.algorithm_to_tensor_collectors and \
                           point.target_point.type == quantization_target_point.type

                for tensor_collector in statistic_points.get_algo_statistics_for_node(
                        target_node_name,
                        filter_func,
                        MinMaxQuantization):
                    command = self._backend_entity.create_activation_quantizer_insertion_command(
                                    quantization_target_point,
                                    self._parameters.activation_quantizer_config,
                                    tensor_collector.get_statistics())
                    transformation_commands.append(command)
            else:
                raise RuntimeError('Inccorrect type of Quantization Target Point!')

        for transformation_command in transformation_commands:
            transformation_layout.register(transformation_command)

        quantized_model = model_transformer.transform(transformation_layout)
        return quantized_model

    def get_statistic_points(self, model: TModel) -> StatisticPointsContainer:
        self._set_backend_entity(model)
        quantization_target_points = self._get_quantization_target_points(model)
        output = StatisticPointsContainer()
        for quantization_target_point in quantization_target_points:
            if quantization_target_point.type in [TargetType.PRE_LAYER_OPERATION, TargetType.POST_LAYER_OPERATION]:
                nncf_logger.debug(f'Adding target point {quantization_target_point.target_node_name} '
                                  f'for statistics collection')
                stat_collector = self._get_stat_collector(self._parameters.activation_quantizer_config)
                output.add_statistic_point(StatisticPoint(target_point=quantization_target_point,
                                                          tensor_collector=stat_collector,
                                                          algorithm=MinMaxQuantization))
            else:
                nncf_logger.debug(f'Skipping collection at {quantization_target_point} - this is a weight quantizer')
        return output
