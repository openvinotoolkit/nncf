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

from math import inf
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

import nncf
from nncf import Dataset
from nncf.common.factory import EngineFactory
from nncf.common.factory import ModelTransformerFactory
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.logging import nncf_logger
from nncf.common.logging.track_progress import track
from nncf.common.tensor_statistics.statistic_point import StatisticPoint
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.experimental.common.tensor_statistics.statistical_functions import mean_per_channel
from nncf.quantization.algorithms.algorithm import Algorithm
from nncf.tensor import Tensor
from nncf.tensor import functions as fns

TModel = TypeVar("TModel")
TTensor = TypeVar("TTensor")

FAST_BIAS_CORRECTION_THRESHOLD = 2


class FastBiasCorrection(Algorithm):
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
        threshold: float = FAST_BIAS_CORRECTION_THRESHOLD,
        apply_for_all_nodes: bool = False,
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
        self.threshold = threshold
        self.apply_for_all_nodes = apply_for_all_nodes
        self.inplace_statistics = inplace_statistics
        self.backend_params = backend_params
        self._backend_entity = None
        self._algorithm_key = f"FBC_{hash(self)}"

        if self.apply_for_all_nodes:
            raise nncf.InternalError("FastBiasCorrection algorithm does not support apply_for_all_nodes=True yet")

    @property
    def available_backends(self) -> List[BackendType]:
        return [BackendType.ONNX, BackendType.OPENVINO, BackendType.TORCH, BackendType.TORCH_FX]

    def _set_backend_entity(self, model: TModel) -> None:
        """
        Creates a helper class with a backed-specific logic of the algorithm.

        :param model: Backend-specific input model.
        """
        model_backend = get_backend(model)
        if model_backend == BackendType.ONNX:
            from nncf.quantization.algorithms.fast_bias_correction.onnx_backend import ONNXFastBiasCorrectionAlgoBackend

            self._backend_entity = ONNXFastBiasCorrectionAlgoBackend()
        elif model_backend == BackendType.OPENVINO:
            from nncf.quantization.algorithms.fast_bias_correction.openvino_backend import (
                OVFastBiasCorrectionAlgoBackend,
            )

            self._backend_entity = OVFastBiasCorrectionAlgoBackend(model)
        elif model_backend == BackendType.TORCH:
            from nncf.quantization.algorithms.fast_bias_correction.torch_backend import PTFastBiasCorrectionAlgoBackend

            self._backend_entity = PTFastBiasCorrectionAlgoBackend()
        elif model_backend == BackendType.TORCH_FX:
            from nncf.quantization.algorithms.fast_bias_correction.torch_fx_backend import (
                FXFastBiasCorrectionAlgoBackend,
            )

            self._backend_entity = FXFastBiasCorrectionAlgoBackend()
        else:
            raise nncf.UnsupportedBackendError(
                "Cannot return backend-specific entity because {} is not supported!".format(model_backend.value)
            )

    def apply(
        self,
        model: TModel,
        graph: NNCFGraph,
        statistic_points: Optional[StatisticPointsContainer] = None,
        dataset: Optional[Dataset] = None,
    ) -> TModel:
        self._set_backend_entity(model)

        model_transformer = ModelTransformerFactory.create(model)

        node_and_bias_value = [
            (node, self._backend_entity.get_bias_value(node, graph, model))
            for node in graph.get_all_nodes()
            if self._backend_entity.is_node_with_bias(node, graph)
        ]

        # Fill `node_and_new_bias_value` list. It is a correspondence between nodes
        # for which we should update bias and new bias values.
        node_and_new_bias_value = []

        for node, bias_value in track(node_and_bias_value, description="Applying Fast Bias correction"):
            node_name = node.node_name

            if not self._backend_entity.is_quantized_weights(node, graph):
                nncf_logger.debug(f"Skipping node {node_name} because weights were not quantized")
                continue

            in_node_name, out_node_name = self._backend_entity.get_node_names_for_input_output_statistics(node, graph)
            input_port_id, _ = self._backend_entity.get_activation_port_ids_for_bias_node(node)

            input_fp, input_shape = self._get_fp_inputs(statistic_points, in_node_name)

            output_fp = self._get_fp_outputs(statistic_points, out_node_name)

            # In case of the matrix multiplication layers, this is crucial to know the correct input port.
            input_id = (in_node_name, input_port_id)
            # Outputs of the subgraphs for the FastBiasCorrection are the same across the backends.
            output_id = (out_node_name, 0)

            extracted_model = self._backend_entity.extract_submodel(model_transformer, input_id, output_id)
            if extracted_model is None:
                nncf_logger.debug(f"Skipping node {node_name} because cant extract submodel")
                continue

            sub_input_name, sub_output_name = self._backend_entity.get_sub_input_output_names(extracted_model)

            output_channel_axis = node.metatype.output_channel_axis
            input_channel_axis = self._backend_entity.get_activation_channel_axis(node, input_port_id, input_shape)
            if bias_value.ndim > 1:
                # Make index positive
                output_channel_axis = range(bias_value.ndim)[output_channel_axis]
                input_channel_axis = range(bias_value.ndim)[input_channel_axis]
            input_blob = self._backend_entity.create_input_data(
                input_shape, input_fp, sub_input_name, input_channel_axis
            )
            bias_shift = self._get_bias_shift(
                model=extracted_model,
                input_blob=input_blob,
                output_channel_axis=output_channel_axis,
                output_fp=output_fp,
                output_name=sub_output_name,
            )

            bias_shift = self._reshape_bias_shift(bias_shift, bias_value, output_channel_axis)
            updated_bias = bias_value + bias_shift
            magnitude = self._get_bias_shift_magnitude(bias_value, updated_bias)

            if magnitude < self.threshold:
                nncf_logger.debug(f"{node_name} bias would be changed")
                node_and_new_bias_value.append((node, updated_bias))
            else:
                nncf_logger.debug(f"{node_name} bias skipped by threshold. Magnitude: {magnitude}")

        # Create commands of bias correction and apply them to the model.
        transformation_layout = TransformationLayout()
        for node, bias_value in node_and_new_bias_value:
            transformation_layout.register(self._backend_entity.create_bias_correction_command(node, bias_value, graph))
        transformed_model = model_transformer.transform(transformation_layout)

        return transformed_model

    @staticmethod
    def _get_bias_shift_magnitude(current_bias_value: Tensor, updated_bias_value: Tensor) -> Tensor:
        """
        Calculates bias shift magnitude based on the current and updated values.

        :param current_bias_value: The original bias value.
        :param updated_bias_value: The updated bias value.
        :return: Magnitude between original and updated bias values.
        """
        bias_shift_magnitude = inf
        if fns.count_nonzero(current_bias_value == 0) == 0:
            bias_shift_magnitude = fns.max(fns.abs((updated_bias_value - current_bias_value) / current_bias_value))
        return bias_shift_magnitude

    @staticmethod
    def _reshape_bias_shift(bias_shift: Tensor, bias_value: Tensor, channel_axis: int) -> Tensor:
        """
        Reshape bias_shift tensor in case of dimensions of bias_value is more then 1.

        :param bias_shift: Bias shift tensor.
        :param bias_value: Bias value tensor.
        :param channel_axis: Axis to update bias.

        :return TTensor: Updated bias_shift.
        """
        if bias_value.ndim > 1:
            new_shape = [1] * bias_value.ndim
            new_shape[channel_axis] = bias_shift.shape[0]
            bias_shift = bias_shift.reshape(new_shape)
        return bias_shift

    def _get_fp_inputs(self, statistic_points: StatisticPointsContainer, node_name: str) -> Tuple[List, List]:
        """
        Makes out per-layer needed data from the floating-point collected statistics.

        :param statistic_points: Filled StatisticPointsContainer.
        :param node_name: Name of the current layer.
        :return: Collected mean tensor data and shape for the further bias calculation.
        """

        def input_filter_func(point):
            return self._algorithm_key in point.algorithm_to_tensor_collectors and point.target_point.type in [
                TargetType.PRE_LAYER_OPERATION,
                TargetType.OPERATOR_PRE_HOOK,
            ]

        input_fp = []
        input_shape = []
        for tensor_collector in statistic_points.get_algo_statistics_for_node(
            node_name, input_filter_func, self._algorithm_key
        ):
            statistics = tensor_collector.get_statistics()
            input_fp.extend(statistics.mean_values)
            input_shape.extend(statistics.shape)
        return input_fp, input_shape

    def _get_fp_outputs(self, statistic_points: StatisticPointsContainer, node_name: str) -> List[TTensor]:
        """
        Makes out per-layer needed data from the floating-point collected statistics.

        :param statistic_points: Filled StatisticPointsContainer.
        :param node_name: Name of the current layer.
        :return: Collected mean tensor data for the further bias calculation.
        """

        def output_filter_func(point):
            return self._algorithm_key in point.algorithm_to_tensor_collectors and point.target_point.type in [
                TargetType.POST_LAYER_OPERATION,
                TargetType.OPERATOR_POST_HOOK,
            ]

        output_fp = []
        for tensor_collector in statistic_points.get_algo_statistics_for_node(
            node_name, output_filter_func, self._algorithm_key
        ):
            output_fp.extend(tensor_collector.get_statistics().mean_values)
        return output_fp

    def _add_statistic_point(self, container: StatisticPointsContainer, point: TargetPoint, axis: int) -> None:
        """
        Adds specific statistic point.

        :param container: StatisticPointsContainer instance.
        :param point: TargetPoint for statistic collection.
        :param axis: Channel axis for the statistics calculation.
        """
        stat_collector = self._backend_entity.mean_statistic_collector(
            channel_axis=axis, num_samples=self.subset_size, inplace=self.inplace_statistics
        )
        container.add_statistic_point(
            StatisticPoint(target_point=point, tensor_collector=stat_collector, algorithm=self._algorithm_key)
        )

    def _get_bias_shift(
        self,
        model: TModel,
        input_blob: Union[TTensor, Dict[str, TTensor]],
        output_channel_axis: Tuple[int],
        output_fp: List[TTensor],
        output_name: str,
    ) -> TTensor:
        """
        Calculates updated bias.

        :param engine: Backend-specific engine instance for the model execution.
        :param model: Backend-specific sub-model for the execution.
        :param input_blob: Input data for the execution.
        :param output_channel_axis: Channel axis for the raw output data aggregation.
        :param output_fp: Output data for the shift calculation.
        :param output_name: Name of the output tensor for the data collection.
        :return: Calculated bias shift.
        """
        engine = EngineFactory.create(model)
        raw_output = engine.infer(input_blob)
        q_outputs = self._backend_entity.process_model_output(raw_output, output_name)
        q_outputs = mean_per_channel(q_outputs, output_channel_axis)
        bias_shift = fns.stack(output_fp) - q_outputs
        return bias_shift

    def get_statistic_points(self, model: TModel, graph: NNCFGraph) -> StatisticPointsContainer:
        self._set_backend_entity(model)
        nodes_with_bias = [
            node for node in graph.get_all_nodes() if self._backend_entity.is_node_with_bias(node, graph)
        ]

        statistic_container = StatisticPointsContainer()
        for node in nodes_with_bias:
            input_port_id, output_port_id = self._backend_entity.get_activation_port_ids_for_bias_node(node)
            in_node_name, out_node_name = self._backend_entity.get_node_names_for_input_output_statistics(node, graph)

            pre_layer_statistic_point = self._backend_entity.target_point(
                TargetType.PRE_LAYER_OPERATION, in_node_name, input_port_id
            )
            post_layer_statistic_point = self._backend_entity.target_point(
                TargetType.POST_LAYER_OPERATION, out_node_name, output_port_id
            )
            input_shape = graph.get_input_edges(node)[input_port_id].tensor_shape
            input_channel_axis = self._backend_entity.get_activation_channel_axis(node, input_port_id, input_shape)

            self._add_statistic_point(statistic_container, pre_layer_statistic_point, input_channel_axis)
            self._add_statistic_point(
                statistic_container, post_layer_statistic_point, node.metatype.output_channel_axis
            )

        return statistic_container
