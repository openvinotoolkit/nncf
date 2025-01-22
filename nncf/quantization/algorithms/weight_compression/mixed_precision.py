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

from abc import ABC
from abc import abstractmethod
from typing import Iterable, List, Optional, Tuple, TypeVar

import nncf
from nncf import Dataset
from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.logging.track_progress import track
from nncf.common.tensor_statistics.statistic_point import StatisticPoint
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.common.utils.registry import Registry
from nncf.parameters import SensitivityMetric
from nncf.quantization.algorithms.algorithm import Algorithm
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionConfig
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionParameters
from nncf.quantization.algorithms.weight_compression.weight_lowering import do_int_dequantization
from nncf.quantization.algorithms.weight_compression.weight_lowering import do_int_quantization
from nncf.quantization.algorithms.weight_compression.weight_lowering import get_integer_quantization_error
from nncf.tensor import Tensor
from nncf.tensor import functions as fns
from nncf.tensor.definitions import TensorDataType

TModel = TypeVar("TModel")
MIXED_PRECISION_CRITERIA = Registry("mixed_precision_criteria")
THE_LOWEST_SENSITIVITY = 0


class MixedPrecisionCriterion(Algorithm):
    """
    Assigns mixed quantization scheme (e.g. uniform int8 or uniform int4/non-uniform fp4)
    for weights based on some criteria.
    """

    def __init__(self, primary_config: WeightCompressionConfig, ratio: float, subset_size: Optional[int] = None):
        """
        :param primary_config: Configuration on how to compress (quantize) weights to primary precision.
        :param ratio: The ratio between primary and backup precisions (e.g. 0.9 means 90% of layers quantized to NF4
            and the rest to INT8_ASYM).
        :param subset_size: Size of dataset subset for statistics.
        """
        self._primary_config = primary_config
        self._ratio = ratio
        self._subset_size = subset_size
        self._algorithm_key = f"MPC_{hash(self)}"
        self._backend_entity = None

    @abstractmethod
    def _calc_sensitivity(
        self,
        model: TModel,
        graph: NNCFGraph,
        weight_params: List[WeightCompressionParameters],
        statistic_points: Optional[StatisticPointsContainer] = None,
    ) -> List[float]:
        """
        Calculates sensitivity of each layer according to a criterion.

        :return: List of values per node to be quantized.
        """

    def apply(
        self,
        model: TModel,
        graph: NNCFGraph,
        statistic_points: Optional[StatisticPointsContainer] = None,
        dataset: Optional[Dataset] = None,
        weight_params: List[WeightCompressionParameters] = None,
    ) -> None:
        """
        Assigns quantization precision based on computed layers' sensitivities, ratio of parameters.
        """
        self._set_backend_entity(model)

        scores = self._calc_sensitivity(model, graph, weight_params, statistic_points)
        num_all_weights = sum(wp.num_weights for wp in weight_params)

        indexes_of_layers_in_ascending_order_of_scores = [
            i[0] for i in sorted(enumerate(scores), reverse=False, key=lambda x: x[1])
        ]
        num_weights_in_4bit = 0
        for index in indexes_of_layers_in_ascending_order_of_scores:
            weight_param = weight_params[index]
            current_ratio = (num_weights_in_4bit + weight_param.num_weights) / num_all_weights
            if current_ratio >= self._ratio:
                break
            weight_param.compression_config = self._primary_config
            num_weights_in_4bit += weight_param.num_weights

    @abstractmethod
    def _set_backend_entity(self, model: TModel) -> None:
        """
        Creates a helper class with a backed-specific logic of the algorithm.

        :param model: Backend-specific input model.
        """

    @abstractmethod
    def get_statistic_points(
        self,
        model: TModel,
        graph: NNCFGraph,
        nodes_and_port_ids: Iterable[Tuple[NNCFNode, int]],
    ) -> StatisticPointsContainer:
        """
        Returns statistic points, for which StatisticsCollector should collect statistics.

        :param model: Model for statistics collection.
        :param graph: Model graph.
        :param nodes_and_port_ids: Nodes and port ids for which statistics should be collected.
        :return: Statistic points, for which StatisticsCollector should collect statistics.
        """


@MIXED_PRECISION_CRITERIA.register(SensitivityMetric.WEIGHT_QUANTIZATION_ERROR)
class DataFreeCriterion(MixedPrecisionCriterion):
    """
    A baseline mixed precision criterion that is based on quantization noise of weights only.
    """

    @property
    def available_backends(self) -> List[BackendType]:
        return [BackendType.OPENVINO, BackendType.TORCH, BackendType.TORCH_FX]

    def _set_backend_entity(self, model: TModel) -> None:
        model_backend = get_backend(model)
        if model_backend == BackendType.OPENVINO:
            from nncf.quantization.algorithms.weight_compression.openvino_backend import OVWeightCompressionAlgoBackend

            self._backend_entity = OVWeightCompressionAlgoBackend(model)
        elif model_backend == BackendType.TORCH:
            from nncf.quantization.algorithms.weight_compression.torch_backend import PTWeightCompressionAlgoBackend

            self._backend_entity = PTWeightCompressionAlgoBackend()
        elif model_backend == BackendType.TORCH_FX:
            from nncf.quantization.algorithms.weight_compression.torch_fx_backend import FXWeightCompressionAlgoBackend

            self._backend_entity = FXWeightCompressionAlgoBackend()
        else:
            raise nncf.UnsupportedBackendError(
                "Cannot return backend-specific entity because {} is not supported!".format(model_backend.value)
            )

    def _calc_weight_sensitivity(
        self,
        weight_param: WeightCompressionParameters,
        model: TModel,
        graph: NNCFGraph,
    ) -> float:
        weight = self._backend_entity.get_weight(
            weight_param.node_with_weight, weight_param.weight_port_id, model, graph
        )
        backup_config = WeightCompressionConfig()
        reduction_axes = weight_param.reduction_axes
        int_error = get_integer_quantization_error(weight, reduction_axes, backup_config)
        eps = fns.finfo(weight).eps
        return 1 / (int_error + eps)

    def _calc_score_per_node(
        self,
        weight_param: WeightCompressionParameters,
        model: TModel,
        graph: NNCFGraph,
        statistic_points: Optional[StatisticPointsContainer] = None,
    ) -> float:
        weight_score = self._calc_weight_sensitivity(weight_param, model, graph)
        return weight_score

    def _calc_sensitivity(
        self,
        model: TModel,
        graph: NNCFGraph,
        weight_params: List[WeightCompressionParameters],
        statistic_points: Optional[StatisticPointsContainer] = None,
    ) -> List[float]:
        scores = []
        for weight_param in track(weight_params, description="Mixed-Precision assignment"):
            scores.append(self._calc_score_per_node(weight_param, model, graph, statistic_points))
        return scores

    def get_statistic_points(
        self,
        model: TModel,
        graph: NNCFGraph,
        nodes_and_port_ids: Iterable[Tuple[NNCFNode, int]],
    ) -> StatisticPointsContainer:
        raise RuntimeError("No statistics collection intended for data-free mixed precision criterion")


class DataBasedCriterion(DataFreeCriterion, ABC):
    """
    Data-based mixed precision criterion that takes into account outliers in the input statistics.
    Expecting statistics of the following shape: [hidden_dim]
    """

    STAT_KEY = None

    @property
    def available_backends(self) -> List[BackendType]:
        return [BackendType.OPENVINO]

    def _set_backend_entity(self, model: TModel) -> None:
        model_backend = get_backend(model)
        if model_backend == BackendType.OPENVINO:
            from nncf.quantization.algorithms.weight_compression.openvino_backend import OVMixedPrecisionAlgoBackend

            self._backend_entity = OVMixedPrecisionAlgoBackend(model)
        else:
            raise nncf.UnsupportedBackendError(
                "Cannot return backend-specific entity because {} is not supported!".format(model_backend.value)
            )

    def _calc_activation_sensitivity(
        self,
        weight_param: WeightCompressionParameters,
        graph: NNCFGraph,
        statistic_points: StatisticPointsContainer,
    ) -> float:
        stats = self._get_statistics_for_node(statistic_points, weight_param.node_with_weight, graph, self.STAT_KEY)
        return stats[0].item()

    def _calc_score_per_node(
        self,
        weight_param: WeightCompressionParameters,
        model: TModel,
        graph: NNCFGraph,
        statistic_points: Optional[StatisticPointsContainer] = None,
    ):
        """
        NOTE: Data-based criteria for assigning 4-bit/8-bit precisions are valid for Matmul operations only.
        However, in some cases it can be beneficial to quantize Gather layers to 4-bit.
        Since there's no data-aware estimation of sensitivity in these layers, they receive the lowest sensitivity.
        It allows assigning Gather operation 4-bit in the first place.
        """
        if weight_param.node_with_weight.metatype in self._backend_entity.embedding_metatypes:
            return THE_LOWEST_SENSITIVITY
        weight_score = self._calc_weight_sensitivity(weight_param, model, graph)
        activation_score = self._calc_activation_sensitivity(weight_param, graph, statistic_points)
        return weight_score * activation_score

    def get_statistic_points(
        self,
        model: TModel,
        graph: NNCFGraph,
        nodes_and_port_ids: Iterable[Tuple[NNCFNode, int]],
    ) -> StatisticPointsContainer:
        self._set_backend_entity(model)

        statistic_container = StatisticPointsContainer()
        for act_node, output_port_id in nodes_and_port_ids:
            n_dims = len(graph.get_output_edges_by_port_id(act_node, output_port_id)[0].tensor_shape)
            if n_dims < 2:
                raise RuntimeError(
                    f"Data-aware mixed precision criteria are not supported for MatMuls with 1D inputs. "
                    f"Node: {act_node.node_name}, number of dimensions: {n_dims}."
                )
            statistic_point = self._backend_entity.target_point(
                TargetType.POST_LAYER_OPERATION, act_node.node_name, port_id=output_port_id
            )
            stat_collector = self._get_statistic_collector()
            statistic_container.add_statistic_point(
                StatisticPoint(
                    target_point=statistic_point, tensor_collector=stat_collector, algorithm=self._algorithm_key
                )
            )

        return statistic_container

    @abstractmethod
    def _get_statistic_collector(self):
        """
        Get statistic collector
        """

    def _get_activation_node_and_port(self, node: NNCFNode, nncf_graph: NNCFGraph) -> Tuple[NNCFNode, int]:
        """
        This method returns the activation layer and corresponding port id for the node.

        :param node: NNCFGraph node for which the activation is sought.
        :param nncf_graph: NNCFGraph instance with the node.
        :return: Tuple with the activation node and port id.
        """
        activation_port = self._backend_entity.get_activation_port_id(node, nncf_graph)
        activation_edge = nncf_graph.get_input_edge_by_port_id(node, activation_port)
        activation_node = activation_edge.from_node
        port_id = activation_edge.output_port_id
        return activation_node, port_id

    def _get_statistics_for_node(
        self, statistic_points: StatisticPointsContainer, node: NNCFNode, nncf_graph: NNCFGraph, stat_key: str
    ) -> List[Tensor]:
        act_node, output_port_id = self._get_activation_node_and_port(node, nncf_graph)

        def input_filter_func(point):
            # For the floating-point statistics collected in POST_LAYER style,
            # we also need to determine the output port id.
            # For the cases when the layer has more than one (0) output port.
            return (
                self._algorithm_key in point.algorithm_to_tensor_collectors
                and point.target_point.type == TargetType.POST_LAYER_OPERATION
                and point.target_point.port_id == output_port_id
            )

        stats = []
        for tensor_collector in statistic_points.get_algo_statistics_for_node(
            act_node.node_name, input_filter_func, self._algorithm_key
        ):
            statistics = tensor_collector.get_statistics()
            for data in statistics.get_data().values():
                if isinstance(data, Tensor):
                    stats.append(data)
                else:
                    stats.extend(data)
        return stats


@MIXED_PRECISION_CRITERIA.register(SensitivityMetric.HESSIAN_INPUT_ACTIVATION)
class HAWQCriterion(DataBasedCriterion):
    """
    Calculates the average Hessian trace of weights with respect to the layer-wise quantization error
    multiplied by L2 norm of 8-bit quantization noise.
    """

    STAT_KEY = SensitivityMetric.HESSIAN_INPUT_ACTIVATION.value

    def _calc_weight_sensitivity(
        self,
        weight_param: WeightCompressionParameters,
        model: TModel,
        graph: NNCFGraph,
    ) -> float:
        weight = self._backend_entity.get_weight(
            weight_param.node_with_weight, weight_param.weight_port_id, model, graph
        )
        backup_config = WeightCompressionConfig()
        reduction_axes = weight_param.reduction_axes

        orig_shape = weight.shape

        if weight.dtype != TensorDataType.float32:
            weight = weight.astype(TensorDataType.float32)

        compressed_weights, scale, zero_point = do_int_quantization(weight, reduction_axes, backup_config)
        decompressed_weight = do_int_dequantization(compressed_weights, scale, zero_point)
        decompressed_weight = decompressed_weight.reshape(orig_shape)
        return fns.linalg.norm(decompressed_weight - weight, ord="fro").item()

    def _get_statistic_collector(self):
        return self._backend_entity.hawq_statistic_collector(self._subset_size)


@MIXED_PRECISION_CRITERIA.register(SensitivityMetric.MEAN_ACTIVATION_VARIANCE)
class MeanVarianceCriterion(DataBasedCriterion):
    """
    The mean variance of the layers' inputs multiplied by inverted 8-bit quantization noise.
    """

    STAT_KEY = SensitivityMetric.MEAN_ACTIVATION_VARIANCE.value

    def _get_statistic_collector(self):
        # Reducing across the second-last dimension, assuming it is the sequence length dimension
        return self._backend_entity.mean_variance_statistic_collector(
            reduction_axes=(-2,), subset_size=self._subset_size
        )


@MIXED_PRECISION_CRITERIA.register(SensitivityMetric.MAX_ACTIVATION_VARIANCE)
class MaxVarianceCriterion(DataBasedCriterion):
    """
    The maximum variance of the layers' inputs multiplied by inverted 8-bit quantization noise.
    """

    STAT_KEY = SensitivityMetric.MAX_ACTIVATION_VARIANCE.value

    def _get_statistic_collector(self):
        # Reducing across the second-last dimension, assuming it is the sequence length dimension
        return self._backend_entity.max_variance_statistic_collector(
            reduction_axes=(-2,), subset_size=self._subset_size
        )


@MIXED_PRECISION_CRITERIA.register(SensitivityMetric.MEAN_ACTIVATION_MAGNITUDE)
class MeanMaxCriterion(DataBasedCriterion):
    """
    The mean magnitude of the layers' inputs multiplied by inverted 8-bit quantization noise.
    """

    STAT_KEY = SensitivityMetric.MEAN_ACTIVATION_MAGNITUDE.value

    def _get_statistic_collector(self):
        # Reducing across the second-last dimension, assuming it is the sequence length dimension
        return self._backend_entity.mean_abs_max_statistic_collector(
            reduction_axes=(-2,), subset_size=self._subset_size
        )
