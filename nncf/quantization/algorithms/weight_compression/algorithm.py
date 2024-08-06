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

from collections import defaultdict
from typing import Dict, List, Optional, OrderedDict, Tuple, TypeVar

import nncf
from nncf import Dataset
from nncf.common.factory import StatisticsAggregatorFactory
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.logging import nncf_logger
from nncf.common.logging.track_progress import track
from nncf.common.scopes import should_consider_scope
from nncf.common.tensor_statistics.statistic_point import StatisticPoint
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.common.utils.helpers import create_table
from nncf.parameters import CompressWeightsMode
from nncf.parameters import SensitivityMetric
from nncf.quantization.advanced_parameters import AdvancedCompressionParameters
from nncf.quantization.algorithms.algorithm import Algorithm
from nncf.quantization.algorithms.weight_compression.awq import AWQ
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionParameters
from nncf.quantization.algorithms.weight_compression.gptq import GPTQ
from nncf.quantization.algorithms.weight_compression.mixed_precision import MIXED_PRECISION_CRITERIA
from nncf.quantization.algorithms.weight_compression.scale_estimation import ScaleEstimation
from nncf.quantization.algorithms.weight_compression.weight_lowering import WeightCompressionConfig
from nncf.scopes import IgnoredScope
from nncf.scopes import get_ignored_node_names_from_ignored_scope
from nncf.tensor import Tensor
from nncf.tensor.definitions import TensorDataType

TModel = TypeVar("TModel")
TTensor = TypeVar("TTensor")


class WeightCompression(Algorithm):
    """
    Post-training Weight Compression algorithm implementation.

    Compresses weights of Linear and Embedding layers to 8-bit integer or
    to 4-bit integer/float depending on mode, ratio and group size.
    """

    def __init__(
        self,
        mode: CompressWeightsMode,
        ratio: float,
        group_size: int,
        ignored_scope: IgnoredScope,
        all_layers: bool,
        sensitivity_metric: SensitivityMetric,
        awq: bool,
        subset_size: int,
        scale_estimation: bool,
        gptq: bool,
        advanced_parameters: Optional[AdvancedCompressionParameters] = None,
    ):
        """
        :param mode: Defines a mode for weight compression.
            INT8_SYM stands for 8-bit integer symmetric quantization of all weights.
                Weights are quantized symmetrically without zero point.
            INT8_ASYM is the same as INT8_SYM mode, but weights are quantized to a primary precision asymmetrically
                with a typical non-fixed zero point.
            INT4_SYM stands for a mixed-precision weights quantization with 4-bit integer as a primary precision.
                Weights are quantized to a primary precision symmetrically without zero point.
                All embeddings and the last layer are always compressed to a backup precision, which is INT8_ASYM,
                by default. All others are quantized whether to 4-bit integer or to a backup precision depending on
                criteria and the given ratio.
            INT4_ASYM is the same as INT4_SYM mode, but weights are quantized to a primary precision asymmetrically
                with a typical non-fixed zero point.
            NF4 is the same as INT4_SYM mode, but primary precision is NF4 data type without zero point.
            E2M1 is the same as INT4_SYM mode, but primary precision is E2M1 data type without zero point.
        :param ratio: the ratio between primary and backup precisions (e.g. 0.9 means 90% of layers quantized to NF4
            and the rest to INT8_ASYM).
        :param group_size: number of weights (e.g. 128) in the channel dimension
            that share quantization parameters (scale). The value -1 means no grouping.
        :param ignored_scope: An ignored scope that defined the list of model control
            flow graph nodes to be ignored during quantization.
        :param all_layers: Indicates whether embeddings and last MatMul layers should be compressed to a primary
            precision. By default, the backup precision is assigned for the embeddings and last MatMul layers.
        :param sensitivity_metric: The sensitivity metric for assigning quantization precision to layers. In order to
            preserve the accuracy of the model, the more sensitive layers receives a higher precision.
        :param awq: determines whether to use or not modified AWQ algorithm.
        :param subset_size: Number of data samples to calculate activation statistics used for assigning different
            quantization precision.
        :param scale_estimation: determines whether to use or not scale estimation for 4 bit layers.
        :param gptq: determines whether to use or not GPTQ algorithm.
        :param advanced_parameters: advanced parameters for algorithms in compression pipeline.
        """
        super().__init__()
        self._mode = mode
        self._group_size = group_size
        self._ratio = ratio
        self._ignored_scope = ignored_scope
        self._backend_entity = None
        self._algorithm_key = f"CW_{hash(self)}"
        self._fp_inputs = defaultdict(list)
        self._all_layers = all_layers
        self._sensitivity_metric = sensitivity_metric
        self._awq = awq
        self._subset_size = subset_size
        self._scale_estimation = scale_estimation
        self._gptq = gptq
        self._advanced_parameters = (
            advanced_parameters if advanced_parameters is not None else AdvancedCompressionParameters()
        )

        if self._gptq:
            gptq_params = self._advanced_parameters.gptq_params
            self._gptq_algo = GPTQ(gptq_params.damp_percent, gptq_params.block_size, gptq_params.subset_size)
            self._gptq_statistics = None

    @property
    def available_backends(self) -> List[BackendType]:
        return [BackendType.OPENVINO, BackendType.TORCH]

    def _set_backend_entity(self, model: TModel) -> None:
        """
        Creates a helper class with a backed-specific logic of the algorithm.

        :param model: Backend-specific input model.
        """
        model_backend = get_backend(model)
        if model_backend == BackendType.OPENVINO:
            from nncf.quantization.algorithms.weight_compression.openvino_backend import OVWeightCompressionAlgoBackend

            self._backend_entity = OVWeightCompressionAlgoBackend(model)
        elif model_backend == BackendType.TORCH:
            from nncf.quantization.algorithms.weight_compression.torch_backend import PTWeightCompressionAlgoBackend

            self._backend_entity = PTWeightCompressionAlgoBackend()
        else:
            raise nncf.UnsupportedBackendError(
                "Cannot return backend-specific entity because {} is not supported!".format(model_backend.value)
            )

    def _get_nodes_to_compress(self, nncf_graph: NNCFGraph) -> List[NNCFNode]:
        """
        Collects nodes in the model's graph corresponding to the layers for weight compression.

        :param nncf_graph: NNCFGraph instance.
        :return: List with the data for each layer.
        """
        weighted_metatypes = (
            self._backend_entity.matmul_metatypes
            + self._backend_entity.embedding_metatypes
            + self._backend_entity.convolution_metatypes
        )

        ordered_nodes_to_compress = []
        ignored_names = get_ignored_node_names_from_ignored_scope(
            self._ignored_scope, nncf_graph, strict=self._ignored_scope.validate
        )
        for node in nncf_graph.topological_sort():
            is_node_with_weights = self._backend_entity.is_node_with_weights(node, nncf_graph)
            is_within_scope = should_consider_scope(node.node_name, ignored_names)
            if node.metatype in weighted_metatypes and is_node_with_weights and is_within_scope:
                ordered_nodes_to_compress.append(node)
        return ordered_nodes_to_compress

    def _get_ratio_defining_params(
        self, all_weight_params: List[WeightCompressionParameters], is_last_layer_shared: bool
    ) -> List[WeightCompressionParameters]:
        """
        Returns the information about weights that are used for ratio calculation between primary
        and backup precisions.

        :param all_weight_params: List of all weight parameters.
        :param is_last_layer_shared: Indicates whether the last layer which shares the weight
            should be quantized or not.
        :return: Information about each weight node that is considered for mixed precision.
        """
        if self._mode in [CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM]:
            return all_weight_params

        ratio_defining_params = list(
            filter(
                lambda wp: wp.node_with_weight.metatype in self._backend_entity.matmul_metatypes,
                all_weight_params,
            )
        )

        # The last MatMul layer is quantized to 4-bits if all_layers=True
        if not self._all_layers and not is_last_layer_shared:
            ratio_defining_params = ratio_defining_params[:-1]

        # Embedding layers are quantized to 4-bits only if all_layers=True.
        if self._all_layers:
            embedding_params = list(
                filter(
                    lambda wp: wp.node_with_weight.metatype in self._backend_entity.embedding_metatypes
                    and len(wp.reduction_axes) == 1,
                    all_weight_params,
                )
            )
            ratio_defining_params.extend(embedding_params)

        return ratio_defining_params

    def _set_weight_compression_config(
        self,
        ratio_defining_params: List[WeightCompressionParameters],
        model: TModel,
        graph: NNCFGraph,
        activations: Optional[Dict[str, List[Tensor]]] = None,
    ) -> None:
        """
        Sets the appropriate compression configuration for weights based on some criteria.

        :param ratio_defining_params: Information about weights that are used for calculating ratio between primary and
            backup precisions.
        :param model: The model.
        :param graph: The model graph associated with the model.
        :param activations: The input activations of the layers considered for compression.
        """
        primary_config = WeightCompressionConfig(mode=self._mode, group_size=self._group_size)
        if self._ratio == 1:
            for weight_param in ratio_defining_params:
                weight_param.compression_config = primary_config
        else:
            criterion_cls = MIXED_PRECISION_CRITERIA.get(self._sensitivity_metric)
            criterion = criterion_cls(
                model, graph, self._backend_entity, ratio_defining_params, primary_config, self._ratio, activations
            )
            criterion.assign_mixed_precision()

    @staticmethod
    def _proportion_str(num_weights_list: List[int], total_num_weights: int, total_num_params: int) -> str:
        """
        Generates a string with proportion between target parameters and all model parameters by number of weights.

        :param num_weights_list: List of number of weights of target model parameters.
        :param total_num_weights: The total number of weights.
        :param total_num_params: The total number of model parameters.
        :return: The string with proportion between target parameters and all model parameters by number of weights.
        """
        percentage = sum(num_weights_list) / max(total_num_weights, 1) * 100
        return f"{percentage:.0f}% ({len(num_weights_list)} / {total_num_params})"

    def _get_bitwidth_distribution_str(
        self, all_params: List[WeightCompressionParameters], ratio_defining_params: List[WeightCompressionParameters]
    ) -> str:
        """
        Generates a table that shows the ratio of weights quantized to different number of bits.

        :param all_params: Information about each weight node.
        :param ratio_defining_params: Information about weights that are used for calculating ratio between primary and
            backup precisions.
        :return: A string containing the table.
        """
        num_bits_vs_num_weights_map = {}
        ratio_defining_weight_names = set(wp.weight_name for wp in ratio_defining_params)
        for data in all_params:
            num_bits = data.compression_config.num_bits
            n_total, n_ratio_defining = num_bits_vs_num_weights_map.get(num_bits, ([], []))
            if data.weight_name in ratio_defining_weight_names:
                n_ratio_defining.append(data.num_weights)
            n_total.append(data.num_weights)
            num_bits_vs_num_weights_map[num_bits] = (n_total, n_ratio_defining)

        num_ratio_defining_weights = sum(ws.num_weights for ws in ratio_defining_params)
        num_ratio_defining_params = len(ratio_defining_params)
        num_total_weights = sum(ws.num_weights for ws in all_params)
        num_params = len(all_params)
        num_bits_vs_num_weights_map = OrderedDict(sorted(num_bits_vs_num_weights_map.items(), reverse=True))
        # Table creation
        header = ["Num bits (N)", "% all parameters (layers)", "% ratio-defining parameters (layers)"]
        rows = []
        for bitwidth, (n_total, n_ratio_defining) in num_bits_vs_num_weights_map.items():
            rows.append(
                [
                    bitwidth,
                    self._proportion_str(n_total, num_total_weights, num_params),
                    self._proportion_str(n_ratio_defining, num_ratio_defining_weights, num_ratio_defining_params),
                ]
            )

        table = create_table(header, rows)
        pretty_string = f"Statistics of the bitwidth distribution:\n{table}"
        return pretty_string

    def apply(
        self,
        model: TModel,
        graph: NNCFGraph,
        statistic_points: Optional[StatisticPointsContainer] = None,
        dataset: Optional[Dataset] = None,
    ) -> TModel:
        self._set_backend_entity(model)
        nodes_to_compress = self._get_nodes_to_compress(graph)

        activations = {}
        if dataset is not None and self._sensitivity_metric != SensitivityMetric.WEIGHT_QUANTIZATION_ERROR:
            activations = self._get_activations(dataset, self._subset_size, nodes_to_compress, graph, model)
        all_weight_params: List[WeightCompressionParameters] = []
        weight_names = set()

        is_last_layer_shared = False
        n = len(nodes_to_compress)
        for i, node in enumerate(nodes_to_compress):
            for weight_name, weight_port_id in self._backend_entity.get_weight_names_and_port_ids(node, graph):
                if weight_name in weight_names:
                    if i == n - 1:
                        is_last_layer_shared = True
                    continue

                weight = self._backend_entity.get_weight(node, weight_port_id, model, graph)
                if weight.dtype not in [
                    TensorDataType.float16,
                    TensorDataType.bfloat16,
                    TensorDataType.float32,
                    TensorDataType.float64,
                ]:
                    continue
                reduction_axes = self._backend_entity.get_reduction_axes(node, weight_port_id, graph)
                if (
                    self._group_size != -1
                    and self._all_layers
                    and node.metatype in self._backend_entity.embedding_metatypes
                    and isinstance(reduction_axes, tuple)
                    and len(reduction_axes) != 1
                ):
                    # NNCF supports multiple reduction axes only for ops with group_size != -1.
                    # Convolution ops are always quantized to 8-bits (without groups).
                    # Embedding layers are quantized to 4-bits only if all_layers=True.
                    # MatMul ops can't have multiple reduction axes.
                    nncf_logger.warning(
                        f"Weight compression expects a single reduction axis, but {len(reduction_axes)} given. "
                        f"Weight shape: {weight.shape}, reduction axes: {reduction_axes}, "
                        f"node name: {node.node_name}. The node will be asymmetrically quantized to 8 bits."
                    )

                weight_params = WeightCompressionParameters(
                    weight_name, node, weight_port_id, weight.size, reduction_axes
                )
                all_weight_params.append(weight_params)
                weight_names.add(weight_name)

        ratio_defining_params = self._get_ratio_defining_params(all_weight_params, is_last_layer_shared)
        self._set_weight_compression_config(ratio_defining_params, model, graph, activations)
        nncf_logger.info(self._get_bitwidth_distribution_str(all_weight_params, ratio_defining_params))

        if (
            self._awq
            and activations is not None
            and self._mode not in [CompressWeightsMode.NF4, CompressWeightsMode.E2M1]
        ):
            awq_params = self._advanced_parameters.awq_params
            awq_algo = AWQ(
                model,
                self._backend_entity.name_to_node_mapping,
                all_weight_params,
                nodes_to_compress,
                activations,
                awq_params.subset_size,
                awq_params.percent_to_apply,
                awq_params.alpha_min,
                awq_params.alpha_max,
                awq_params.steps,
            )
            awq_algo.apply(model, graph)

        scales = {}
        zero_points = {}
        if (
            self._scale_estimation
            and activations is not None
            and self._mode not in [CompressWeightsMode.NF4, CompressWeightsMode.E2M1]
        ):
            scale_estimation_params = self._advanced_parameters.scale_estimation_params
            scale_algo = ScaleEstimation(
                model,
                self._backend_entity.name_to_node_mapping,
                all_weight_params,
                nodes_to_compress,
                activations,
                scale_estimation_params.subset_size,
                scale_estimation_params.initial_steps,
                scale_estimation_params.scale_steps,
                scale_estimation_params.weight_penalty,
            )
            scales = scale_algo.apply(model, graph)

        if self._gptq:
            model, scales, zero_points = self._gptq_algo.apply(
                model=model,
                graph=graph,
                dataset=dataset,
                weight_compression_parameters=all_weight_params,
                statistic_points=self._gptq_statistics,
                backend_entity=self._backend_entity,
            )

        # Sort weight params to start compression with the bigger constants. This lowers peak memory footprint.
        all_weight_params = sorted(all_weight_params, key=lambda wp: wp.num_weights, reverse=True)

        # Compress model using weight compression parameters
        transformed_model = self._backend_entity.transform_model(
            model,
            graph,
            track(all_weight_params, description="Applying Weight Compression"),
            scales,
            zero_points,
        )

        self._backend_entity.dump_parameters(
            model,
            parameters={
                "mode": self._mode.value,
                "group_size": self._group_size,
                "ratio": self._ratio,
                "all_layers": self._all_layers,
                "ignored_scope": self._ignored_scope,
                "sensitivity_metric": self._sensitivity_metric.value,
                "awq": self._awq,
                "scale_estimation": self._scale_estimation,
                "gptq": self._gptq,
            },
            algo_name="weight_compression",
        )
        return transformed_model

    def get_statistic_points(self, model: TModel, graph: NNCFGraph) -> StatisticPointsContainer:
        pass

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

    def _get_fp_inputs(self, statistic_points: StatisticPointsContainer, node_name: str, port_id: int) -> List[Tensor]:
        """
        Collects floating-point statistics for the given node and port id.

        :param statistic_points: Filled StatisticPointsContainer.
        :param node_name: Name of the current layer.
        :param port_id: Port id for statistics collection.
        :return: Collected list of tensor data.
        """

        def input_filter_func(point):
            # For the floating-point statistics collected in POST_LAYER style,
            # we also need to determine the output port id.
            # For the cases when the layer has more than one (0) output port.
            return (
                self._algorithm_key in point.algorithm_to_tensor_collectors
                and point.target_point.type == TargetType.POST_LAYER_OPERATION
                and point.target_point.port_id == port_id
            )

        input_id = (node_name, port_id)
        if input_id in self._fp_inputs:
            return self._fp_inputs[input_id]

        input_fp = []
        for tensor_collector in statistic_points.get_algo_statistics_for_node(
            node_name, input_filter_func, self._algorithm_key
        ):
            for value in tensor_collector.get_statistics().values:
                input_fp.append(value)
        self._fp_inputs[input_id] = input_fp
        return self._fp_inputs[input_id]

    def _get_activations(
        self, dataset: Dataset, subset_size: int, nodes_to_compress: List[NNCFNode], graph: NNCFGraph, model: TModel
    ) -> Dict[str, List[Tensor]]:
        """
        Collects input activations for the given nodes on the dataset.

        :param dataset: Dataset to collect values.
        :param subset_size: Number of data samples to calculate activation statistics used for assigning different
            quantization precision.
        :param nodes_to_compress: List of nodes, whose inputs are collected.
        :param model: Model for statistics collection.
        :param graph: Model graph.
        :return: statistics values itself per node name.
        """
        activations = {}
        _collected_stat_inputs_map = {}
        statistic_container = StatisticPointsContainer()
        all_act_nodes = set()
        act_vs_shared_node_names_mapping = defaultdict(list)
        matmul_metatypes = self._backend_entity.matmul_metatypes
        filtered_nodes = filter(lambda node: node.metatype in matmul_metatypes, nodes_to_compress)
        for node in filtered_nodes:
            act_node, output_port_id = self._get_activation_node_and_port(node, graph)
            act_node_name = act_node.node_name
            if act_node_name in all_act_nodes:
                act_vs_shared_node_names_mapping[act_node_name].append(node.node_name)
                continue
            all_act_nodes.add(act_node_name)
            output_id = (act_node_name, output_port_id)
            _collected_stat_inputs_map[node.node_name] = output_id

            statistic_point = self._backend_entity.target_point(
                TargetType.POST_LAYER_OPERATION, act_node_name, port_id=output_port_id
            )
            stat_collector = self._backend_entity.raw_statistic_collector(num_samples=subset_size)
            statistic_container.add_statistic_point(
                StatisticPoint(
                    target_point=statistic_point, tensor_collector=stat_collector, algorithm=self._algorithm_key
                )
            )

        statistics_aggregator = StatisticsAggregatorFactory.create(model, dataset)
        statistics_aggregator.register_statistic_points(statistic_container)

        if self._gptq:
            self._gptq_statistics = self._gptq_algo.get_statistic_points(
                model, graph, nodes_to_compress, self._backend_entity
            )
            statistics_aggregator.register_statistic_points(self._gptq_statistics)

        statistics_aggregator.collect_statistics(model, graph)

        for node_name, output_id in _collected_stat_inputs_map.items():
            act_node_name, output_port_id = output_id
            x_fp = self._get_fp_inputs(statistic_container, node_name=act_node_name, port_id=output_port_id)
            x_fp = [i.squeeze() for i in x_fp]  # List[tensor(seq_length, hidden_dim)]
            activations[node_name] = x_fp

            for shared_node_name in act_vs_shared_node_names_mapping[act_node_name]:
                activations[shared_node_name] = x_fp

        return activations
