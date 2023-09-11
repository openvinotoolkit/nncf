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

from itertools import islice
from typing import Callable, Dict, List, Optional, Tuple, TypeVar

from tqdm import tqdm

from nncf import Dataset
from nncf.common import factory
from nncf.common.deprecation import warning_deprecated
from nncf.common.engine import Engine
from nncf.common.factory import NNCFGraphFactory
from nncf.common.factory import StatisticsAggregatorFactory
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.model_transformer import ModelTransformer
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.logging import nncf_logger
from nncf.common.quantization.structs import QuantizationPreset
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import copy_model
from nncf.common.utils.backend import get_backend
from nncf.parameters import ModelType
from nncf.parameters import TargetDevice
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from nncf.quantization.algorithms.algorithm import Algorithm
from nncf.quantization.algorithms.bias_correction.algorithm import BIAS_CORRECTION_THRESHOLD
from nncf.quantization.algorithms.bias_correction.algorithm import BiasCorrection
from nncf.quantization.algorithms.channel_alignment.algorithm import ChannelAlignment
from nncf.quantization.algorithms.fast_bias_correction.algorithm import FAST_BIAS_CORRECTION_THRESHOLD
from nncf.quantization.algorithms.fast_bias_correction.algorithm import FastBiasCorrection
from nncf.quantization.algorithms.min_max.algorithm import MinMaxQuantization
from nncf.quantization.algorithms.smooth_quant.algorithm import SmoothQuant
from nncf.scopes import IgnoredScope

TModel = TypeVar("TModel")
TPass = Callable[[TModel], TModel]


class PostTrainingQuantization(Algorithm):
    """
    Implements Post-Training Quantization algorithm, which basically includes:
    1) ChannelAlignment
    2) MinMaxQuantization
    3) FastBiasCorrection or BiasCorrection
    """

    def __init__(
        self,
        preset: QuantizationPreset = QuantizationPreset.PERFORMANCE,
        target_device: TargetDevice = TargetDevice.ANY,
        subset_size: int = 300,
        fast_bias_correction: bool = True,
        model_type: Optional[ModelType] = None,
        ignored_scope: Optional[IgnoredScope] = None,
        advanced_parameters: Optional[AdvancedQuantizationParameters] = None,
    ):
        """
        :param preset: A preset that controls the quantization mode
            (symmetric and asymmetric). It can take the following values:
            - `performance`: Symmetric quantization of weights and activations.
            - `mixed`: Symmetric quantization of weights and asymmetric
            quantization of activations.
        :param target_device: A target device the specificity of which will be taken
            into account while compressing in order to obtain the best performance
            for this type of device.
        :param subset_size: Size of a subset to calculate activations
            statistics used for quantization.
        :param fast_bias_correction: Setting this option to `False` enables a different
            bias correction method which is more accurate, in general, and takes
            more time but requires less memory.
        :param model_type: Model type is needed to specify additional patterns
            in the model. Supported only `transformer` now.
        :param ignored_scope: An ignored scope that defined the list of model control
            flow graph nodes to be ignored during quantization.
        :param advanced_parameters: Advanced quantization parameters for
            fine-tuning the quantization algorithm
        """
        super().__init__()
        self.algorithms = []
        self.subset_size = subset_size
        self.first_stage_algorithms: List[Algorithm] = []

        if target_device is TargetDevice.VPU:
            warning_deprecated("VPU device is deprecated and will no longer be supported in the future.")

        if advanced_parameters is None:
            advanced_parameters = AdvancedQuantizationParameters()
        self.intermediate_model_dir = advanced_parameters.intermediate_model_dir

        if model_type == ModelType.TRANSFORMER and advanced_parameters.smooth_quant_alpha >= 0:
            smooth_quant_algorithm = SmoothQuant(
                subset_size=subset_size,
                inplace_statistics=advanced_parameters.inplace_statistics,
                alpha=advanced_parameters.smooth_quant_alpha,
            )
            self.first_stage_algorithms.append(smooth_quant_algorithm)

        if not advanced_parameters.disable_channel_alignment:
            channel_alignment = ChannelAlignment(
                subset_size=subset_size,
                inplace_statistics=advanced_parameters.inplace_statistics,
            )
            self.first_stage_algorithms.append(channel_alignment)

        min_max_quantization = MinMaxQuantization(
            preset=preset,
            target_device=target_device,
            subset_size=subset_size,
            model_type=model_type,
            ignored_scope=ignored_scope,
            overflow_fix=advanced_parameters.overflow_fix,
            quantize_outputs=advanced_parameters.quantize_outputs,
            inplace_statistics=advanced_parameters.inplace_statistics,
            activations_quantization_params=advanced_parameters.activations_quantization_params,
            weights_quantization_params=advanced_parameters.weights_quantization_params,
            activations_range_estimator_params=advanced_parameters.activations_range_estimator_params,
            weights_range_estimator_params=advanced_parameters.weights_range_estimator_params,
            backend_params=advanced_parameters.backend_params,
        )

        self.algorithms.append(min_max_quantization)

        if advanced_parameters.disable_bias_correction:
            return

        bias_correction_params = advanced_parameters.bias_correction_params
        if fast_bias_correction:
            threshold = FAST_BIAS_CORRECTION_THRESHOLD
            if bias_correction_params.threshold is not None:
                threshold = bias_correction_params.threshold
            bias_correction = FastBiasCorrection(
                subset_size=subset_size,
                threshold=threshold,
                apply_for_all_nodes=bias_correction_params.apply_for_all_nodes,
                inplace_statistics=advanced_parameters.inplace_statistics,
                backend_params=advanced_parameters.backend_params,
            )
        else:
            threshold = BIAS_CORRECTION_THRESHOLD
            if bias_correction_params.threshold is not None:
                threshold = bias_correction_params.threshold
            bias_correction_subset_size = max(int(subset_size * 0.2), 1)
            bias_correction = BiasCorrection(
                subset_size=bias_correction_subset_size,
                threshold=threshold,
                apply_for_all_nodes=bias_correction_params.apply_for_all_nodes,
                inplace_statistics=advanced_parameters.inplace_statistics,
                backend_params=advanced_parameters.backend_params,
            )

        self.algorithms.append(bias_correction)

    @property
    def available_backends(self) -> Dict[str, BackendType]:
        return

    def _set_backend_entity(self, model_backend: BackendType) -> None:
        """
        Creates a helper class with a backed-specific logic of the algorithm.

        :param model_backend: Backend.
        """
        if model_backend == BackendType.OPENVINO:
            from nncf.quantization.algorithms.post_training.openvino_backend import OVPostTrainingBackend

            self._backend_entity = OVPostTrainingBackend()
        else:
            raise RuntimeError(
                "Cannot return backend-specific entity because {} is not supported!".format(model_backend)
            )

    def get_statistic_points(self, model: TModel, graph: NNCFGraph) -> StatisticPointsContainer:
        if self.first_stage_algorithms:
            raise NotImplementedError(
                "Statistic points are not supported yet for SmoothQuant and ChannelAlignment algorithms."
            )

        output = StatisticPointsContainer()
        for algorithm in self.algorithms:
            for statistic_points in algorithm.get_statistic_points(model, graph).values():
                for statistic_point in statistic_points:
                    output.add_statistic_point(statistic_point)
        return output

    def _apply(
        self,
        model: TModel,
        graph: NNCFGraph,
        statistic_points: Optional[StatisticPointsContainer] = None,
        dataset: Optional[Dataset] = None,
    ) -> TModel:
        for algorithm in self.first_stage_algorithms:
            backend = get_backend(model)
            if isinstance(algorithm, SmoothQuant) and backend != BackendType.OPENVINO:
                nncf_logger.debug(f"{backend.name} does not support SmoothQuant algorithm yet.")
                continue

            if isinstance(algorithm, ChannelAlignment) and backend != BackendType.OPENVINO:
                nncf_logger.debug(f"{backend.name} does not support ChannelAlignment algorithm yet.")
                continue

            statistics_aggregator = StatisticsAggregatorFactory.create(model, dataset)
            algo_statistic_points = algorithm.get_statistic_points(model, graph)
            statistics_aggregator.register_statistic_points(algo_statistic_points)
            statistics_aggregator.collect_statistics(model, graph)
            model = algorithm.apply(model, graph, statistics_aggregator.statistic_points)
            graph = NNCFGraphFactory.create(model)

        if statistic_points is None:
            statistics_aggregator = StatisticsAggregatorFactory.create(model, dataset)
            for algorithm in self.algorithms:
                algo_statistic_points = algorithm.get_statistic_points(model, graph)
                statistics_aggregator.register_statistic_points(algo_statistic_points)

            statistics_aggregator.collect_statistics(model, graph)
            statistic_points = statistics_aggregator.statistic_points

        for algorithm in self.algorithms[:-1]:
            model = algorithm.apply(model, graph, statistic_points)
            graph = NNCFGraphFactory.create(model)
        # building the model graph is not required after the last algorithm
        model = self.algorithms[-1].apply(model, graph, statistic_points)

        return model

    def apply(
        self,
        model: TModel,
        graph: NNCFGraph,
        statistic_points: Optional[StatisticPointsContainer] = None,
        dataset: Optional[Dataset] = None,
    ) -> TModel:
        model_copy = copy_model(model)
        backend = get_backend(model)
        if backend in [BackendType.ONNX, BackendType.TORCH]:
            return self._apply(model_copy, graph, statistic_points, dataset)
        self._set_backend_entity(backend)
        if not self._has_if_op(graph, self._backend_entity.if_node_metatype):
            return self._apply(model_copy, graph, statistic_points, dataset)
        nncf_logger.info("The model has If operations. The iteratively each body of If operations will be quantized.")
        quantized_model, _ = self._dfs_quantize_models(model_copy, graph, dataset, statistic_points, 0)
        return quantized_model

    def _make_dataset_for_child_model(
        self,
        engine: Engine,
        calibration_dataset: Dataset,
        if_cond_input_name: str,
        child_model_input_names: List[str],
        if_submodel_condition: bool,
        model_cnt: int,
    ) -> Dataset:
        """
        Returns dataset for a child model.

        :param engine: Engine to infer parent model to obtain dataitems for a child dataset.
        :param calibration_dataset: Dataset to infer parent model.
        :param if_cond_input_name: Input name of If node condition.
        :param child_model_input_names: - Names of inputs for child model
        (should be in the order of passing them to a model).
        :param if_submodel_condition: If node submodel condition.
        :param model_cnt: Global counter of a child model.
        :return Dataset: Dataset for child model.
        """
        dataset = []
        calibration_dataset_size = min(self.subset_size, calibration_dataset.get_length())
        for input_data in tqdm(
            islice(calibration_dataset.get_inference_data(), calibration_dataset_size),
            total=calibration_dataset_size,
            desc=f"Collect dataset for {model_cnt} model:",
        ):
            data_item = []
            results = engine.infer(input_data)
            if (if_submodel_condition and results[if_cond_input_name]) or (
                not if_submodel_condition and not results[if_cond_input_name]
            ):
                for input_name in child_model_input_names:
                    data_item.append(results[input_name])
                dataset.append(data_item)
        nncf_logger.info(f"The final length of a dataset for {model_cnt} model is {len(dataset)}")
        return Dataset(dataset)

    def _has_if_op(self, nncf_graph: NNCFGraph, if_node_metatype: OperatorMetatype) -> bool:
        """
        Returns True if NNCGraph has If node.

        :param nncf_graph: NNCFGraph instance.
        :param if_node_metatype: backend-specific If node metatype.
        :return: True if NNCFGraph has If node, else - otherwise.
        """
        if nncf_graph.get_nodes_by_metatypes([if_node_metatype]):
            return True
        return False

    def _extract_if_submodel(
        self, model_transformer: ModelTransformer, if_node: NNCFNode, if_submodel_condition: bool
    ) -> TModel:
        """
        Returns if submodel of If node laying on an input port if_submodel_port_id of If node.

        :param model_transformer: ModelTransformer instance.
        :param if_node: If node.
        :param if_submodel_condition: If True returns True submodel of If node, otherwise - False submodel.
        :return: If submodel.
        """
        transformation_layout = TransformationLayout()
        command = self._backend_entity.create_extract_if_subgraph_command(if_node, if_submodel_condition)
        transformation_layout.register(command)
        return model_transformer.transform(transformation_layout)

    def _update_if_submodel(
        self, model_transformer: ModelTransformer, if_node: NNCFNode, if_submodel_condition: bool, submodel: TModel
    ) -> TModel:
        """
        Update submodel of If node.

        :param model_transformer: ModelTransformer instance.
        :param if_node: If node.
        :param if_submodel_condition: Condition of If node submodel.
        :param submodel: New submodel.
        :return: Updated model with a new submodel.
        """
        transformation_layout = TransformationLayout()
        command = self._backend_entity.create_update_subgraph_command(if_node, if_submodel_condition, submodel)
        transformation_layout.register(command)
        return model_transformer.transform(transformation_layout)

    def _add_outputs_before_if_node(
        self, model_transformer: ModelTransformer, model: TModel, if_node: NNCFNode
    ) -> TModel:
        """
        Inserts extra outputs on If node inputs.

        :param model_transformer: ModelTransformer instance.
        :param model: Model instance.
        :param if_node: If node.
        :return: Model with extra outputs before If node.
        """
        transformation_layout = TransformationLayout()
        for command in self._backend_entity.create_output_insertion_commands(model, if_node):
            transformation_layout.register(command)
        return model_transformer.transform(transformation_layout)

    def _dfs_quantize_models(
        self,
        parent_model: TModel,
        parent_graph: NNCFGraph,
        parent_dataset: Dataset,
        parent_statistic_points: Optional[StatisticPointsContainer],
        parent_model_cnt: int,
    ) -> Tuple[TModel, int]:
        """

        :param parent_model:
        :param parent_graph:
        :param parent_dataset:
        :param parent_statistic_points:
        :param parent_model_cnt:
        :return:
        """
        if self._has_if_op(parent_graph, self._backend_entity.if_node_metatype):
            model_transformer = factory.ModelTransformerFactory.create(parent_model)

            global_model_cnt = parent_model_cnt
            for if_node in parent_graph.get_nodes_by_metatypes([self._backend_entity.if_node_metatype]):
                for if_submodel_condition in (True, False):
                    model_cnt = global_model_cnt + 1
                    child_model = self._extract_if_submodel(model_transformer, if_node, if_submodel_condition)
                    child_model_input_names = self._backend_entity.get_if_subgraph_input_names(
                        parent_model, if_node, if_submodel_condition
                    )
                    if_cond_input_name = self._backend_entity.get_if_cond_input_name(parent_model, if_node)
                    parent_model_with_additional_outputs = self._add_outputs_before_if_node(
                        model_transformer, parent_model, if_node
                    )

                    child_dataset = self._make_dataset_for_child_model(
                        factory.EngineFactory.create(parent_model_with_additional_outputs),
                        parent_dataset,
                        if_cond_input_name,
                        child_model_input_names,
                        if_submodel_condition,
                        model_cnt,
                    )

                    child_quantized_model, model_cnt = self._dfs_quantize_models(
                        child_model, NNCFGraphFactory.create(child_model), child_dataset, None, model_cnt
                    )
                    global_model_cnt = model_cnt

                    nncf_logger.info(f"Set quantized model number {model_cnt} to the original model")
                    parent_model = self._update_if_submodel(
                        model_transformer, if_node, if_submodel_condition, child_quantized_model
                    )
                    if self.intermediate_model_dir:
                        nncf_logger.info(
                            f"Save quantized model number {model_cnt} to dir {self.intermediate_model_dir}"
                        )
                        self._backend_entity.dump_submodel(
                            child_quantized_model, self.intermediate_model_dir, if_node, if_submodel_condition
                        )

        nncf_logger.info(f"Quantize a model number {parent_model_cnt}")
        quantized_model = self._apply(parent_model, parent_graph, parent_statistic_points, parent_dataset)
        return quantized_model, parent_model_cnt
