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
from itertools import islice
from typing import Callable, Dict, Iterable, List, Optional, Tuple, TypeVar

from tqdm import tqdm

from nncf import Dataset
from nncf.common import factory
from nncf.common.deprecation import warning_deprecated
from nncf.common.factory import NNCFGraphFactory
from nncf.common.factory import StatisticsAggregatorFactory
from nncf.common.graph.graph import NNCFGraph
from nncf.common.logging import nncf_logger
from nncf.common.quantization.structs import QuantizationPreset
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import copy_model
from nncf.common.utils.backend import get_backend
from nncf.data.dataset import DataItem
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
from nncf.quantization.passes import insert_null_biases_pass
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

    @dataclass
    class FirstStageAlgorithm:
        algorithm: "Algorithm"
        pre_passes: List[TPass]

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
        self.first_stage_algorithms: List[self.FirstStageAlgorithm] = []
        self.subset_size = subset_size

        if target_device is TargetDevice.VPU:
            warning_deprecated("VPU device is deprecated and will no longer be supported in the future.")

        if advanced_parameters is None:
            advanced_parameters = AdvancedQuantizationParameters()
        self.intermediate_model_dir = advanced_parameters.intermediate_model_dir

        if model_type == ModelType.TRANSFORMER:
            smooth_quant_algorithm = SmoothQuant(
                subset_size=subset_size,
                inplace_statistics=advanced_parameters.inplace_statistics,
                alpha=advanced_parameters.smooth_quant_alpha,
            )
            self.first_stage_algorithms.append(self.FirstStageAlgorithm(smooth_quant_algorithm, []))

        if not advanced_parameters.disable_channel_alignment:
            channel_alignment = ChannelAlignment(
                subset_size=subset_size,
                inplace_statistics=advanced_parameters.inplace_statistics,
                backend_params=advanced_parameters.backend_params,
            )
            self.first_stage_algorithms.append(self.FirstStageAlgorithm(channel_alignment, [insert_null_biases_pass]))

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

    def _is_single_model(self, model: TModel) -> bool:
        model_backend = get_backend(model)
        if model_backend == BackendType.ONNX:
            return True
        elif model_backend == BackendType.OPENVINO:
            from nncf.quantization.algorithms.post_training.openvino_backend import OVPostTrainingBackend

            return OVPostTrainingBackend.is_single_model(model)
        elif model_backend == BackendType.TORCH:
            return True
        else:
            raise RuntimeError(
                "Cannot return backend-specific entity because {} is not supported!".format(model_backend)
            )

    def _set_backend_entity(self, model: TModel) -> None:
        """
        Creates a helper class with a backed-specific logic of the algorithm

        :param model: backend-specific input model
        """
        model_backend = get_backend(model)
        if model_backend == BackendType.ONNX:
            raise RuntimeError(
                "Cannot return backend-specific entity because {} is not supported!".format(model_backend)
            )
        elif model_backend == BackendType.OPENVINO:
            from nncf.quantization.algorithms.post_training.openvino_backend import OVPostTrainingBackend

            self._backend_entity = OVPostTrainingBackend()
        elif model_backend == BackendType.TORCH:
            raise RuntimeError(
                "Cannot return backend-specific entity because {} is not supported!".format(model_backend)
            )
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
        for first_stage_algorithm in self.first_stage_algorithms:
            algorithm = first_stage_algorithm.algorithm
            backend = get_backend(model)
            if isinstance(algorithm, SmoothQuant) and backend != BackendType.OPENVINO:
                nncf_logger.debug(f"{backend.name} does not support SmoothQuant algorithm yet.")
                continue

            if isinstance(algorithm, ChannelAlignment) and backend != BackendType.OPENVINO:
                nncf_logger.debug(f"{backend.name} does not support ChannelAlignment algorithm yet.")
                continue

            for pre_pass in first_stage_algorithm.pre_passes:
                model = pre_pass(model, graph)
                graph = NNCFGraphFactory.create(model)

            statistics_aggregator = StatisticsAggregatorFactory.create(model, dataset)
            algo_statistic_points = algorithm.get_statistic_points(model, graph)
            statistics_aggregator.register_statistic_points(algo_statistic_points)
            statistics_aggregator.collect_statistics(model, graph)
            model = algorithm.apply(model, graph, statistics_aggregator.statistic_points)
            model = NNCFGraphFactory.create(model)

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
        if self._is_single_model(model_copy):
            return self._apply(model_copy, graph, statistic_points, dataset)
        self._set_backend_entity(model)
        nncf_logger.info("The model consists of child submodels. The iteratively each submodel will be quantized.")
        quantized_model, _ = self._dfs_quantize_models(model_copy, graph, dataset, statistic_points, 0)
        return quantized_model

    def collect_dataitems_for_children_models(
        self, engine, result_names, calibration_dataset, subset_size, model_cnt
    ) -> Iterable[DataItem]:
        """
        Returns dataitems for children models of the main model.

        :param model: Model to infer to collect dataitems.
        :param calibration_dataset: Dataset is used to collect new dataitems.
        :param subset_size: Size of dataitems to collect
        :param model_cnt: Global model number.
        """
        dataset = []
        for input_data in tqdm(
            islice(calibration_dataset.get_inference_data(), subset_size),
            total=subset_size,
            desc=f"Collect dataset for children models of {model_cnt} model:",
        ):
            results = engine.infer(input_data)
            data_item = []
            for result_name in result_names:
                data_item.append(results[result_name])
            dataset.append(tuple(data_item))
        return dataset

    def _dfs_quantize_models(
        self,
        parent_model: TModel,
        parent_graph: NNCFGraph,
        parent_dataset: Dataset,
        parent_statistic_points: Optional[StatisticPointsContainer],
        parent_model_cnt: int,
    ) -> Tuple[TModel, int]:
        if not self._backend_entity.is_single_model(parent_model):
            parent_model_with_additional_outputs, result_names = self._backend_entity.add_additional_outputs(
                parent_model
            )
            engine = factory.EngineFactory.create(parent_model_with_additional_outputs)
            dataitems = self.collect_dataitems_for_children_models(
                engine, result_names, parent_dataset, self.subset_size, parent_model_cnt
            )
            global_model_cnt = parent_model_cnt
            for child_model, backend_params in self._backend_entity.get_child_models(parent_model):
                child_dataset = self._backend_entity.make_dataset_for_child_models(dataitems, **backend_params)

                child_q_model, model_cnt = self._dfs_quantize_models(
                    child_model, NNCFGraphFactory.create(child_model), child_dataset, None, global_model_cnt + 1
                )
                global_model_cnt = model_cnt

                nncf_logger.info(f"Set quantized model number {model_cnt} to the original model")
                self._backend_entity.set_child_model(child_q_model, **backend_params)
                if self.intermediate_model_dir:
                    nncf_logger.info(f"Save quantized model number {model_cnt} to dir {self.intermediate_model_dir}")
                    self._backend_entity.dump_model(child_q_model, self.intermediate_model_dir, **backend_params)

        nncf_logger.info(f"Quantize a model number {parent_model_cnt}")
        quantized_model = self._apply(parent_model, parent_graph, parent_statistic_points, parent_dataset)
        return quantized_model, parent_model_cnt
