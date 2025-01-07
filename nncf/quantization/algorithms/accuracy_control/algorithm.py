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

import sys
from typing import Iterable, List, Optional, Tuple, TypeVar

import nncf
from nncf.common.factory import NNCFGraphFactory
from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.utils import get_number_of_quantized_ops
from nncf.common.logging import nncf_logger
from nncf.common.quantization.quantizer_removal import revert_operations_to_floating_point_precision
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.common.utils.os import get_available_cpu_count
from nncf.common.utils.os import get_available_memory_amount
from nncf.data.dataset import Dataset
from nncf.parameters import DropType
from nncf.quantization.advanced_parameters import RestoreMode
from nncf.quantization.algorithms.accuracy_control.backend import AccuracyControlAlgoBackend
from nncf.quantization.algorithms.accuracy_control.evaluator import Evaluator
from nncf.quantization.algorithms.accuracy_control.evaluator import MetricResults
from nncf.quantization.algorithms.accuracy_control.ranker import Ranker

TModel = TypeVar("TModel")
TTensor = TypeVar("TTensor")
PREPARATION_MODEL_THRESHOLD = 1
OVERHEAD_COEFFICIENT = 2
MEMORY_INCREASE_COEFFICIENT = 2


def get_algo_backend(backend: BackendType) -> AccuracyControlAlgoBackend:
    """
    Returns backend for accuracy control algorithm.

    :param backend: Backend.
    :return: The backend for accuracy control algorithm.
    """
    if backend == BackendType.OPENVINO:
        from nncf.quantization.algorithms.accuracy_control.openvino_backend import OVAccuracyControlAlgoBackend

        return OVAccuracyControlAlgoBackend()

    if backend == BackendType.ONNX:
        from nncf.quantization.algorithms.accuracy_control.onnx_backend import ONNXAccuracyControlAlgoBackend

        return ONNXAccuracyControlAlgoBackend()

    raise nncf.UnsupportedBackendError(
        f"Cannot create the backend for the accuracy control algorithm because {backend} is not supported."
    )


def _create_message(nodes: Iterable[NNCFNode]) -> str:
    names = [f"\t{x.node_name}" for x in nodes]
    return "\n".join(names)


def calculate_accuracy_drop(
    initial_metric: float, quantized_metric: float, max_drop: float, drop_type: DropType
) -> Tuple[bool, Optional[float]]:
    """
    Calculates accuracy drop and termination boolean flag.

    :param initial_metric: Metric value for initial model.
    :param quantized_metric: Metric value for quantized model.
    :param max_drop: Maximum accuracy drop that should be achieved.
    :param drop_type: Accuracy drop type.
    :return: A tuple (should_terminate, accuracy_drop) where:
        - should_terminate: Whether the algorithm should terminate or not.
        - accuracy_drop: Accuracy drop value.
    """
    should_terminate = None
    accuracy_drop = None

    if quantized_metric >= initial_metric:
        drop_values_by_drop_type = {
            DropType.RELATIVE: None,
            DropType.ABSOLUTE: initial_metric - quantized_metric,
        }
        accuracy_drop = drop_values_by_drop_type[drop_type]
        should_terminate = True
    else:
        drop_values_by_drop_type = {
            DropType.RELATIVE: abs(1 - quantized_metric / initial_metric),
            DropType.ABSOLUTE: initial_metric - quantized_metric,
        }
        accuracy_drop = drop_values_by_drop_type[drop_type]
        should_terminate = accuracy_drop <= max_drop

    return should_terminate, accuracy_drop


class QuantizationAccuracyRestorerReport:
    """
    Contains execution information about accuracy-aware algorithm.

    :param removed_groups: All groups of quantizers which were removed.
    :param removed_all: True if all quantizers were removed, False otherwise.
    :param reached_required_drop: True if the required accuracy drop was reached, False otherwise.
    :param num_quantized_operations: Number of quantized operations in the model.
    :param num_iterations: Number of iterations performed.
    """

    def __init__(self):
        self.removed_groups = []
        self.removed_all = False
        self.reached_required_drop = False
        self.num_quantized_operations = None
        self.num_iterations = None

    @property
    def removed_quantizers(self) -> List[NNCFNode]:
        """
        Returns all removed quantizers during accuracy-aware algorithm.
        """
        quantizers = []
        for group in self.removed_groups:
            quantizers.extend(group.quantizers)
        return quantizers

    @property
    def reverted_operations(self) -> List[NNCFNode]:
        """
        Returns all operations which were reverted to original precision
        during accuracy-aware algorithm.
        """
        operations = []
        for group in self.removed_groups:
            operations.extend(group.operations)
        return operations


class QuantizationAccuracyRestorer:
    """
    Implementation of the accuracy-aware loop.
    """

    def __init__(
        self,
        ranking_subset_size: int = 300,
        max_num_iterations: int = sys.maxsize,
        max_drop: float = 0.01,
        drop_type: DropType = DropType.ABSOLUTE,
        num_ranking_workers: Optional[int] = None,
        restore_mode: RestoreMode = RestoreMode.ACTIVATIONS_AND_WEIGHTS,
    ):
        """
        :param ranking_subset_size: The number of data items that will be selected from
            the dataset to rank groups of quantizers.
        :param max_num_iterations: A maximal number of iterations.
        :param max_drop: The maximum accuracy drop that should be achieved.
        :param drop_type: The accuracy drop type, which determines how the maximum
            accuracy drop between the original model and the compressed model is
            calculated.
        :param num_ranking_workers: The number of parallel workers that are used to rank
            quantization operations.
        :param restore_mode: Restore mode.
        """
        self.ranking_subset_size = ranking_subset_size
        self.max_num_iterations = max_num_iterations
        self.max_drop = max_drop
        self.drop_type = drop_type
        self.num_ranking_workers = num_ranking_workers
        self.restore_mode = restore_mode

    def apply(
        self,
        initial_model: TModel,
        initial_metric_results: MetricResults,
        quantized_model: TModel,
        quantized_metric_results: MetricResults,
        validation_dataset: Dataset,
        validation_dataset_size: int,
        evaluator: Evaluator,
    ) -> TModel:
        """
        Restores the accuracy of the quantized model by removing the groups of quantizers
        that contribute the most to the drop in accuracy.

        :param initial_model: Initial model (not quantized).
        :param initial_metric_results: Initial model metrics.
        :param quantized_model: Quantized model.
        :param quantized_metric_results: Quantized model metrics.
        :param validation_dataset: A dataset for the validation process.
        :param validation_dataset_size: Validation dataset size.
        :param evaluator: The instance of `Evaluator` to validate model and collect values
            for each item from dataset.
        :return: The quantized model whose metric `final_metric` is satisfied
            the maximum accuracy drop condition.
        """
        algo_backend = get_algo_backend(get_backend(initial_model))

        should_terminate, accuracy_drop = calculate_accuracy_drop(
            initial_metric_results.metric_value, quantized_metric_results.metric_value, self.max_drop, self.drop_type
        )

        if should_terminate:
            QuantizationAccuracyRestorer._print_completion_message(accuracy_drop, self.drop_type)
            return quantized_model

        nncf_logger.info(f"Accuracy drop: {accuracy_drop} ({self.drop_type})")

        # Accuracy drop is greater than the maximum drop so we need to restore accuracy
        return self._apply(
            initial_model,
            initial_metric_results,
            quantized_model,
            quantized_metric_results,
            validation_dataset,
            validation_dataset_size,
            evaluator,
            accuracy_drop,
            algo_backend,
        )

    def _apply(
        self,
        initial_model: TModel,
        initial_metric_results: MetricResults,
        quantized_model: TModel,
        quantized_metric_results: MetricResults,
        validation_dataset: Dataset,
        validation_dataset_size: int,
        evaluator: Evaluator,
        accuracy_drop: float,
        algo_backend: AccuracyControlAlgoBackend,
    ) -> TModel:
        """
        An internal function that implements an iterative approach to restoring the accuracy of
        the quantized model by removing the groups of quantizers that contribute the most to
        the drop in accuracy.

        :param initial_model: Initial model (not quantized).
        :param initial_metric_results: Initial model metrics.
        :param quantized_model: Quantized model.
        :param quantized_metric_results: Quantized model metrics.
        :param validation_dataset: A dataset for the validation process.
        :param validation_dataset_size: Validation dataset size.
        :param evaluator: The instance of `Evaluator` to validate model and collect values
            for each item from dataset.
        :param accuracy_drop: Accuracy drop between initial and quantized models.
        :param algo_backend: The `AccuracyControlAlgoBackend` algo backend.
        :return: The quantized model whose metric `final_metric` is satisfied
            the maximum accuracy drop condition.
        """
        initial_model_graph = NNCFGraphFactory.create(initial_model)
        quantized_model_graph = NNCFGraphFactory.create(quantized_model)

        # Collect original biases and weights because these values are
        # required to undo bias correction and weight correction.
        # Store this data inside the `node.attributes` dictionary.
        # This data will be used in the `revert_operations_to_floating_point_precision()` method.
        QuantizationAccuracyRestorer._collect_original_biases_and_weights(
            initial_model_graph, quantized_model_graph, initial_model, algo_backend
        )

        # Show the number of quantized operations in the model.
        report = QuantizationAccuracyRestorerReport()
        report.num_quantized_operations = get_number_of_quantized_ops(
            quantized_model_graph, algo_backend.get_quantizer_metatypes(), algo_backend.get_quantizable_metatypes()
        )
        nncf_logger.info(f"Total number of quantized operations in the model: {report.num_quantized_operations}")

        # Calculate number of parallel processes for Ranker
        num_ranking_workers = self.num_ranking_workers
        if num_ranking_workers is None:
            model_size = algo_backend.get_model_size(quantized_model)
            num_ranking_workers = self._calculate_number_ranker_workers(
                model_size,
                quantized_metric_results.preparation_time,
                quantized_metric_results.validation_time,
                validation_dataset_size,
            )

        nncf_logger.info(f"Number of parallel workers to rank quantized operations: {num_ranking_workers}")

        ranker = Ranker(self.ranking_subset_size, validation_dataset, algo_backend, evaluator, num_ranking_workers)
        groups_to_rank = ranker.find_groups_of_quantizers_to_rank(quantized_model_graph)
        ranked_groups = ranker.rank_groups_of_quantizers(
            groups_to_rank,
            quantized_model,
            quantized_model_graph,
            initial_metric_results.values_for_each_item,
            quantized_metric_results.values_for_each_item,
        )

        previous_model = quantized_model
        previous_approximate_values_for_each_item = quantized_metric_results.values_for_each_item
        previous_accuracy_drop = accuracy_drop
        current_model = None
        current_approximate_values_for_each_item = None
        current_accuracy_drop = None
        is_step_back = True

        nncf_logger.info("Changing the scope of quantizer nodes was started")
        for iteration in range(self.max_num_iterations):
            if current_model is not None:
                previous_model = current_model

            # greedy removal of the FQ node with the highest importance score
            current_group = ranked_groups.pop()
            current_model = revert_operations_to_floating_point_precision(
                current_group.operations,
                current_group.quantizers,
                previous_model,
                quantized_model_graph,
                self.restore_mode,
                algo_backend.get_op_with_weights_metatypes(),
                algo_backend.is_node_with_weight,
                algo_backend.get_weight_tensor_port_ids,
            )
            report.removed_groups.append(current_group)

            nncf_logger.debug(
                f"Removed a block of {len(current_group.quantizers)} quantizers:"
                f"\n{_create_message(current_group.quantizers)}"
            )
            nncf_logger.info(
                f"Reverted {len(current_group.operations)} operations to the floating-point "
                f"precision: \n{_create_message(current_group.operations)}"
            )

            # Calculate drop for new quantization scope.
            current_metric, current_approximate_values_for_each_item = evaluator.validate(
                current_model, validation_dataset
            )

            should_terminate, current_accuracy_drop = calculate_accuracy_drop(
                initial_metric_results.metric_value, current_metric, self.max_drop, self.drop_type
            )

            if not ranked_groups:
                nncf_logger.info(
                    "All layers have been checked and the AccuracyAwareQuantization "
                    "will not be able to achieve the required accuracy drop"
                )
                report.removed_all = True
                break

            # Accuracy was restored to the acceptable drop.
            if should_terminate:
                report.reached_required_drop = True
                QuantizationAccuracyRestorer._print_completion_message(current_accuracy_drop, self.drop_type)
                break

            nncf_logger.info(
                f"Accuracy drop with the new quantization scope is {float(current_accuracy_drop)} ({self.drop_type})"
            )

            # Continue greedy quantizer remove
            if current_accuracy_drop <= previous_accuracy_drop or (
                current_accuracy_drop > previous_accuracy_drop and is_step_back
            ):
                is_step_back = False
                previous_accuracy_drop = current_accuracy_drop
                continue

            if current_accuracy_drop > previous_accuracy_drop:
                current_model = previous_model
                current_approximate_values_for_each_item = previous_approximate_values_for_each_item
                report.removed_groups.pop()
                ranked_groups.append(current_group)
                is_step_back = True

            previous_accuracy_drop = current_accuracy_drop

            nncf_logger.info("Re-calculating ranking scores for remaining groups")
            if current_approximate_values_for_each_item is None:
                current_approximate_values_for_each_item = evaluator.collect_values_for_each_item(
                    current_model, validation_dataset
                )

            ranked_groups = ranker.rank_groups_of_quantizers(
                ranked_groups,
                current_model,
                quantized_model_graph,
                initial_metric_results.values_for_each_item,
                current_approximate_values_for_each_item,
            )

        report.num_iterations = iteration
        QuantizationAccuracyRestorer._print_report(report, self.max_num_iterations)

        return current_model

    def _calculate_number_ranker_workers(
        self,
        model_size: int,
        preparation_time: float,
        validation_time: float,
        validation_dataset_size: int,
    ) -> int:
        """
        Calculate the number of parallel ranker workers

        :param model_size: Target model size.
        :param preparation_time: The time it takes to prepare the model.
        :param validation_time: The time it takes to validate the model.
        :param validation_dataset_size: Validation dataset size.
        :return: The number of parallel ranker workers
        """
        if preparation_time < PREPARATION_MODEL_THRESHOLD:
            return 1

        # Calculate the number of parallel processes needed to override model preparation and
        # metric calculation on the ranking subset
        ranking_time = validation_time * self.ranking_subset_size / validation_dataset_size
        n_workers = max(round((preparation_time / ranking_time + 1) * OVERHEAD_COEFFICIENT), 2)

        # Apply limitation by number of CPU cores
        n_cores = get_available_cpu_count(logical=True)
        n_workers = max(min(n_workers, n_cores // 2), 1)

        # Apply limitation by memory
        ram = get_available_memory_amount()
        n_copies = ram // (model_size * MEMORY_INCREASE_COEFFICIENT)
        n_workers = max(min(n_workers, n_copies - 1), 1)

        return n_workers

    @staticmethod
    def _collect_original_biases_and_weights(
        initial_model_graph: NNCFGraph,
        quantized_model_graph: NNCFGraph,
        initial_model: TModel,
        algo_backend: AccuracyControlAlgoBackend,
    ) -> None:
        """
        Collects initial biases and weights and stores them inside the `node.attributes['original_bias']` and
        `node.attributes['original_weight']` where `node` is a node from `quantized_model_graph`.

        :param initial_model_graph: Graph for initial model.
        :param quantized_model_graph: Graph for quantized model.
        :param initial_model: Initial model.
        :param algo_backend: The `AccuracyControlAlgoBackend` algo backend.
        """
        for node in initial_model_graph.get_all_nodes():
            if algo_backend.is_node_with_bias(node, initial_model_graph):
                node_with_bias = quantized_model_graph.get_node_by_name(node.node_name)
                node_with_bias.attributes["original_bias"] = algo_backend.get_bias_value(
                    node, initial_model_graph, initial_model
                )
            if algo_backend.is_node_with_weight(node):
                node_with_weight = quantized_model_graph.get_node_by_name(node.node_name)
                for port_id in algo_backend.get_weight_tensor_port_ids(node_with_weight):
                    weight = algo_backend.get_weight_value(node, initial_model, port_id)
                    node_with_weight.attributes[f"original_weight.{port_id}"] = weight

    @staticmethod
    def _print_report(report: QuantizationAccuracyRestorerReport, max_num_iterations: int) -> None:
        """
        Shows report.

        :param report: Report.
        :param max_num_iterations: A maximal number of iterations.
        """
        if report.removed_all or not report.reached_required_drop:
            nncf_logger.info("The algorithm could not achieve the required accuracy drop.")

        if report.num_iterations + 1 >= max_num_iterations:
            nncf_logger.info("Maximum number of iteration was reached.")

        if not report.removed_all:
            nncf_logger.debug(f"Quantizers that were removed:\n{_create_message(report.removed_quantizers)}")
            nncf_logger.info(
                f"{len(report.reverted_operations)} out of {report.num_quantized_operations} "
                "were reverted back to the floating-point precision:"
                f"\n{_create_message(report.reverted_operations)}"
            )

    @staticmethod
    def _print_completion_message(accuracy_drop: float, drop_type: DropType) -> None:
        if accuracy_drop is None or accuracy_drop < 0:
            reason = "metric of the quantized model is greater than the metric of the initial model"
        else:
            reason = f"achieved required accuracy drop {float(accuracy_drop)} ({drop_type})"
        nncf_logger.info(f"Algorithm completed: {reason}")
