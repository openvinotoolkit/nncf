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

import operator
import sys
from typing import Any, Callable, Iterable, List, Optional, Tuple, TypeVar

from nncf.common.factory import NNCFGraphFactory
from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.utils import get_number_of_quantized_ops
from nncf.common.logging import nncf_logger
from nncf.common.quantization.quantizer_removal import revert_operations_to_floating_point_precision
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.data.dataset import Dataset
from nncf.parameters import DropType
from nncf.quantization.algorithms.accuracy_control.backend import AccuracyControlAlgoBackend
from nncf.quantization.algorithms.accuracy_control.rank_functions import normalized_mse
from nncf.quantization.algorithms.accuracy_control.ranker import LogitsBasedRanker
from nncf.quantization.algorithms.accuracy_control.ranker import MetricBasedRanker
from nncf.quantization.algorithms.accuracy_control.ranker import Ranker

TModel = TypeVar("TModel")


def get_algo_backend(backend: BackendType) -> AccuracyControlAlgoBackend:
    """
    Returns backend for accuracy control algorithm.

    :param backend: Backend.
    :return: The backend for accuracy control algorithm.
    """
    if backend == BackendType.OPENVINO:
        from nncf.quantization.algorithms.accuracy_control.openvino_backend import OVAccuracyControlAlgoBackend

        return OVAccuracyControlAlgoBackend()

    raise RuntimeError(
        "Cannot create the backend for the accuracy control algorithm " f"because {backend} is not supported."
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
    ):
        """
        :param ranking_subset_size: The number of data items that will be selected from
            the dataset to rank groups of quantizers.
        :param max_num_iterations: A maximal number of iterations.
        :param max_drop: The maximum accuracy drop that should be achieved.
        :param drop_type: The accuracy drop type, which determines how the maximum
            accuracy drop between the original model and the compressed model is
            calculated.
        """
        self.ranking_subset_size = ranking_subset_size
        self.max_num_iterations = max_num_iterations
        self.max_drop = max_drop
        self.drop_type = drop_type

    def restore_accuracy(
        self,
        initial_model: TModel,
        initial_metric: float,
        quantized_model: TModel,
        quantized_metric: float,
        validation_dataset: Dataset,
        validation_fn: Callable[[Any, Iterable[Any]], float],
    ) -> TModel:
        """
        Restores the accuracy of the quantized model by removing the groups of quantizers
        that contribute the most to the drop in accuracy.

        :param initial_model: Initial model (not quantized).
        :param initial_metric: Metric value for initial model.
        :param quantized_model: Quantized model.
        :param quantized_metric: Metric value for quantized model.
        :param validation_dataset: A dataset for the validation process.
        :param validation_fn: A validation function to validate the model. It should take
            two argumets:
            - `model`: model to be validate.
            - `validation_dataset`: dataset that provides data items to
                validate the provided model.
            The function should return the value of the metric with the following meaning:
            A higher value corresponds to better performance of the model.
        :return: The quantized model whose metric `final_metric` is satisfied
            the maximum accuracy drop condition.
        """
        backend = get_backend(initial_model)
        algo_backend = get_algo_backend(backend)

        should_terminate, accuracy_drop = calculate_accuracy_drop(
            initial_metric, quantized_metric, self.max_drop, self.drop_type
        )

        if should_terminate:
            QuantizationAccuracyRestorer._print_completion_message(accuracy_drop, self.drop_type)
            return quantized_model

        nncf_logger.info(f"Accuracy drop: {accuracy_drop} ({self.drop_type})")

        if accuracy_drop <= self.max_drop:
            return quantized_model

        initial_model_graph = NNCFGraphFactory.create(initial_model)
        quantized_model_graph = NNCFGraphFactory.create(quantized_model)

        # Collect original biases and weights because these values are
        # required to undo bias correction and weight correction.
        # Store this data inside the `node.data` dictionary.
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

        nncf_logger.info("Ranking groups of quantizers was started")
        ranker = QuantizationAccuracyRestorer._create_ranker(
            initial_model, validation_fn, validation_dataset, self.ranking_subset_size, algo_backend
        )
        groups_to_rank = ranker.find_groups_of_quantizers_to_rank(quantized_model_graph)
        ranked_groups = ranker.rank_groups_of_quantizers(
            groups_to_rank, initial_model, quantized_model, quantized_model_graph
        )

        previous_model = quantized_model
        previous_accuracy_drop = accuracy_drop
        current_model = None
        current_accuracy_drop = None
        is_step_back = True

        nncf_logger.info("Changing the scope of quantizer nodes was started")
        for iteration in range(self.max_num_iterations):
            if current_model is not None:
                previous_model = current_model

            # greedy removal of the FQ node with the highest importance score
            current_group = ranked_groups.pop()
            current_model = revert_operations_to_floating_point_precision(
                current_group.operations, current_group.quantizers, previous_model, quantized_model_graph
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
            current_metric = validation_fn(
                algo_backend.prepare_for_inference(current_model), validation_dataset.get_data()
            )

            should_terminate, current_accuracy_drop = calculate_accuracy_drop(
                initial_metric, current_metric, self.max_drop, self.drop_type
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
                report.removed_groups.pop()
                ranked_groups.append(current_group)
                is_step_back = True

            previous_accuracy_drop = current_accuracy_drop

            nncf_logger.info("Re-calculating ranking scores for remaining groups")
            ranked_groups = ranker.rank_groups_of_quantizers(
                ranked_groups, initial_model, current_model, quantized_model_graph
            )

        report.num_iterations = iteration
        QuantizationAccuracyRestorer._print_report(report, self.max_num_iterations)

        return current_model

    @staticmethod
    def _collect_original_biases_and_weights(
        initial_model_graph: NNCFGraph,
        quantized_model_graph: NNCFGraph,
        initial_model: TModel,
        algo_backend: AccuracyControlAlgoBackend,
    ) -> None:
        """
        Collects initial biases and weights and stores them inside the `node.data['original_bias']` and
        `node.data['original_weight']` where `node` is a node from `quantized_model_graph`.

        :param initial_model_graph: Graph for initial model.
        :param quantized_model_graph: Graph for quantized model.
        :param initial_model: Initial model.
        :param algo_backend: The `AccuracyControlAlgoBackend` algo backend.
        """
        for node in initial_model_graph.get_all_nodes():
            if algo_backend.is_node_with_bias(node, initial_model_graph):
                node_with_bias = quantized_model_graph.get_node_by_name(node.node_name)
                node_with_bias.data["original_bias"] = algo_backend.get_bias_value(
                    node, initial_model_graph, initial_model
                )
            if algo_backend.is_node_with_weight(node):
                node_with_weight = quantized_model_graph.get_node_by_name(node.node_name)
                for port_id in algo_backend.get_weight_tensor_port_ids(node_with_weight):
                    weight = algo_backend.get_weight_value(node, initial_model, port_id)
                    node_with_weight.data[f"original_weight.{port_id}"] = weight

    @staticmethod
    def _create_ranker(
        initial_model: TModel,
        validation_fn: Callable[[Any, Iterable[Any]], float],
        validation_dataset: Dataset,
        ranking_subset_size: int,
        algo_backend: AccuracyControlAlgoBackend,
    ) -> Ranker:
        """
        Creates an instance of the `Ranker` class.

        :param initial_model: Initial model.
        :param validation_fn: A validation function to validate the model.
        :param validation_dataset: A dataset for the validation process.
        :param ranking_subset_size: The number of data items that will be selected from
            the dataset to rank groups of quantizers.
        :param algo_backend: The `AccuracyControlAlgoBackend` algo backend.
        :return: An instance of the `Ranker` class.
        """
        # Check whether it is possible to calculate the metric for one data item.
        # pylint: disable=W0703
        try:
            _ = validation_fn(algo_backend.prepare_for_inference(initial_model), validation_dataset.get_data([0]))
            ranker = MetricBasedRanker(
                ranking_subset_size, operator.sub, validation_dataset, algo_backend, validation_fn
            )
        except Exception:
            ranker = LogitsBasedRanker(ranking_subset_size, normalized_mse, validation_dataset, algo_backend)
        nncf_logger.info(
            f'The {"original" if isinstance(ranker, MetricBasedRanker) else "NMSE"} '
            "metric will be used to rank quantizers"
        )
        return ranker

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
