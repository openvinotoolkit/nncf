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
from abc import ABC
from abc import abstractmethod
from itertools import islice
from typing import Any, Dict, List, Optional, TypeVar

import nncf
from nncf.common import factory
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.logging.logger import nncf_logger
from nncf.common.logging.track_progress import track
from nncf.common.tensor import NNCFTensor
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.data.dataset import Dataset

TensorType = TypeVar("TensorType")
TModel = TypeVar("TModel")

EMPTY_DATASET_MESSAGE = (
    "Calibration dataset must not be empty. Please provide calibration dataset with at least one sample."
)
BATCH_SIZE_IS_BIGGER_THAN_SUBSET_SIZE_MESSAGE = (
    "Provided dataset has a batch size value is bigger than subset size for statistics collection. "
    "Please increase the number of samples for a statistics collection "
    "or decrease the batch size value in the dataset."
)
BATCH_SIZE_MODEL_WARNING = (
    "For the particular model the batch size > 1 can lead to inaccurate collected statistics. "
    "The recomendation is to provide dataloader instance with the batch_size = 1."
)
DECREASING_SAMPLES_NUMBER_MESSAGE = (
    "The number of samples for statistics collection is decreased "
    "to align with the provided batch size value of the dataset."
)


class StatisticsAggregator(ABC):
    """
    Base class for statistics collection.
    """

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.stat_subset_size = None
        self.batch_size = self.dataset.get_batch_size() or 1
        dataset_len = self.dataset.get_length()
        self.dataset_sample_size = (
            dataset_len * self.batch_size if dataset_len is not None else dataset_len
        )  # Number of samples in the dataset
        if self.dataset_sample_size == 0:
            raise nncf.ValidationError(EMPTY_DATASET_MESSAGE)
        self.statistic_points = StatisticPointsContainer()

    def _get_number_samples_for_statistics(
        self,
    ) -> Optional[int]:
        """
        Returns number of samples for statistics collection.

        :return: Number of samples for statistics collection.
        """
        return (
            min(self.dataset_sample_size or self.stat_subset_size, self.stat_subset_size)
            if self.stat_subset_size is not None
            else None
        )

    def _get_iterations_num(self, total_statistics_samples: int) -> int:
        """
        Returns number of iterations to collect statistics.

        :param total_statistics_samples: Number of statistics samples are used.
        :return: Iterations number of statistics collection.
        """
        return total_statistics_samples // self.batch_size

    def collect_statistics(self, model: TModel, graph: NNCFGraph) -> None:
        """
        Collects statistics for registered StatisticPoints.
        The statistics are stored in self.statistic_points.

        :param model: Backend-specific model instance.
        :param graph: Model graph.
        """
        if not self.statistic_points:
            return
        if self.batch_size > 1 and self.is_model_has_no_batch_axis(graph):
            nncf_logger.warning(BATCH_SIZE_MODEL_WARNING)
        model_transformer = factory.ModelTransformerFactory.create(model)
        merged_statistics = self._get_merged_statistic_points(self.statistic_points, model, graph)
        transformation_layout = self._get_transformation_layout_extra_outputs(merged_statistics)
        model_with_outputs = model_transformer.transform(transformation_layout)
        engine = factory.EngineFactory.create(model_with_outputs)

        statistics_samples_num = self._get_number_samples_for_statistics()
        iterations_num = (
            self._get_iterations_num(statistics_samples_num) if statistics_samples_num is not None else None
        )
        if iterations_num is not None:
            if iterations_num == 0:
                raise nncf.ValidationError(BATCH_SIZE_IS_BIGGER_THAN_SUBSET_SIZE_MESSAGE)
            samples_num = iterations_num * self.batch_size
            if samples_num != statistics_samples_num:
                nncf_logger.warning(DECREASING_SAMPLES_NUMBER_MESSAGE)
                statistics_samples_num = samples_num
        empty_statistics = True
        with track(total=statistics_samples_num, description="Statistics collection") as pbar:
            for input_data in islice(self.dataset.get_inference_data(), iterations_num):
                outputs = engine.infer(input_data)
                processed_outputs = self._process_outputs(outputs)
                self._register_statistics(processed_outputs, merged_statistics)
                pbar.progress.update(pbar.task, advance=self.batch_size)
                empty_statistics = False
        if empty_statistics:
            raise nncf.ValidationError(EMPTY_DATASET_MESSAGE)

    def register_statistic_points(self, statistic_points: StatisticPointsContainer) -> None:
        """
        Register statistic points for statistics collection and recalculates the maximum number samples
        for collecting statistics, based on the maximum value from the all algorithms.

        :param statistic_points: StatisticPointsContainer instance with the statistic points
        """
        for _, _statistic_points in statistic_points.items():
            for _statistic_point in _statistic_points:
                self.statistic_points.add_statistic_point(_statistic_point)

        for _, _statistic_points in self.statistic_points.items():
            for _statistic_point in _statistic_points:
                for _, tensor_collectors in _statistic_point.algorithm_to_tensor_collectors.items():
                    for tensor_collector in tensor_collectors:
                        if self.stat_subset_size is None:
                            self.stat_subset_size = tensor_collector.num_samples
                        elif tensor_collector.num_samples is not None:
                            self.stat_subset_size = max(self.stat_subset_size, tensor_collector.num_samples)

    def is_model_has_no_batch_axis(self, graph: NNCFGraph) -> bool:
        """
        Returns True if NNCFGraph contains metatypes with no batch axis in output tensor.

        :param graph: NNCFGraph.
        :return: True if NNCFGraph contains metatypes with no batch axis in output tensor.
        """
        unique_graph_metatypes = set(node.metatype for node in graph.get_all_nodes())
        return any(metatype in self.metatypes_no_batch_support for metatype in unique_graph_metatypes)

    @property
    @abstractmethod
    def metatypes_no_batch_support(self) -> List[OperatorMetatype]:
        """
        These metatypes mix outputs for different samples into one axis.
        If reducers and aggregators collect statistics at the output of the following operations,
        assuming that 0-axis is batch axis, they get only 1 value instead of batch_size values.
        It could lead to inaccurate/incorrect statistics result.
        """

    @abstractmethod
    def _register_statistics(self, outputs: Dict[str, NNCFTensor], statistic_points: StatisticPointsContainer) -> None:
        """
        Process prepared raw model outputs and statistic points for the further usage.

        :param outputs: prepared raw model outputs
        :param statistic_points: StatisticPointsContainer instance with the statistic points
        """

    @abstractmethod
    def _get_transformation_layout_extra_outputs(
        self, statistic_points: StatisticPointsContainer
    ) -> TransformationLayout:
        """
        Creates backend-specific transformation layout for the further statistics collection.

        :param statistic_points: StatisticPointsContainer to add outputs
        :return: TransformationLayout with the corresponding transformations
        """

    @staticmethod
    @abstractmethod
    def _get_merged_statistic_points(
        statistic_points: StatisticPointsContainer, model: TModel, graph: NNCFGraph
    ) -> StatisticPointsContainer:
        """
        Creates a new StatisticPointContainer that has no duplicated tensor collectors for one
        unique statistic point. Alters statistic collectors in the given statistic point container so statistics
        collected by merged statistic collectors will be available in all corresponding statistic collectors
        from the given statistic point container.

        :param statistic_points: Registered statistic points with possible tensor collectors duplicates.
        :param model: Backend-specific target model.
        :param graph: Model graph.
        :return: Merged statistic points container bounded with given statistic point container.
        """

    @staticmethod
    @abstractmethod
    def _process_outputs(outputs: Any) -> Dict[str, NNCFTensor]:
        """
        Post-process model outputs for the further statistics collection.

        :param outputs: raw model outputs
        :return: processed model outputs in Dict[str, NNCFTensor] format
        """
