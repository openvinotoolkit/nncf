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
from abc import ABC
from abc import abstractmethod
from itertools import islice
from typing import Any, Dict, List, TypeVar

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


class StatisticsAggregator(ABC):
    """
    Base class for statistics collection.
    """

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.stat_subset_size = None
        self.batch_size = self.dataset.get_batch_size() or 1
        self.dataset_size = self.dataset.get_length()
        self.dataset_size = self.dataset_size * self.batch_size if self.dataset_size is not None else self.dataset_size
        self.statistic_points = StatisticPointsContainer()

    def collect_statistics(self, model: TModel, graph: NNCFGraph) -> None:
        """
        Collects statistics for registered StatisticPoints.
        The statistics are stored in self.statistic_points.

        :param model: Backend-specific model instance.
        :param graph: Model graph.
        """
        if self.batch_size > 1 and self.is_model_batch_size_limited_support(graph):
            nncf_logger.warning("The batch size > 1 for the specific model can lead to accuracy degradation")
        if not self.statistic_points:
            return
        model_transformer = factory.ModelTransformerFactory.create(model)
        merged_statistics = self._get_merged_statistic_points(self.statistic_points, model, graph)
        transformation_layout = self._get_transformation_layout_extra_outputs(merged_statistics)
        model_with_outputs = model_transformer.transform(transformation_layout)
        engine = factory.EngineFactory.create(model_with_outputs)

        calibration_samples_num = (
            min(self.dataset_size or self.stat_subset_size, self.stat_subset_size)
            if self.stat_subset_size is not None
            else None
        )  # Maybe subsample should be in terms of a tensor with arbitary batch_size
        collected_statistics_num = 0
        with track(total=calibration_samples_num, description="Statistics collection") as pbar:
            for input_data in islice(self.dataset.get_inference_data(), self.stat_subset_size):
                batch_size_to_collect = (
                    min(calibration_samples_num - collected_statistics_num, self.batch_size)
                    if calibration_samples_num is not None
                    else self.batch_size
                )
                outputs = engine.infer(input_data)
                processed_outputs = self._process_outputs(outputs)
                self._register_statistics(processed_outputs, merged_statistics)
                collected_statistics_num += batch_size_to_collect
                pbar.progress.update(pbar.task, advance=batch_size_to_collect)
                if calibration_samples_num and collected_statistics_num == calibration_samples_num:
                    break
        if collected_statistics_num == 0:
            raise RuntimeError(
                "Calibration dataset must not be empty. Please provide calibration dataset with at least one sample."
            )

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

    def is_model_batch_size_limited_support(self, graph: NNCFGraph) -> bool:
        """
        :param NNCFGraph graph: _description_
        :return bool: _description_
        """
        for metatype in self.metatypes_output_has_no_batch_axis:
            if metatype in set(node.metatype for node in graph.get_all_nodes()):
                return True
        return False

    @property
    @abstractmethod
    def metatypes_output_has_no_batch_axis(self) -> List[OperatorMetatype]:
        """ """

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
