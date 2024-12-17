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

from typing import Dict, List, Optional, TypeVar, Union

from nncf.common.factory import StatisticsAggregatorFactory
from nncf.common.logging import nncf_logger
from nncf.common.model import ModelWrapper
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.data.dataset import Dataset
from nncf.quantization.algorithms.algorithm import Algorithm

TModel = TypeVar("TModel")
PipelineStep = List[Algorithm]


def collect_statistics(
    containers: Union[StatisticPointsContainer, List[StatisticPointsContainer]],
    model_wrapper: ModelWrapper,
    dataset: Dataset,
) -> StatisticPointsContainer:
    """
    Utility method for collecting statistics by model.

    :param statistic_points: Statistic points that need to be collected.
    :param model_wrapper: A wrapper object containing the model
    :param graph: A graph associated with a model.
    :param dataset: A dataset.
    :return: Collected statistics.
    """
    if not isinstance(containers, list):
        containers = [containers]

    statistics_aggregator = StatisticsAggregatorFactory.create(model_wrapper.model, dataset)
    for container in containers:
        statistics_aggregator.register_statistic_points(container)
    statistics_aggregator.collect_statistics(model_wrapper.model, model_wrapper.graph)

    return statistics_aggregator.statistic_points


class Pipeline:
    """
    A class for creating pipelines that apply algorithms to a model.

    This class is used for creating custom model processing pipelines
    that encapsulate a series of algorithms to be applied to a model
    using a provided dataset.

    A pipeline consists of pipeline steps. Each pipeline step is a
    sequence of Algorithm class instances whose statistic points are
    combined and collected using the model obtained after the previous
    pipeline step. The collected statistic points are used for all
    algorithms in this step.
    """

    def __init__(self, pipeline_steps: List[PipelineStep]):
        """
        :param pipeline_steps: A sequence of pipeline steps to be executed in order.
        """
        self._pipeline_steps = pipeline_steps

    @property
    def pipeline_steps(self) -> List[PipelineStep]:
        """
        Property that defines the sequence of distinct pipeline steps to
        be executed in order.

        :return: A sequence of pipeline steps to be executed in order.
        """
        return self._pipeline_steps

    def run(self, model: TModel, dataset: Dataset) -> TModel:
        """
        Executes the pipeline on the provided model.

        :param model: A model to which pipeline will be applied.
        :param dataset: A dataset that holds the data items for algorithms.
        :return: The updated model after executing the entire pipeline.
        """
        return self.run_from_step(model, dataset)

    def run_step(
        self,
        step_index: int,
        step_statistics: StatisticPointsContainer,
        model_wrapper: ModelWrapper,
    ) -> TModel:
        """
        Executes a provided pipeline step on the provided model.

        :param step_index: Zero-based index of the pipeline step that should be executed
        :param step_statistics: Statistics required to execute a pipeline step.
        :param model: A model to which a pipeline step will be applied.
        :param graph: A graph associated with a model.
        :return: The updated model after executing the pipeline step.
        """
        current_model = model_wrapper

        pipeline_steps = self._remove_unsupported_algorithms(model_wrapper.backend)
        pipeline_step = pipeline_steps[step_index]
        for algorithm in pipeline_step:
            current_model = algorithm.apply(current_model, statistic_points=step_statistics)
        return current_model

    def run_from_step(
        self,
        model_wrapper: ModelWrapper,
        dataset: Dataset,
        start_step_index: int = 0,
        step_index_to_statistics: Optional[Dict[int, StatisticPointsContainer]] = None,
    ) -> ModelWrapper:
        """
        Executes the pipeline from the specified pipeline step to the end.

        :param model: This is the model after the (start_step_index - 1)-th pipeline
            step, or the initial model if start_step_index is 0.
        :param dataset: A dataset that holds the data items for pipeline steps.
        :param graph: A graph associated with a model.
        :param start_step_index: Zero-based pipeline step index from which the pipeline
            should be executed.
        :param step_index_to_statistics: A mapping from pipeline step index to statistics
            required to execute pipeline step.
        :return: The updated model after executing the pipeline from the specified pipeline
            step to the end.
        """
        pipeline_steps = self._remove_unsupported_algorithms(model_wrapper.backend)
        if step_index_to_statistics is None:
            step_index_to_statistics = {}

        # The `step_model` and `step_graph` entities are required to execute `step_index`-th pipeline step
        step_model_wrapper = model_wrapper
        for step_index in range(start_step_index, len(pipeline_steps)):
            # Collect statistics required to run current pipeline step
            step_statistics = step_index_to_statistics.get(step_index)
            if step_statistics is None:
                statistic_points = self.get_statistic_points_for_step(step_index, step_model_wrapper)
                step_statistics = collect_statistics(statistic_points, step_model_wrapper, dataset)

            # Run current pipeline step
            step_model_wrapper = self.run_step(step_index, step_statistics, step_model_wrapper)

        return step_model_wrapper

    def get_statistic_points_for_step(self, step_index: int, model_wrapper: ModelWrapper) -> StatisticPointsContainer:
        """
        Returns statistics that should be collected to execute `step_index`-th pipeline step.

        :param step_index: Zero-based index of the pipeline step.
        :param model: A model.
        :param graph: A graph associated with a model.
        :return: Statistics that should be collected to execute `step_index`-th pipeline step.
        """
        container = StatisticPointsContainer()
        pipeline_steps = self._remove_unsupported_algorithms(get_backend(model_wrapper.model))
        pipeline_step = pipeline_steps[step_index]
        for algorithm in pipeline_step:
            for statistic_points in algorithm.get_statistic_points(model_wrapper).values():
                for statistic_point in statistic_points:
                    container.add_statistic_point(statistic_point)

        return container

    def _remove_unsupported_algorithms(self, backend: BackendType) -> List[PipelineStep]:
        pipeline_steps = []
        for pipeline_step in self._pipeline_steps:
            step = []
            for algorithm in pipeline_step:
                if backend not in algorithm.available_backends:
                    nncf_logger.debug(f"{backend.name} does not support {algorithm.__class__.__name__} algorithm yet.")
                    continue
                step.append(algorithm)

            if step:
                pipeline_steps.append(step)

        return pipeline_steps
