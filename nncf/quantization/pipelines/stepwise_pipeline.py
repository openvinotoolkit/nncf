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

from typing import Dict, List, Optional, TypeVar

from nncf.common.factory import NNCFGraphFactory
from nncf.common.factory import StatisticsAggregatorFactory
from nncf.common.graph.graph import NNCFGraph
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.data.dataset import Dataset
from nncf.quantization.algorithms.algorithm import Algorithm
from nncf.quantization.pipelines.pipeline import Pipeline

TModel = TypeVar("TModel")
PipelineStep = List[Algorithm]


def get_statistic_points(pipeline_step: PipelineStep, model: TModel, graph: NNCFGraph) -> StatisticPointsContainer:
    """
    TODO

    :param pipeline_step:
    :param model:
    :param graph:
    :return:
    """
    container = StatisticPointsContainer()
    for algorithm in pipeline_step:
        for statistic_points in algorithm.get_statistic_points(model, graph).values():
            for statistic_point in statistic_points:
                container.add_statistic_point(statistic_point)

    return container


def collect_statistics(
    statistic_points: StatisticPointsContainer, model: TModel, graph: NNCFGraph, dataset: Dataset
) -> StatisticPointsContainer:
    """
    TODO:

    :param statistic_points:
    :param model:
    :param graph:
    :param dataset:
    :return:
    """
    statistics_aggregator = StatisticsAggregatorFactory.create(model, dataset)
    statistics_aggregator.register_statistic_points(statistic_points)
    statistics_aggregator.collect_statistics(model, graph)

    return statistics_aggregator.statistic_points


class StepwisePipeline(Pipeline):
    """
    A class for creating sequential model processing pipelines with distinct steps.

    This class extends the base `Pipeline` class to provide access to each distinct
    step of the pipeline. Each pipeline step is a sequence of `Algorithm` class
    instances whose statistic points are combained and collected using the model
    that was obtained after previous pipeline step. Collected statistic points are
    used for all algorothms in this step.
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
        TODO:

        :param model: A model to which pipeline will be applied.
        :param dataset: A dataset that holds the data items for algorithms.
        :return: The updated model after executing the entire pipeline.
        """
        return run_pipeline_from_step(self, model, dataset)


def run_pipeline_step(
    pipeline_step: PipelineStep,
    pipeline_step_statistics: StatisticPointsContainer,
    model: TModel,
    graph: NNCFGraph,
) -> TModel:
    """
    Executes a provided pipeline step on the provided model.

    :param pipeline_step: A sequence of algorithms representing a pipeline step.
    :param pipeline_step_statistics: Statistics required to execute a pipeline step.
    :param model: A model to which a pipeline step will be applied.
    :param graph: A graph assosiated with a model.
    :return: The updated model after executing the pipeline step.
    """
    current_model = model
    current_graph = graph

    for algorithm in pipeline_step[:-1]:
        current_model = algorithm.apply(current_model, current_graph, pipeline_step_statistics)
        current_graph = NNCFGraphFactory.create(current_model)
    current_model = pipeline_step[-1].apply(current_model, current_graph, pipeline_step_statistics)

    return current_model


def run_pipeline_from_step(
    pipeline: StepwisePipeline,
    model: TModel,
    dataset: Dataset,
    graph: Optional[NNCFGraph],
    start_step_index: int = 0,
    step_index_to_statistics: Optional[Dict[int, StatisticPointsContainer]] = None,
) -> TModel:
    """
    Execute the pipeline from the specified pipeline step to the end.

    :param pipeline: A pipeline part of which should be executed.
    :param model: This is the model after the (start_step_index - 1)-th pipeline
        step, or the initial model if start_step_index is 0.
    :param dataset: A dataset that holds the data items for pipeline steps.
    :param graph: A graph assosiated with a model.
    :param start_step_index: Zero-based pipeline step index from which the pipeline
        should be executed.
    :param step_index_to_statistics: A mapping from pipeline step index to statistics
        required to execute pipeline step.
    :return: The updated model after executing the pipeline from the specified pipeline
        step to the end.
    """
    if step_index_to_statistics is None:
        step_index_to_statistics = {}

    # The `step_model` and `step_graph` entities are required to execute `step_index`-th pipeline step
    step_model = model
    step_graph = graph
    step_index = start_step_index

    for pipeline_step in pipeline.pipeline_steps[start_step_index:]:
        # Create graph required to run current pipeline step
        if step_graph is None:
            step_graph = NNCFGraphFactory.create(step_model)

        # Collect statistics required to run current pipeline step
        step_statistics = step_index_to_statistics.get(step_index)
        if step_statistics is None:
            statistic_points = get_statistic_points(pipeline_step, step_model, step_graph)
            step_statistics = collect_statistics(statistic_points, step_model, step_graph, dataset)

        # Run current pipeline step
        step_model = run_pipeline_step(pipeline_step, step_statistics, step_model, step_graph)

        step_graph = None  # We should rebuild the graph for the next pipeline step
        step_index += 1

    return step_model
