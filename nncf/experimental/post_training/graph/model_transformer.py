"""
 Copyright (c) 2022 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from abc import ABC, abstractmethod
from typing import List
from typing import TypeVar

from nncf.common.graph.model_transformer import ModelTransformer
from nncf.common.graph.transformations.commands import TransformationCommand
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.experimental.post_training.statistics.statistic_point import StatisticPointsContainer

ModelType = TypeVar('ModelType')


# pylint: disable=no-member
class StaticModelTransformerBase(ModelTransformer, ABC):

    def __init__(self, model: ModelType):
        super().__init__(model)
        self._transformation_layout = None
        self._transformations_list = []

    def set_model(self, model: ModelType) -> None:
        """
        Sets model

        :param model: input model
        """
        self._model = model

    def prepare_model_for_statistics_collection(self, statistic_points: StatisticPointsContainer) -> ModelType:
        """
        Prepares model for statics collection by adding external outputs

        :param statistic_points: StatisticPointsContainer
        :return: model after transformations
        """
        transformation_layout = self._get_transformation_layout_extra_outputs(statistic_points)
        return self.transform(transformation_layout)

    @abstractmethod
    def transform(self, transformation_layout: TransformationLayout) -> ModelType:
        """
        Applies transformation layout on the model

        :param transformation_layout: TransformationLayout
        :return: model after transformations
        """

    def _get_transformation_layout_extra_outputs(
            self,
            statistic_points: StatisticPointsContainer) -> TransformationLayout:
        """
        Collects transformations layout by statistic_points

        :param statistic_points: StatisticPointsContainer
        :return: transformation_layout
        """
        raise NotImplementedError(
            '_get_transformation_layout_extra_outputs method must be implemented before call')

    def _apply_quantizer_insertion_transformations(self, transformations: List[TransformationCommand]) -> None:
        """
        Applies quantizer insertion transformations to the model

        :param transformations: lisf of the TransformationCommand transformations
        """
        raise NotImplementedError(
            '_apply_quantizer_insertion_transformations method must be implemented before call')

    def _apply_output_insertion_transformations(self, transformations: List[TransformationCommand]) -> None:
        """
        Applies output insertion transformations to the model

        :param transformations: list of the TransformationCommand transformations
        """
        raise NotImplementedError(
            '_apply_output_insertion_transformations method must be implemented before call')

    def _apply_bias_correction_transformations(self, transformations: List[TransformationCommand]) -> None:
        """
        Applies incoming transformations to the model

        :param transformations: list of the TransformationCommand transformations
        """
        raise NotImplementedError(
            '_apply_bias_correction_transformations method must be implemented before call')

    @staticmethod
    def extract_model_by_inputs_outputs(model: ModelType, inputs: List[str], outputs: List[str]):
        """
        Extracts or builds sub-model from the original based on the inputs and outputs names
        """
        raise NotImplementedError('extract_model_by_inputs_outputs must be implemented before call')
