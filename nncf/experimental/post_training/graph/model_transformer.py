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

from abc import ABC
from abc import abstractmethod
from typing import List
from typing import TypeVar

from nncf.common.graph.model_transformer import ModelTransformer
from nncf.common.graph.transformations.commands import TransformationCommand
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.experimental.post_training.statistics.statistic_point import StatisticPointsContainer

ModelType = TypeVar('ModelType')


# pylint: disable=no-member
class StaticModelTransformerBase(ModelTransformer, ABC):
    QUANTIZER_NAME_PREFIX = 'QuantizeLinear_'
    DEQUANTIZER_NAME_PREFIX = 'DequantizeLinear_'
    SCALE_TENSOR_NAME_PREFIX = 'scale_'
    ZERO_POINT_NAME_PREFIX = 'zero_point_'

    def __init__(self, model: ModelType):
        super().__init__(model)
        self._transformation_layout = None
        self._transformations_list = []

        # Transformation commands list
        self._output_insertion_command = None
        self._quantizer_insertion_command = None

        self._callbacks_by_commands = {
            self._output_insertion_command: self._apply_output_insertion_transformations,
            self._quantizer_insertion_command: self._apply_quantizer_insertion_transformations,
        }

    def set_model(self, model: ModelType):
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

    def transform(self, transformation_layout: TransformationLayout) -> ModelType:
        """
        Applies transformation layout on the model

        :param transformation_layout: TransformationLayout
        :return: model after transformations
        """
        commands = list(self._callbacks_by_commands.keys())
        transformations_list = []
        for transformation in transformation_layout.transformations:
            if type(transformation) in commands:
                transformations_list.append(transformation)
        self._apply_transformations(transformations_list)
        return self._model

    def _get_transformation_layout_extra_outputs(self,
                                                 statistic_points: StatisticPointsContainer) -> TransformationLayout:
        """
        Collects transformations layout by statistic_points

        :param statistic_points: StatisticPointsContainer
        :return: transformation_layout
        """
        transformation_layout = self._transformation_layout()
        transformation_commands = []
        for _statistic_points in statistic_points.values():
            for _statistic_point in _statistic_points:
                transformation_commands.append(
                    self._output_insertion_command(_statistic_point.target_point))

        for transformation_command in transformation_commands:
            transformation_layout.register(transformation_command)

        return transformation_layout

    def _apply_transformations(self, transformations: List[TransformationCommand]):
        """
        Applies transformations by type-callback on the model

        :param transformations: lisf of the TransformationCommand transformations
        """
        transformations_by_types = {c: [] for c in self._callbacks_by_commands}
        for transformation in transformations:
            transformation_type = type(transformation)
            transformations_by_types[transformation_type].append(transformation)
        for transform_type, callback in self._callbacks_by_commands.items():
            if transformations_by_types[transform_type]:
                callback(transformations_by_types[transform_type])

    @abstractmethod
    def _apply_quantizer_insertion_transformations(self, transformations: List[TransformationCommand]):
        """
        Applies transformations on the model

        :param transformations: lisf of the TransformationCommand transformations
        """

    @abstractmethod
    def _apply_output_insertion_transformations(self, transformations: List[TransformationCommand]):
        """
        Applies incoming transformations to the model

        :param transformations: list of the TransformationCommand transformations
        """
