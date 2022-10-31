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

T_model = TypeVar('T_model')


# pylint: disable=no-member
class StaticModelTransformerBase(ModelTransformer, ABC):

    def __init__(self, model: T_model):
        super().__init__(model)
        self._transformation_layout = None
        self._transformations_list = []

    @abstractmethod
    def transform(self, transformation_layout: TransformationLayout) -> T_model:
        """
        Applies transformation layout on the model

        :param transformation_layout: TransformationLayout
        :return: model after transformations
        """

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
        Applies bias correction transformations on the model

        :param transformations: lisf of the TransformationCommand transformations
        """
        raise NotImplementedError(
            '_apply_bias_correction_transformations method must be implemented before call')

    @staticmethod
    def _apply_model_extraction_transformation(transformations: List[TransformationCommand]) -> T_model:
        """
        Extracts or builds sub-model from the original based on the inputs and outputs names

        :param transformations: list of the TransformationCommand transformations
        """
        raise NotImplementedError(
            '_apply_model_extraction_transformation must be implemented before call')
