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

from typing import TypeVar

from nncf.common.graph.transformations.layout import TransformationLayout

TModel = TypeVar("TModel")


class ModelTransformer:
    """
    Applies transformations to the model.
    """

    def __init__(self, model: TModel):
        """
        Initializes Model Transformer.

        :param model: The model to be transformed.
        """
        self._model = model

    def transform(self, transformation_layout: TransformationLayout) -> TModel:  # type:ignore
        """
        Applies transformations to the model.

        :param transformation_layout: An instance of `TransformationLayout` that
            includes a list of transformations to be applied to the model.
        :return: The transformed model.
        """
        raise NotImplementedError()
