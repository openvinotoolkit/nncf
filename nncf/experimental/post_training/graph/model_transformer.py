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

from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.experimental.onnx.compressed_model import CompressedModel


class ModelTransformer(ABC):

    def transform(self, model: CompressedModel, transformation_layout: TransformationLayout) -> CompressedModel:
        pass
