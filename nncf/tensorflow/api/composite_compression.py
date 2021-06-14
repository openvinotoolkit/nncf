"""
 Copyright (c) 2020 Intel Corporation
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

from typing import List, Optional, TypeVar

from nncf import NNCFConfig
from nncf.common.composite_compression import CompositeCompressionAlgorithmBuilder
from nncf.common.composite_compression import CompositeCompressionAlgorithmController
from nncf.tensorflow.api.compression import TFCompressionAlgorithmBuilder
from nncf.tensorflow.api.compression import TFCompressionAlgorithmController
from nncf.tensorflow.graph.transformations.layout import TFTransformationLayout

ModelType = TypeVar('ModelType')
DatasetType = TypeVar('DatasetType')
LossType = TypeVar('LossType')


class TFCompositeCompressionAlgorithmController(
    CompositeCompressionAlgorithmController, TFCompressionAlgorithmController):
    def __init__(self, target_model: ModelType):
        super().__init__(target_model)
        self._initializer = None

    def initialize(self,
                   dataset: Optional[DatasetType] = None,
                   loss: Optional[LossType] = None) -> None:
        for ctrl in self.child_ctrls:
            ctrl.initialize(dataset, loss)


class TFCompositeCompressionAlgorithmBuilder(
    CompositeCompressionAlgorithmBuilder, TFCompressionAlgorithmBuilder):
    def __init__(self, config: Optional[NNCFConfig] = None, should_init: bool = True):
        super().__init__(config, should_init)

    @property
    def child_builders(self) -> List[TFCompressionAlgorithmBuilder]:
        return self._child_builders

    def add(self, child_builder: TFCompressionAlgorithmBuilder) -> None:
        self._child_builders.append(child_builder)

    def build_controller(self, model: ModelType) -> TFCompositeCompressionAlgorithmController:
        composite_ctrl = TFCompositeCompressionAlgorithmController(model)
        for builder in self.child_builders:
            composite_ctrl.add(builder.build_controller(model))
        return composite_ctrl

    def get_transformation_layout(self, model: ModelType) -> TFTransformationLayout:
        transformations = TFTransformationLayout()
        for builder in self.child_builders:
            transformations.update(builder.get_transformation_layout(model))
        return transformations
