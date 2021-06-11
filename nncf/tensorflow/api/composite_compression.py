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

from typing import TypeVar

from nncf import NNCFConfig
from nncf.common.composite_compression import CompositeCompressionAlgorithmBuilder
from nncf.common.composite_compression import CompositeCompressionAlgorithmController
from nncf.config.extractors import extract_compression_algorithm_configs
from nncf.tensorflow.algorithm_selector import get_compression_algorithm_builder
from nncf.tensorflow.api.compression import TFCompressionAlgorithmBuilder
from nncf.tensorflow.api.compression import TFCompressionAlgorithmController
from nncf.tensorflow.graph.transformations.layout import TFTransformationLayout

ModelType = TypeVar('ModelType')

class TFCompositeCompressionAlgorithmController(
    CompositeCompressionAlgorithmController, TFCompressionAlgorithmController):
    pass


class TFCompositeCompressionAlgorithmBuilder(
    CompositeCompressionAlgorithmBuilder, TFCompressionAlgorithmBuilder):
    def __init__(self, config: NNCFConfig, should_init: bool = True):
        super().__init__(config, should_init)

        algorithm_configs = extract_compression_algorithm_configs(config)
        for algo_config in algorithm_configs:
            self._child_builders.append(
                get_compression_algorithm_builder(algo_config)(algo_config, should_init=should_init))

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

    def initialize(self, model: ModelType) -> None:
        for builder in self.child_builders:
            builder.initialize(model)
