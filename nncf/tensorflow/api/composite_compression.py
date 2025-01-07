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

import nncf
from nncf import NNCFConfig
from nncf.common.composite_compression import CompositeCompressionAlgorithmBuilder
from nncf.common.composite_compression import CompositeCompressionAlgorithmController
from nncf.config.extractors import extract_algorithm_names
from nncf.tensorflow.algorithm_selector import get_compression_algorithm_builder
from nncf.tensorflow.api.compression import TFCompressionAlgorithmBuilder
from nncf.tensorflow.graph.transformations.layout import TFTransformationLayout

TModel = TypeVar("TModel")


class TFCompositeCompressionAlgorithmBuilder(CompositeCompressionAlgorithmBuilder, TFCompressionAlgorithmBuilder):
    def __init__(self, config: NNCFConfig, should_init: bool = True):
        super().__init__(config, should_init)

        algo_names = extract_algorithm_names(config)
        if len(algo_names) < 2:
            raise nncf.ValidationError(
                "Composite algorithm builder must be supplied with a config with more than one "
                "compression algo specified!"
            )
        for algo_name in algo_names:
            algo_builder_cls = get_compression_algorithm_builder(algo_name)
            self._child_builders.append(algo_builder_cls(config, should_init=should_init))

    def _build_controller(self, model: TModel) -> CompositeCompressionAlgorithmController:
        composite_ctrl = CompositeCompressionAlgorithmController(model)
        for builder in self.child_builders:
            composite_ctrl.add(builder.build_controller(model))
        return composite_ctrl

    def get_transformation_layout(self, model: TModel) -> TFTransformationLayout:
        transformations = TFTransformationLayout()
        for builder in self.child_builders:
            transformations.update(builder.get_transformation_layout(model))
        return transformations

    def initialize(self, model: TModel) -> None:
        for builder in self.child_builders:
            builder.initialize(model)
