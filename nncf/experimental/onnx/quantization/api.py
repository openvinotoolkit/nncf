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
from typing import Dict

from nncf.common.compression import BaseCompressionAlgorithmBuilder
from nncf.experimental.onnx.api.engine import OnnxEngine
from nncf.experimental.onnx.graph.model_transformer import ONNXModelTransformer
from nncf.experimental.post_training_api.compressed_model import CompressedModel


class PostTrainingQuantizationBuilder(BaseCompressionAlgorithmBuilder):

    def __init__(self, quantization_config: Dict[str, object], engine: OnnxEngine):
        self.quantization_config = quantization_config
        self.engine = engine
        self.priority = 11  # TODO: change to algorithms priority enum

    def apply_to(self, model: CompressedModel) -> CompressedModel:
        transformation_layout = self.get_transformation_layout(model)
        transformer = ONNXModelTransformer(model)
        transformed_model = transformer.transform(transformation_layout)

        if self.should_init:
            self.initialize(transformed_model)

        return transformed_model

    def initialize(self, model: CompressedModel):
        ...

    def __lt__(self, other):
        return self.priority < other.priority

    def __gt__(self, other):
        return self.priority > other.priority
