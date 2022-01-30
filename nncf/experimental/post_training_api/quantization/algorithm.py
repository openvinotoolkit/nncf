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
from nncf.experimental.post_training_api.engine import Engine
from nncf.experimental.post_training_api.dataloader import DataLoader
from nncf.experimental.post_training_api.graph.model_transformer import ModelTransformer
from nncf.experimental.post_training_api.quantization.initialization.initializer import Initializer
from nncf.experimental.post_training_api.compressed_model import CompressedModel


class PostTrainingQuantization(BaseCompressionAlgorithmBuilder):
    """
    This class has to
    """

    def __init__(self, quantization_config: Dict[str, object], engine: Engine, dataloader: DataLoader):
        super().__init__(None)  # TODO: what should do with NNCFConfig?
        self.quantization_config = quantization_config
        self.initializer = Initializer(engine, dataloader, quantization_config.get('initilization'))
        self.priority = None  # Priority of algorithms application sei by CompressionBuilder

    def apply_to(self, compressed_model: CompressedModel) -> CompressedModel:
        transformation_layout = self.get_transformation_layout(compressed_model)
        transformed_compressed_model = ModelTransformer.transform(compressed_model, transformation_layout)

        initialized_compressed_model = self.initialize(transformed_compressed_model)

        return initialized_compressed_model

    def initialize(self, compressed_model: CompressedModel) -> CompressedModel:
        while self.initializer.is_empty():
            algorithm = self.initializer.pop()
            compressed_model = self.initializer.run_algorithm(algorithm, compressed_model)
        return compressed_model

    def __lt__(self, other):
        return self.priority < other.priority

    def __gt__(self, other):
        return self.priority > other.priority
