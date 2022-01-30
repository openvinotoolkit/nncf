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
from typing import List

from nncf.experimental.onnx.engine import OnnxEngine
from nncf.experimental.post_training_api.quantization.algorithm import PostTrainingQuantization
from nncf.experimental.onnx.compressed_model import ONNXCompressedModel
from nncf.experimental.onnx.graph.transformations.layout import ONNXTransformationLayout


class ONNXPostTrainingQuantization(PostTrainingQuantization):

    def __init__(self, quantization_config: Dict[str, object], engine: OnnxEngine, dataloader):
        super().__init__(quantization_config, engine, dataloader)

    def get_transformation_layout(self, model: ONNXCompressedModel) -> List[ONNXTransformationLayout]:
        pass
        # 1. Get quantizers points
        # 2. Get ONNXTransformations from quantizers points
