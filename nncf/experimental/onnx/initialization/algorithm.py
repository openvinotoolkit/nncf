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

from nncf.experimental.post_training.compressed_model import CompressedModel
from nncf.experimental.post_training.initialization.algorithm import InitializationAlgorithm


class ONNXInitializationAlgorithm(InitializationAlgorithm):
    def __init__(self, engine, dataloader):
        super().__init__(engine, dataloader)

    def run(self, model: CompressedModel):
        pass

        # for i, (_input, ...) in self.dataloader:
        #     self.engine.infer_model(_input)

    def _add_outputs_to_activations(self, model: CompressedModel):
        pass
