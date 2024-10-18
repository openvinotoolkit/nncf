# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torchvision import models

from tests.torch.fx.performance_check.model_builders.base import BaseModelBuilder


class TorchvisionModelBuilder(BaseModelBuilder):
    INPUT_SHAPE = (1, 3, 224, 224)

    def __init__(self, model_cls: str, model_weights: models.WeightsEnum):
        self._model_cls = model_cls
        self._model_weights = model_weights
        self._example_input = self._model_weights.transforms()(torch.ones(self.INPUT_SHAPE))

    def build(self):
        return self._model_cls(weights=self._model_weights).eval()

    def get_example_inputs(self) -> torch.Tensor:
        return (self._example_input,)

    def get_input_sizes(self):
        return tuple(self._example_input.shape)
