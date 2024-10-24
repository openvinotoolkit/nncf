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
from ultralytics.models.yolo import YOLO

from tests.torch.fx.performance_check.model_builders.base import BaseModelBuilder


class UltralyticsModelBuilder(BaseModelBuilder):
    INPUT_SHAPE = (1, 3, 224, 224)

    def __init__(self, model_id: str):
        self._model_id = model_id

    def build(self):
        pt_model = YOLO(self._model_id).model
        return torch.export.export(pt_model, self.get_example_inputs(), strict=False).module()

    def get_example_inputs(self) -> torch.Tensor:
        return (torch.ones(self.INPUT_SHAPE),)

    def get_input_sizes(self):
        return self.INPUT_SHAPE
