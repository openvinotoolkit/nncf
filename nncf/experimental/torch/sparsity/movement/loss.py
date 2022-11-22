"""
 Copyright (c) 2019-2020 Intel Corporation
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

import torch
from nncf.torch.compression_method_api import PTCompressionLoss


class ImportanceLoss(PTCompressionLoss):
    def __init__(self, sparse_layers=None, penalty_scheduler=None):
        super().__init__()
        self._sparse_layers = sparse_layers
        self.penalty_scheduler = penalty_scheduler

    def calculate(self) -> torch.Tensor:
        if not self._sparse_layers:
            return 0.
        loss = self._sparse_layers[0].loss()
        for sparse_layer in self._sparse_layers[1:]:
            loss = loss + sparse_layer.loss()
        multiplier = 1.0
        if self.penalty_scheduler is not None:
            multiplier = self.penalty_scheduler.current_importance_lambda
        return loss / len(self._sparse_layers) * multiplier
