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
from typing import List

from nncf.experimental.torch.sparsity.movement.layers import MovementSparsifier
from nncf.torch.compression_method_api import PTCompressionLoss


class ImportanceLoss(PTCompressionLoss):
    def __init__(self, sparse_layers: List[MovementSparsifier]):
        super().__init__()
        self.sparse_layers = sparse_layers
        self._disabled = False

    def disable(self):
        self._disabled = True

    def calculate(self):
        if not self.sparse_layers or self._disabled:
            return 0.
        loss = self.sparse_layers[0].loss()
        n = 1
        for sparse_layer in self.sparse_layers[1:]:
            loss = loss * (n / (n + 1)) + sparse_layer.loss() / (n + 1)  # avoid overflow
            n += 1
        return loss
