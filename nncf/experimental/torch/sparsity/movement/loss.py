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

import torch

from nncf.experimental.torch.sparsity.movement.layers import MovementSparsifier
from nncf.torch.compression_method_api import PTCompressionLoss


class ImportanceLoss(PTCompressionLoss):
    """
    Module to calculate the compression loss of movement sparsity.
    """

    def __init__(self, sparse_layers: List[MovementSparsifier]):
        """
        Initializes the loss of movement sparsity in its algorithm controller.

        :param sparse_layers: List of movement sparsity operands for each layer to sparsify.
        """
        super().__init__()
        self.sparse_layers = sparse_layers
        self._disabled = False
        self._device = next(sparse_layers[0].parameters()).device

    def disable(self):
        self._disabled = True

    def calculate(self):
        if self._disabled:
            return torch.zeros([], device=self._device)
        loss = 0.
        for n, sparse_layer in enumerate(self.sparse_layers):
            loss = loss * (n / (n + 1)) + sparse_layer.loss() / (n + 1)  # avoid overflow
        return loss
