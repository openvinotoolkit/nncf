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
from typing import List

import torch

from nncf.experimental.torch.sparsity.movement.layers import MovementSparsifier
from nncf.torch.compression_method_api import PTCompressionLoss


class ImportanceLoss(PTCompressionLoss):
    """
    Module to calculate the compression loss of movement sparsity.
    """

    def __init__(self, operands: List[MovementSparsifier]):
        """
        Initializes the loss of movement sparsity in its algorithm controller.

        :param operands: List of movement sparsity operands for each layer to sparsify.
        """
        super().__init__()
        assert len(operands) > 0, "No sparse layers to calculate importance loss."
        self.operands = operands
        self._disabled = False

    def disable(self):
        self._disabled = True

    def calculate(self) -> torch.Tensor:
        loss = torch.tensor(0.0, device=self._get_device())
        if not self._disabled:
            for n, operand in enumerate(self.operands):
                loss = loss * (n / (n + 1)) + operand.loss() / (n + 1)  # avoid overflow
        return loss

    def _get_device(self) -> torch.device:
        return next(self.operands[0].parameters()).device
