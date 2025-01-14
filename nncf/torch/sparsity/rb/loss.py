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

import torch

from nncf.torch.compression_method_api import PTCompressionLoss


# Actually in responsible to lean density to target value
class SparseLoss(PTCompressionLoss):
    def __init__(self, sparse_layers=None, target=1.0, p=0.05):
        super().__init__()
        self._sparse_layers = sparse_layers
        self.target = target
        self.p = p
        self.disabled = False
        self.current_sparsity: float = 0.0
        self.mean_sparse_prob = 0.0

    def set_layers(self, sparse_layers):
        self._sparse_layers = sparse_layers

    def disable(self):
        if not self.disabled:
            self.disabled = True

            for sparse_layer in self._sparse_layers:
                sparse_layer.frozen = True

    def calculate(self) -> torch.Tensor:
        if self.disabled:
            return 0

        params = 0
        loss = 0
        sparse_prob_sum = 0
        for sparse_layer in self._sparse_layers:
            if not self.disabled and sparse_layer.frozen:
                raise AssertionError(
                    "Invalid state of SparseLoss and SparsifiedWeight: mask is frozen for enabled loss"
                )
            if not sparse_layer.frozen:
                sw_loss = sparse_layer.loss()
                params = params + sw_loss.view(-1).size(0)
                loss = loss + sw_loss.sum()
                sparse_prob_sum += torch.sigmoid(sparse_layer.mask).sum()

        self.mean_sparse_prob = (sparse_prob_sum / params).item()
        self.current_sparsity = 1 - loss / params
        return ((loss / params - self.target) / self.p).pow(2)

    @property
    def target_sparsity_rate(self):
        rate = 1 - self.target
        if rate < 0 or rate > 1:
            raise IndexError("Target is not within range(0,1)")
        return rate

    def set_target_sparsity_loss(self, sparsity_level):
        self.target = 1 - sparsity_level


class SparseLossForPerLayerSparsity(SparseLoss):
    def __init__(self, sparse_layers=None, target=1.0, p=0.05):
        super().__init__(sparse_layers, target, p)
        self.per_layer_target = {}
        for sparse_layer in self._sparse_layers:
            self.per_layer_target[sparse_layer] = self.target

    def calculate(self) -> torch.Tensor:
        if self.disabled:
            return 0

        params = 0
        sparse_prob_sum = 0
        sparse_layers_loss = 0
        for sparse_layer in self._sparse_layers:
            if not self.disabled and not sparse_layer.sparsify:
                raise AssertionError(
                    "Invalid state of SparseLoss and SparsifiedWeight: mask is frozen for enabled loss"
                )
            if sparse_layer.sparsify:
                sw_loss = sparse_layer.loss()
                params_layer = sw_loss.view(-1).size(0)
                params += params_layer
                sparse_layers_loss -= torch.abs(sw_loss.sum() / params_layer - self.per_layer_target[sparse_layer])
                sparse_prob_sum += torch.sigmoid(sparse_layer.mask).sum()

        self.mean_sparse_prob = (sparse_prob_sum / params).item()
        return (sparse_layers_loss / self.p).pow(2)

    def set_target_sparsity_loss(self, target, sparse_layer):
        self.per_layer_target[sparse_layer] = 1 - target
