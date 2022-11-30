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
from typing import List, Optional


from nncf.common.schedulers import BaseCompressionScheduler
from nncf.torch.compression_method_api import PTCompressionLoss


class ImportanceLoss(PTCompressionLoss):
    def __init__(self, sparse_layers: Optional[List] = None,
                 penalty_scheduler: Optional[BaseCompressionScheduler] = None):
        super().__init__()
        self.sparse_layers = sparse_layers
        self.penalty_scheduler = penalty_scheduler

    def calculate(self):
        if not self.sparse_layers:
            return 0.
        loss = self.sparse_layers[0].loss()
        for sparse_layer in self.sparse_layers[1:]:
            loss = loss + sparse_layer.loss()
        multiplier = 1.0
        if self.penalty_scheduler is not None:
            multiplier = self.penalty_scheduler.current_importance_lambda
        return loss / len(self.sparse_layers) * multiplier
