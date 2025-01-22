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
from torch import nn

from nncf.torch.dynamic_graph.context import TracingContext


class KnowledgeDistillationLossHandler(nn.Module):
    """
    Encapsulates knowledge distillation logic. Controls knowledge distillation loss calculation. Notice that knowledge
    distillation loss is computed between results of original model and compressed model inferences only with latest
    inputs. And storages loss values in context at storage_device for further access. Such complex method of storage
    is required for DataParallel model replication logic.
    """

    KD_LOSS_STORAGE_NAME = "kd_loss"
    KD_STORAGE_DEVICE = "kd_storage_device"

    def __init__(
        self, context: TracingContext, kd_original_model: nn.Module, calculate_kd_loss_fn, storage_device: torch.device
    ):
        super().__init__()
        self._compressed_context = context
        self._kd_original_model = kd_original_model
        self._calculate_kd_loss_fn = calculate_kd_loss_fn
        self._compressed_context.register_global_buffer(self.KD_LOSS_STORAGE_NAME, [])
        self._compressed_context.register_global_buffer(self.KD_STORAGE_DEVICE, storage_device)

    def zero_kd_loss(self):
        """
        Frees storage space for further next iteration loss value storage.
        """
        self._compressed_context.global_buffer_store[self.KD_LOSS_STORAGE_NAME] = []

    def get_kd_loss(self) -> List[torch.Tensor]:
        if len(self._compressed_context.global_buffer_store[self.KD_LOSS_STORAGE_NAME]) == 0:
            return [torch.zeros([], device=self._compressed_context.global_buffer_store[self.KD_STORAGE_DEVICE])]
        return self._compressed_context.global_buffer_store[self.KD_LOSS_STORAGE_NAME]

    def forward(self, compressed_model_outputs, *args, **kwargs):
        """
        Infers kd original model with latest NNCFNetwork forward inputs (*args, **kwargs) and computes distillation loss
        between results of kd original model forward and compressed model forward (inputs). Then stores loss values
        in context at storage device.

        :param compressed_model_outputs: Results of compressed model forward used for
        knowledge distillation loss calculations.
        """
        self.zero_kd_loss()
        with torch.no_grad():
            original_model_outputs = self._kd_original_model(*args, **kwargs)
        kd_loss = self._calculate_kd_loss_fn(compressed_model_outputs, original_model_outputs)
        if kd_loss is not None:
            self._compressed_context.global_buffer_store[self.KD_LOSS_STORAGE_NAME].append(
                kd_loss.to(self._compressed_context.global_buffer_store[self.KD_STORAGE_DEVICE])
            )
