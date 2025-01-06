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
from typing import Dict

import torch
from torch import Tensor

from nncf.common.quantization.structs import QuantizerConfig
from nncf.torch.dynamic_graph.context import no_nncf_trace


class PerturbationObserver:
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.perturbation = None
        self.numels = None

    def calc_perturbation(self, module, inputs: Tensor, output: Tensor):
        input_ = inputs[0] if isinstance(inputs, tuple) else inputs
        with no_nncf_trace():
            self.perturbation = torch.norm(input_ - output, p=2) ** 2
            self.numels = input_.size().numel()
            self.input_norm = torch.norm(input_, p=2) ** 2

    def reset(self):
        self.perturbation = None
        self.numels = None

    def get_observation(self):
        return self.perturbation

    def get_numels(self):
        return self.numels

    def get_input_norm(self):
        return self.input_norm


class Perturbations:
    def __init__(self):
        self._perturbations: Dict[int, Dict[QuantizerConfig, Tensor]] = {}

    def add(self, layer_id: int, qconfig: QuantizerConfig, perturbation: Tensor):
        if layer_id in self._perturbations:
            self._perturbations[layer_id].update({qconfig: perturbation})
        else:
            self._perturbations[layer_id] = {qconfig: perturbation}

    def get(self, layer_id: int, qconfig: QuantizerConfig) -> Tensor:
        layer_perturbations = self._perturbations[layer_id]
        return layer_perturbations[qconfig]

    def get_all(self) -> Dict[int, Dict[QuantizerConfig, Tensor]]:
        return self._perturbations
