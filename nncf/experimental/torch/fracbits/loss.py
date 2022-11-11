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

from numbers import Number
from typing import Dict, Union
import torch
from nncf.common.utils.registry import Registry
from nncf.experimental.torch.fracbits.params import FracBitsLossParams
from nncf.torch.compression_method_api import PTCompressionLoss
from nncf.torch.module_operations import UpdateWeight
from nncf.torch.nncf_network import NNCFNetwork
from torch import nn
from dataclasses import dataclass
from nncf.torch.quantization.layers import BaseQuantizer
from nncf.common.utils.logger import logger as nncf_logger


FRACBITS_LOSSES = Registry("fracbits_loss")
EPS = 1e-6


@dataclass
class ModuleQuantizerPair:
    module: nn.Module
    quantizer: BaseQuantizer


@FRACBITS_LOSSES.register("model_size")
class ModelSizeCompressionLoss(PTCompressionLoss):
    def __init__(self, model: NNCFNetwork, params: FracBitsLossParams):
        super().__init__()
        self._model = model
        self._compression_rate = torch.FloatTensor([params.compression_rate])
        self._criteria = self._get_criteria(params.criteria)

        self._w_q_pairs: Dict[str, ModuleQuantizerPair] = {}

        for name, module in self._model.named_modules():
            if isinstance(module, UpdateWeight):
                parent_name = ".".join(name.split(".")[:-2])
                parent_module = self._model.get_submodule(parent_name)

                self._w_q_pairs[parent_name] = ModuleQuantizerPair(parent_module, module.op)

        with torch.no_grad():
            self._init_model_size = self._get_model_size()

        self._flip_loss = params.flip_loss
        self._alpha = params.alpha

    def calculate(self) -> torch.Tensor:
        if self._flip_loss:
            cur_comp_rate = self._get_frac_model_size() / self._init_model_size
            tgt_comp_rate = 1 / self._compression_rate.to(device=cur_comp_rate.device)
        else:
            cur_comp_rate = self._init_model_size / (self._get_frac_model_size() + EPS)
            tgt_comp_rate = self._compression_rate.to(device=cur_comp_rate.device)

        return self._alpha * self._criteria(cur_comp_rate, tgt_comp_rate)

    def _get_criteria(self, criteria) -> nn.modules.loss._Loss:
        if criteria == "L1":
            return nn.L1Loss()
        if criteria == "L2":
            return nn.MSELoss()
        raise RuntimeError(f"Unknown criteria = {criteria}.")

    @staticmethod
    def _get_module_size(module: nn.Module, num_bits: Union[int, torch.Tensor]) -> Union[torch.Tensor, Number]:
        if isinstance(module, (nn.modules.conv._ConvNd, nn.Linear)):  # pylint: disable=protected-access
            return module.weight.shape.numel() * num_bits
        nncf_logger.warning("module={module} is not supported by ModelSizeCompressionLoss. Skip it.")
        return 0.

    def _get_frac_model_size(self) -> torch.Tensor:
        return sum([self._get_module_size(pair.module, pair.quantizer.frac_num_bits)
                    for pair in self._w_q_pairs.values()])

    def _get_model_size(self) -> Number:
        return sum([self._get_module_size(pair.module, pair.quantizer.num_bits) for pair in self._w_q_pairs.values()])

    @torch.no_grad()
    def get_state(self) -> Dict[str, Number]:
        curr_model_size = self._get_model_size()
        frac_model_size = self._get_frac_model_size()

        states = {
            "current_model_size": curr_model_size,
            "fractional_model_size": self._init_model_size / (frac_model_size + EPS),
            "compression_rate": self._init_model_size / (curr_model_size + EPS)
        }

        for name, pair in self._w_q_pairs.items():
            states[f"frac_bits/{name}"] = pair.quantizer.frac_num_bits.item()

        return states


@FRACBITS_LOSSES.register("bitops")
class BitOpsCompressionLoss(PTCompressionLoss):
    def calculate(self) -> torch.Tensor:
        raise NotImplementedError()

    @torch.no_grad()
    def get_state(self) -> Dict[str, Number]:
        raise NotImplementedError()
