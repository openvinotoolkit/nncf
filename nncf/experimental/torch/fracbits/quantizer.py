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

# Reference: Yang, Linjie, and Qing Jin. "Fracbits: Mixed precision quantization via fractional bit-widths."
# Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 35. No. 12. 2021.

from typing import Dict
import torch

from nncf.torch.layer_utils import COMPRESSION_MODULES, CompressionParameter
from nncf.torch.quantization.layers import (
    QUANTIZATION_MODULES, AsymmetricQuantizer, PTQuantizerSpec, SymmetricQuantizer)
from nncf.torch.quantization.quantize_functions import asymmetric_quantize, symmetric_quantize
from nncf.torch.utils import no_jit_trace
from nncf.common.quantization.structs import QuantizationMode


class FracBitsQuantizationMode(QuantizationMode):
    SYMMETRIC = 'fracbits_symmetric'
    ASYMMETRIC = 'fracbits_asymmetric'


@COMPRESSION_MODULES.register()
@QUANTIZATION_MODULES.register(FracBitsQuantizationMode.SYMMETRIC)
class FracBitsSymmetricQuantizer(SymmetricQuantizer):
    def __init__(self, qspec: PTQuantizerSpec):
        super().__init__(qspec)
        self._min_num_bits = int(0.5 * qspec.num_bits)
        self._max_num_bits = int(1.5 * qspec.num_bits)
        self._num_bits = CompressionParameter(torch.FloatTensor([qspec.num_bits]), requires_grad=True,
                                              compression_lr_multiplier=qspec.compression_lr_multiplier)

    @property
    def frac_num_bits(self):
        return torch.clamp(self._num_bits, self._min_num_bits, self._max_num_bits)

    @property
    def num_bits(self):
        if self.is_num_bits_frozen:
            return super().num_bits

        with no_jit_trace():
            return self.frac_num_bits.round().int().item()

    @num_bits.setter
    def num_bits(self, num_bits: int):
        if num_bits < self._min_num_bits or num_bits > self._max_num_bits:
            raise RuntimeError(
                f"{num_bits} should be in [{self._min_num_bits}, {self._max_num_bits}]")
        self._num_bits.fill_(num_bits)

    @property
    def is_num_bits_frozen(self) -> bool:
        return not self._num_bits.requires_grad

    def set_min_max_num_bits(self, min_num_bits: int, max_num_bits: int):
        if min_num_bits >= max_num_bits:
            raise ValueError(
                f"min_num_bits({min_num_bits}) >= max_num_bits({max_num_bits})")
        self._min_num_bits = min_num_bits
        self._max_num_bits = max_num_bits

    def unfreeze_num_bits(self) -> None:
        self._num_bits.requires_grad_(True)

    def freeze_num_bits(self) -> None:
        self._num_bits.requires_grad_(False)
        super().set_level_ranges()

    def enable_gradients(self):
        super().enable_gradients()
        self.unfreeze_num_bits()

    def disable_gradients(self):
        super().disable_gradients()
        self.freeze_num_bits()

    def _quantize_with_n_bits(self, x, num_bits, execute_traced_op_as_identity: bool = False):
        scaled_num_bits = 1 if self._half_range else 0

        level_low, level_high, levels = self.calculate_level_ranges(
            num_bits - scaled_num_bits, self.signed)

        return symmetric_quantize(x, levels, level_low, level_high, self.scale, self.eps,
                                  skip=execute_traced_op_as_identity)

    def quantize(self, x, execute_traced_op_as_identity: bool = False):
        if self.is_num_bits_frozen:
            return super().quantize(x, execute_traced_op_as_identity)

        fl_num_bits = self.frac_num_bits.floor().int().item()
        ce_num_bits = fl_num_bits + 1

        fl_q = self._quantize_with_n_bits(
            x, fl_num_bits, execute_traced_op_as_identity)
        ce_q = self._quantize_with_n_bits(
            x, ce_num_bits, execute_traced_op_as_identity)

        return (self.frac_num_bits - fl_num_bits) * ce_q + (ce_num_bits - self.frac_num_bits) * fl_q

    def get_trainable_params(self) -> Dict[str, torch.Tensor]:
        return {self.SCALE_PARAM_NAME: self.scale.detach(), "num_bits": self.frac_num_bits.detach()}

    def _prepare_export_quantization(self, x: torch.Tensor):
        self.freeze_num_bits()
        return super()._prepare_export_quantization(x)

    @torch.no_grad()
    def get_input_range(self):
        self.set_level_ranges()
        input_low, input_high = self._get_input_low_input_high(
            self.scale, self.level_low, self.level_high, self.eps)
        return input_low, input_high


@COMPRESSION_MODULES.register()
@QUANTIZATION_MODULES.register(FracBitsQuantizationMode.ASYMMETRIC)
class FracBitsAsymmetricQuantizer(AsymmetricQuantizer):
    def __init__(self, qspec: PTQuantizerSpec):
        super().__init__(qspec)
        self._min_num_bits = int(0.5 * qspec.num_bits)
        self._max_num_bits = int(1.5 * qspec.num_bits)
        self._num_bits = CompressionParameter(torch.FloatTensor([qspec.num_bits]), requires_grad=True,
                                              compression_lr_multiplier=qspec.compression_lr_multiplier)

    @property
    def frac_num_bits(self):
        return torch.clamp(self._num_bits, self._min_num_bits, self._max_num_bits)

    @property
    def num_bits(self) -> int:
        if self.is_num_bits_frozen:
            return super().num_bits

        with no_jit_trace():
            return self.frac_num_bits.round().int().item()

    @num_bits.setter
    def num_bits(self, num_bits: int):
        if num_bits < self._min_num_bits or num_bits > self._max_num_bits:
            raise RuntimeError(
                f"{num_bits} should be in [{self._min_num_bits}, {self._max_num_bits}]")
        self._num_bits.fill_(num_bits)

    @property
    def is_num_bits_frozen(self) -> bool:
        return not self._num_bits.requires_grad

    def set_min_max_num_bits(self, min_num_bits: int, max_num_bits: int):
        if min_num_bits >= max_num_bits:
            raise ValueError(
                f"min_num_bits({min_num_bits}) >= max_num_bits({max_num_bits})")
        self._min_num_bits = min_num_bits
        self._max_num_bits = max_num_bits

    def unfreeze_num_bits(self) -> None:
        self._num_bits.requires_grad_(True)

    def freeze_num_bits(self) -> None:
        self._num_bits.requires_grad_(False)
        super().set_level_ranges()

    def enable_gradients(self):
        super().enable_gradients()
        self.unfreeze_num_bits()

    def disable_gradients(self):
        super().disable_gradients()
        self.freeze_num_bits()

    def _quantize_with_n_bits(self, x, num_bits, execute_traced_op_as_identity: bool = False):
        scaled_num_bits = 1 if self._half_range else 0

        level_low, level_high, levels = self.calculate_level_ranges(
            num_bits - scaled_num_bits)

        return asymmetric_quantize(x, levels, level_low, level_high, self.input_low, self.input_range, self.eps,
                                   skip=execute_traced_op_as_identity)

    def quantize(self, x, execute_traced_op_as_identity: bool = False):
        if self.is_num_bits_frozen:
            return super().quantize(x, execute_traced_op_as_identity)

        fl_num_bits = self.frac_num_bits.floor().int().item()
        ce_num_bits = fl_num_bits + 1

        fl_q = self._quantize_with_n_bits(
            x, fl_num_bits, execute_traced_op_as_identity)
        ce_q = self._quantize_with_n_bits(
            x, ce_num_bits, execute_traced_op_as_identity)

        return (self.frac_num_bits - fl_num_bits) * ce_q + (ce_num_bits - self.frac_num_bits) * fl_q

    def get_trainable_params(self) -> Dict[str, torch.Tensor]:
        return {self.INPUT_LOW_PARAM_NAME: self.input_low.detach(),
                self.INPUT_RANGE_PARAM_NAME: self.input_range.detach(),
                "num_bits": self._num_bits.detach()}

    def _prepare_export_quantization(self, x: torch.Tensor):
        self.freeze_num_bits()
        return super()._prepare_export_quantization(x)

    @torch.no_grad()
    def get_input_range(self):
        self.set_level_ranges()
        input_low, input_high = self._get_input_low_input_high(self.input_range,
                                                               self.input_low,
                                                               self.levels,
                                                               self.eps)
        return input_low, input_high
