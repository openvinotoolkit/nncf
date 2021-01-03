"""
 Copyright (c) 2019 Intel Corporation
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
from enum import Enum
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from functools import partial
from torch import distributed

from nncf.debug import is_debug
from nncf.functions import clamp
from nncf.nncf_logger import logger as nncf_logger
from .quantize_functions import symmetric_quantize, asymmetric_quantize, ExportQuantizeToFakeQuantize, \
    get_scale_zp_from_input_low_input_high, ExportQuantizeToONNXQuantDequant, TuneRange
from ..layer_utils import COMPRESSION_MODULES
from ..registry import Registry
from ..utils import get_per_channel_scale_shape, get_flat_tensor_contents_string, no_jit_trace, is_tracing_state

QUANTIZATION_MODULES = Registry('quantization_modules')
INITIALIZABLE_MODULES = Registry('initializable_modules')


class QuantizationMode:
    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"


class QuantizerConfig:
    def __init__(self, bits=8,
                 mode=QuantizationMode.SYMMETRIC,
                 signedness_to_force=None,
                 per_channel=False,
                 input_shape=None,
                 is_weights=False,
                 logarithm_scale=False):
        self.bits = bits
        self.mode = mode
        self.signedness_to_force = signedness_to_force
        self.per_channel = per_channel
        self.is_weights = is_weights
        self.input_shape = input_shape
        self.logarithm_scale = logarithm_scale

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __str__(self):
        return "B:{bits} M:{mode} SGN:{signedness} W:{is_weights} PC:{per_channel}".format(
            bits=self.bits,
            mode='S' if self.mode == QuantizationMode.SYMMETRIC else 'A',
            signedness='ANY' if self.signedness_to_force is None else ('S' if self.signedness_to_force else 'U'),
            is_weights='Y' if self.is_weights else 'N',
            per_channel='Y' if self.per_channel else 'N')

    def __hash__(self):
        return hash(str(self))

    def __lt__(self, other):
        return self.bits < other.bits or \
               (self.mode == QuantizationMode.SYMMETRIC and other.mode == QuantizationMode.ASYMMETRIC) or \
               (self.signedness_to_force is None and other.signedness_to_force is not None) or \
               (not self.per_channel and other.per_channel)

    def compatible_with_a_unified_scale_linked_qconfig(self, linked_qconfig: 'QuantizerConfig'):
        return self.bits == linked_qconfig.bits and \
               self.mode == linked_qconfig.mode and \
               self.signedness_to_force == linked_qconfig.signedness_to_force and \
               self.per_channel == linked_qconfig.per_channel

    def is_a_bitwidth_variant(self, other_qconfig: 'QuantizerConfig') -> bool:
        return self.per_channel == other_qconfig.per_channel and \
               self.signedness_to_force == other_qconfig.signedness_to_force and \
               self.is_weights == other_qconfig.is_weights and \
               self.mode == other_qconfig.mode


class QuantizerExportMode(Enum):
    FAKE_QUANTIZE = "fake_quantize"
    ONNX_QUANTIZE_DEQUANTIZE_PAIRS = "quantize_dequantize"

    @staticmethod
    def from_str(config_value: str) -> 'HWConfigType':
        if config_value == QuantizerExportMode.FAKE_QUANTIZE.value:
            return QuantizerExportMode.FAKE_QUANTIZE
        if config_value == QuantizerExportMode.ONNX_QUANTIZE_DEQUANTIZE_PAIRS.value:
            return QuantizerExportMode.ONNX_QUANTIZE_DEQUANTIZE_PAIRS
        raise RuntimeError("Unknown quantizer ONNX export mode string")


class BaseQuantizer(nn.Module):
    def __init__(self, config: QuantizerConfig):
        super().__init__()
        self.input_shape = config.input_shape
        self.per_channel = config.per_channel
        self.is_weights = config.is_weights
        self.signedness_to_force = config.signedness_to_force
        self._is_using_log_scale_storage = config.logarithm_scale
        self._num_bits = nn.Parameter(torch.IntTensor([config.bits]), requires_grad=False)
        self.level_high = None
        self.level_low = None

        self.levels = 0

        self.register_buffer('enabled', torch.IntTensor([1]))
        self.initialized = False
        self.state_dict_name = None
        self.call_count = 0
        self.scale_shape = [1]
        self._export_mode = QuantizerExportMode.FAKE_QUANTIZE

        class LoadStateListener:
            """
               Check whether a quantization module are going to be updated by new values from state_dict or checkpoint.
            """

            def __init__(self, module):
                # pylint: disable=protected-access
                self.hook = module._register_load_state_dict_pre_hook(partial(self.hook_fn, module=module))

            def hook_fn(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs,
                        module):
                if module.state_dict_name:
                    for module_key in module.state_dict().keys():
                        candidate = module.state_dict_name + '.' + module_key
                        if candidate in state_dict:
                            module.initialized = True

            def close(self):
                self.hook.remove()

        self.load_listener = LoadStateListener(self)

    def enable_gradients(self):
        raise NotImplementedError

    def disable_gradients(self):
        raise NotImplementedError

    def is_enabled_quantization(self):
        return self.enabled[0] == 1

    def enable_quantization(self):
        self.enabled[0] = 1
        self.enable_gradients()

    def disable_quantization(self):
        self.enabled[0] = 0
        self.disable_gradients()

    def forward(self, x):
        if is_debug():
            self.call_count += 1
        # TODO: refactor to get rid of extra if's and calls on each forward
        if not self.is_enabled_quantization():
            return x
        self.set_level_ranges()
        if is_tracing_state():
            return self.run_export_quantization(x)

        return self.quantize(x)

    def quantize(self, x):
        raise NotImplementedError

    def reset_call_counter(self):
        self.call_count = 0

    def get_trainable_params(self) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def apply_minmax_init(self, min_values, max_values, log_module_name: str = None):
        """min_values and max_values must have the same shape as specified in self.scale_shape"""
        if self.initialized:
            nncf_logger.debug("Skipped initializing {} - loaded from checkpoint".format(log_module_name))
            return
        if torch.any(torch.eq(min_values, np.inf)) or torch.any(torch.eq(max_values, -np.inf)):
            raise AttributeError('Statistics is not collected for {}'.format(log_module_name))
        own_device = next(self.parameters()).device
        min_values = min_values.to(own_device)
        max_values = max_values.to(own_device)
        self._apply_minmax_init(min_values, max_values, log_module_name)

    def _apply_minmax_init(self, min_values, max_values, log_module_name: str = None):
        raise NotImplementedError

    def set_level_ranges(self):
        raise NotImplementedError

    @property
    def is_using_log_scale_storage(self):
        return self._is_using_log_scale_storage

    @property
    def signed(self):
        raise NotImplementedError

    @property
    def num_bits(self):
        return self._num_bits.item()

    @num_bits.setter
    def num_bits(self, num_bits: int):
        self._num_bits.fill_(num_bits)

    def broadcast_initialized_params(self, src: int = 0):
        distributed.broadcast(self._num_bits, src=src)

    def set_export_mode(self, mode: QuantizerExportMode):
        self._export_mode = mode

    def run_export_quantization(self, x: torch.Tensor):
        raise NotImplementedError

    def extra_repr(self):
        return 'bit={}, ch={}, wt={}'.format(
            self.num_bits, self.per_channel, self.is_weights)

    def get_current_config(self) -> QuantizerConfig:
        raise NotImplementedError


class QuantizersSwitcher:
    """ Enables/disables quantizers with saving and restoring original state """

    def __init__(self, quantizers: List[BaseQuantizer]):
        self.originally_disabled = []  # type: List[BaseQuantizer]
        self.originally_enabled = []  # type: List[BaseQuantizer]
        self._quantizers = quantizers

    def disable_quantizers(self):
        for module in self._quantizers:  # type: BaseQuantizer
            if not module.is_enabled_quantization():
                self.originally_disabled.append(module)
            if module not in self.originally_enabled:
                module.disable_quantization()
        self.originally_enabled = []

    def enable_quantizers(self):
        for module in self._quantizers:  # type: BaseQuantizer
            if module.is_enabled_quantization():
                self.originally_enabled.append(module)
            if module not in self.originally_disabled:
                module.enable_quantization()
        self.originally_disabled = []


class StorageRedirectingLoadStateDictHook:
    def __init__(self, storage_attribute_in_module: str, name_in_state_dict: str,
                 use_log_storage_in_module: bool = False):
        self._storage_attribute_in_module = storage_attribute_in_module
        self._name_in_state_dict = name_in_state_dict
        self._use_log_storage_in_module = use_log_storage_in_module

    def __call__(self, state_dict, prefix, local_metadata, strict,
                 missing_keys, unexpected_keys, error_msgs) -> None:
        state_dict_key = prefix + self._name_in_state_dict
        if state_dict_key in state_dict:
            v = state_dict.pop(state_dict_key)
            if self._use_log_storage_in_module:
                v = v.abs().log().detach()
            state_dict[prefix + self._storage_attribute_in_module] = v
        else:
            missing_keys.append(state_dict_key)


class StorageRedirectingStateDictHook:
    def __init__(self, storage_attribute_in_module: str, name_in_state_dict: str,
                 use_log_storage_in_module: bool = False):
        self._storage_attribute_in_module = storage_attribute_in_module
        self._name_in_state_dict = name_in_state_dict
        self._use_log_storage_in_module = use_log_storage_in_module

    def __call__(self, module, state_dict, prefix, local_metadata) -> None:
        v = state_dict.pop(prefix + self._storage_attribute_in_module)
        if self._use_log_storage_in_module:
            v = v.exp().detach()
        state_dict[prefix + self._name_in_state_dict] = v


@COMPRESSION_MODULES.register()
@QUANTIZATION_MODULES.register(QuantizationMode.SYMMETRIC)
class SymmetricQuantizer(BaseQuantizer):
    SCALE_PARAM_NAME = 'scale'
    _SCALE_PARAM_STORAGE_ATTR = '_scale_param_storage'

    def __init__(self, config):
        super().__init__(config)
        self.signed_tensor = nn.Parameter(torch.IntTensor([0]), requires_grad=False)
        self.collect_scale_statistics = False
        if self.per_channel:
            self.scale_shape = get_per_channel_scale_shape(self.input_shape, self.is_weights)

        setattr(self, self._SCALE_PARAM_STORAGE_ATTR, nn.Parameter(torch.ones(self.scale_shape), requires_grad=True))
        if self._is_using_log_scale_storage:
            self._scale_param_storage.data.log_()
            self.eps = 0
        else:
            self.eps = 1e-16
        if config.signedness_to_force is not None:
            self.signed = int(config.signedness_to_force)
        self.set_level_ranges()

        self._register_load_state_dict_pre_hook(StorageRedirectingLoadStateDictHook(
            storage_attribute_in_module=self._SCALE_PARAM_STORAGE_ATTR,
            name_in_state_dict=self.SCALE_PARAM_NAME,
            use_log_storage_in_module=self._is_using_log_scale_storage
        ))

        self._register_state_dict_hook(StorageRedirectingStateDictHook(
            storage_attribute_in_module=self._SCALE_PARAM_STORAGE_ATTR,
            name_in_state_dict=self.SCALE_PARAM_NAME,
            use_log_storage_in_module=self._is_using_log_scale_storage
        ))

    @property
    def scale(self):
        return self._scale_param_storage.exp() if self._is_using_log_scale_storage else self._scale_param_storage

    @scale.setter
    def scale(self, v):
        self._scale_param_storage = v
        if self._is_using_log_scale_storage:
            self._scale_param_storage.data.log_()

    def __setattr__(self, key, value):
        """Need to handle the redirect-storage attributes (which are implemented using Python properties
         here) specially - otherwise the torch.nn.Module's __setattr__ will try to set them during
         assignment."""
        if key == self.SCALE_PARAM_NAME:
            object.__setattr__(self, key, value)
        else:
            super().__setattr__(key, value)

    def enable_gradients(self):
        self._scale_param_storage.requires_grad = True

    def disable_gradients(self):
        self._scale_param_storage.requires_grad = False

    def set_level_ranges(self):
        self.level_low, self.level_high, self.levels = self.calculate_level_ranges(self.num_bits,
                                                                                   self.signed)

    @staticmethod
    def calculate_level_ranges(num_bits, signed):
        levels = 2 ** num_bits
        if signed:
            level_high = (levels // 2) - 1
            level_low = -(levels // 2)
        else:
            level_high = levels - 1
            level_low = 0
        return level_low, level_high, levels

    @property
    def signed(self):
        return self.signed_tensor.item() == 1

    @signed.setter
    def signed(self, signed: bool):
        self.signed_tensor.fill_(signed)

    def quantize(self, x):
        return symmetric_quantize(x, self.levels, self.level_low, self.level_high, self.scale, self.eps)

    def get_trainable_params(self) -> Dict[str, torch.Tensor]:
        return {self.SCALE_PARAM_NAME: self.scale.detach()}

    def _apply_minmax_init(self, min_values, max_values, log_module_name: str = None):
        if torch.any(torch.eq(min_values, np.inf)) or torch.any(torch.eq(max_values, -np.inf)):
            raise AttributeError('Statistics is not collected for {}'.format(log_module_name))
        sign = torch.any(torch.lt(min_values, 0))
        if self.signedness_to_force is not None and sign != self.signedness_to_force:
            nncf_logger.warning("Forcing signed to {} for module {}".format(self.signedness_to_force, log_module_name))
            sign = self.signedness_to_force
        self.signed = int(sign)

        abs_max = torch.max(torch.abs(max_values), torch.abs(min_values))
        SCALE_LOWER_THRESHOLD = 0.1
        mask = torch.gt(abs_max, SCALE_LOWER_THRESHOLD)
        self._scale_param_storage.data = torch.where(mask, abs_max,
                                                     SCALE_LOWER_THRESHOLD * torch.ones_like(self._scale_param_storage))
        if self._is_using_log_scale_storage:
            self._scale_param_storage.data.log_()

        nncf_logger.info("Set sign: {} and scale: {} for {}".format(self.signed,
                                                                    get_flat_tensor_contents_string(self.scale),
                                                                    log_module_name))

    def broadcast_initialized_params(self, src: int = 0):
        super().broadcast_initialized_params(src)
        distributed.broadcast(self._scale_param_storage, src=src)
        distributed.broadcast(self.signed_tensor, src=src)

    def run_export_quantization(self, x: torch.Tensor):
        with no_jit_trace():
            input_range = abs(self.scale) + self.eps
            # todo: take bias into account during input_low/input_high calculation
            input_low = input_range * self.level_low / self.level_high
            input_high = input_range

            if self._export_mode == QuantizerExportMode.ONNX_QUANTIZE_DEQUANTIZE_PAIRS:
                y_scale, y_zero_point = get_scale_zp_from_input_low_input_high(self.level_low,
                                                                               self.level_high,
                                                                               input_low,
                                                                               input_high)

        if self._export_mode == QuantizerExportMode.ONNX_QUANTIZE_DEQUANTIZE_PAIRS:
            if self.per_channel:
                if torch.allclose(y_scale - y_scale[0], torch.zeros_like(y_scale)) and torch.allclose(
                        y_zero_point - y_zero_point[0], torch.zeros_like(y_zero_point)):
                    y_scale, y_zero_point = y_scale[0], y_zero_point[0]
                    return ExportQuantizeToONNXQuantDequant.apply(x, y_scale, y_zero_point)
                raise RuntimeError("PyTorch v1.5.0 export to ONNX using QuantizeLinear-DequantizeLinear "
                                   "doesn't support per channel quantization")
            return ExportQuantizeToONNXQuantDequant.apply(x, y_scale, y_zero_point)
        if self._export_mode == QuantizerExportMode.FAKE_QUANTIZE:
            return ExportQuantizeToFakeQuantize.apply(x, self.levels, input_low, input_high, input_low, input_high)
        raise RuntimeError

    def get_current_config(self) -> QuantizerConfig:
        return QuantizerConfig(bits=self.num_bits,
                               mode=QuantizationMode.SYMMETRIC,
                               signedness_to_force=self.signed,
                               per_channel=self.per_channel,
                               input_shape=self.input_shape,
                               is_weights=self.is_weights,
                               logarithm_scale=self.is_using_log_scale_storage)

@COMPRESSION_MODULES.register()
@QUANTIZATION_MODULES.register(QuantizationMode.ASYMMETRIC)
class AsymmetricQuantizer(BaseQuantizer):
    INPUT_LOW_PARAM_NAME = 'input_low'
    INPUT_RANGE_PARAM_NAME = 'input_range'
    _INPUT_RANGE_PARAM_STORAGE_ATTR = '_input_range_param_storage'

    def __init__(self, config):
        super().__init__(config)
        if self.per_channel:
            self.scale_shape = get_per_channel_scale_shape(self.input_shape, self.is_weights)

        self.input_low = nn.Parameter(torch.zeros(self.scale_shape), requires_grad=True)
        setattr(self, self._INPUT_RANGE_PARAM_STORAGE_ATTR,
                nn.Parameter(torch.ones(self.scale_shape), requires_grad=True))

        if self._is_using_log_scale_storage:
            self._input_range_param_storage.data.log_()
            self.eps = 0
        else:
            self.eps = 1e-16
        self.set_level_ranges()

        self._register_load_state_dict_pre_hook(StorageRedirectingLoadStateDictHook(
            storage_attribute_in_module=self._INPUT_RANGE_PARAM_STORAGE_ATTR,
            name_in_state_dict=self.INPUT_RANGE_PARAM_NAME,
            use_log_storage_in_module=self._is_using_log_scale_storage
        ))

        self._register_state_dict_hook(StorageRedirectingStateDictHook(
            storage_attribute_in_module=self._INPUT_RANGE_PARAM_STORAGE_ATTR,
            name_in_state_dict=self.INPUT_RANGE_PARAM_NAME,
            use_log_storage_in_module=self._is_using_log_scale_storage
        ))

    @property
    def input_range(self):
        if self._is_using_log_scale_storage:
            return self._input_range_param_storage.exp()
        return self._input_range_param_storage

    @input_range.setter
    def input_range(self, v: torch.Tensor):
        self._input_range_param_storage = v
        if self._is_using_log_scale_storage:
            self._input_range_param_storage.data.log_()

    def __setattr__(self, key, value):
        """Need to handle the redirect-storage attributes (which are implemented using Python properties
         here) specially - otherwise the torch.nn.Module's __setattr__ will try to set them during
         assignment."""
        if key == self.INPUT_RANGE_PARAM_NAME:
            object.__setattr__(self, key, value)
        else:
            super().__setattr__(key, value)

    def enable_gradients(self):
        self.input_low.requires_grad = True
        self._input_range_param_storage.requires_grad = True

    def disable_gradients(self):
        self.input_low.requires_grad = False
        self._input_range_param_storage.requires_grad = False

    @property
    def signed(self):
        return True

    def set_level_ranges(self):
        self.level_low, self.level_high, self.levels = self.calculate_level_ranges(self.num_bits)

    @staticmethod
    def calculate_level_ranges(num_bits):
        level_high = 2 ** num_bits - 1
        level_low = 0
        levels = 2 ** num_bits
        return level_low, level_high, levels

    def quantize(self, x):
        return asymmetric_quantize(x, self.levels, self.level_low, self.level_high, self.input_low, self.input_range,
                                   self.eps)

    def get_trainable_params(self) -> Dict[str, torch.Tensor]:
        return {self.INPUT_LOW_PARAM_NAME: self.input_low.detach(),
                self.INPUT_RANGE_PARAM_NAME: self.input_range.detach()}

    def _apply_minmax_init(self, min_values, max_values, log_module_name: str = None):
        ranges = max_values - min_values
        max_range = torch.max(max_values - min_values)
        eps = 1e-2
        correction = (clamp(ranges, low=eps * max_range, high=max_range) - ranges) * 0.5
        self._input_range_param_storage.data = (ranges + 2 * correction).data
        if self._is_using_log_scale_storage:
            self._input_range_param_storage.data.log_()

        self.input_low.data = (min_values - correction).data

        nncf_logger.info("Set input_low: {} and input_range: {} for {}"
                         .format(get_flat_tensor_contents_string(self.input_low),
                                 get_flat_tensor_contents_string(self.input_range), log_module_name))

    def broadcast_initialized_params(self, src: int = 0):
        super().broadcast_initialized_params(src)
        distributed.broadcast(self.input_low, src)
        distributed.broadcast(self._input_range_param_storage, src)

    def run_export_quantization(self, x: torch.Tensor):
        with no_jit_trace():
            input_range_safe = abs(self.input_range) + self.eps
            input_low_tuned, input_range_tuned = TuneRange.apply(self.input_low, input_range_safe, self.levels)
            input_high_tuned = input_low_tuned + input_range_tuned

            if self._export_mode == QuantizerExportMode.ONNX_QUANTIZE_DEQUANTIZE_PAIRS:
                y_scale, y_zero_point = get_scale_zp_from_input_low_input_high(self.level_low,
                                                                               self.level_high,
                                                                               input_low_tuned,
                                                                               input_high_tuned)

        if self._export_mode == QuantizerExportMode.ONNX_QUANTIZE_DEQUANTIZE_PAIRS:
            if self.per_channel:
                if torch.allclose(y_scale - y_scale[0], torch.zeros_like(y_scale)) and torch.allclose(
                        y_zero_point - y_zero_point[0], torch.zeros_like(y_zero_point)):
                    y_scale, y_zero_point = y_scale[0], y_zero_point[0]
                    return ExportQuantizeToONNXQuantDequant.apply(x, y_scale, y_zero_point)
                raise RuntimeError("PyTorch v1.5.0 export to ONNX using QuantizeLinear-DequantizeLinear "
                                   "doesn't support per channel quantization")
            return ExportQuantizeToONNXQuantDequant.apply(x, y_scale, y_zero_point)
        if self._export_mode == QuantizerExportMode.FAKE_QUANTIZE:
            return ExportQuantizeToFakeQuantize.apply(x, self.levels,
                                                      input_low_tuned, input_high_tuned,
                                                      input_low_tuned, input_high_tuned)
        raise RuntimeError

    def get_current_config(self) -> QuantizerConfig:
        return QuantizerConfig(bits=self.num_bits,
                               mode=QuantizationMode.ASYMMETRIC,
                               signedness_to_force=self.signed,
                               per_channel=self.per_channel,
                               input_shape=self.input_shape,
                               is_weights=self.is_weights,
                               logarithm_scale=self.is_using_log_scale_storage)
