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
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from functools import partial

from nncf.torch.dynamic_graph.context import no_nncf_trace
from torch import distributed

from nncf.torch.checkpoint_loading import OPTIONAL_PARAMETERS_REGISTRY
from nncf.torch.debug import is_debug
from nncf.torch.functions import clamp
from nncf.common.utils.logger import logger as nncf_logger
from nncf.common.quantization.structs import QuantizationMode, QuantizerConfig, QuantizerSpec
from nncf.common.quantization.quantizers import calculate_symmetric_level_ranges
from nncf.common.quantization.quantizers import calculate_asymmetric_level_ranges
from nncf.torch.quantization.quantize_functions import symmetric_quantize, asymmetric_quantize, \
    ExportQuantizeToFakeQuantize, get_scale_zp_from_input_low_input_high, ExportQuantizeToONNXQuantDequant, TuneRange
from nncf.torch.layer_utils import COMPRESSION_MODULES, CompressionParameter
from nncf.common.utils.registry import Registry
from nncf.torch.utils import get_flat_tensor_contents_string, no_jit_trace, is_tracing_state

QUANTIZATION_MODULES = Registry('quantization_modules')
INITIALIZABLE_MODULES = Registry('initializable_modules')


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


class PTQuantizerSpec(QuantizerSpec):
    def __init__(self, num_bits: int,
                 mode: QuantizationMode,
                 signedness_to_force: Optional[bool],
                 narrow_range: bool,
                 half_range: bool,
                 scale_shape: Tuple[int, ...],
                 logarithm_scale: bool,
                 compression_lr_multiplier: float = None):
        super().__init__(num_bits, mode, signedness_to_force, narrow_range, half_range)
        self.scale_shape = scale_shape
        self.logarithm_scale = logarithm_scale
        self.compression_lr_multiplier = compression_lr_multiplier


    @classmethod
    def from_config(cls, qconfig: QuantizerConfig, narrow_range: bool,
                    half_range: bool, scale_shape: Tuple[int],
                    logarithm_scale: bool, compression_lr_multiplier: float) -> 'PTQuantizerSpec':
        return cls(qconfig.num_bits,
                   qconfig.mode,
                   qconfig.signedness_to_force,
                   narrow_range,
                   half_range,
                   scale_shape,
                   logarithm_scale,
                   compression_lr_multiplier)


class BaseQuantizer(nn.Module):
    # pylint:disable=too-many-public-methods
    def __init__(self, qspec: PTQuantizerSpec):
        super().__init__()
        self._narrow_range = qspec.narrow_range
        self._signedness_to_force = qspec.signedness_to_force
        self._is_using_log_scale_storage = qspec.logarithm_scale
        self._half_range = qspec.half_range
        self._num_bits = CompressionParameter(torch.IntTensor([qspec.num_bits]), requires_grad=False,
                                              compression_lr_multiplier=qspec.compression_lr_multiplier)
        OPTIONAL_PARAMETERS_REGISTRY.register('_num_bits')
        self.level_high = None
        self.level_low = None

        self.levels = 0
        ENABLED_VAR_NAME = 'enabled'
        self.register_buffer(ENABLED_VAR_NAME, torch.IntTensor([1]))
        OPTIONAL_PARAMETERS_REGISTRY.register(ENABLED_VAR_NAME)
        self.initialized = False
        self.call_count = 0
        self._scale_shape = qspec.scale_shape
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
                for module_key in module.state_dict().keys():
                    candidate = prefix + module_key
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
        is_exporting = is_tracing_state()
        if is_exporting:
            with no_nncf_trace():
                x = self.run_export_quantization(x)

            # The underlying operator (registered via register_operator) must be executed,
            # otherwise the dynamic graph won't be traced as it was during regular inference.
            # While this does not impact the regular, non-RNN models, for which the graph
            # building and pre-/post-hook calling is only determined by input-agnostic,
            # graph-structure independent trace info (e.g. current op scope and call count),
            # this is important for LSTMs etc. where determining the "first nodes in iteration
            # scopes" depends on whether the input tensors to an operation were traced or not.
            return self.quantize(x, execute_traced_op_as_identity=True)

        return self.quantize(x, execute_traced_op_as_identity=False)

    def quantize(self, x, execute_traced_op_as_identity: bool = False):
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

    @property
    def narrow_range(self) -> bool:
        return self._narrow_range

    @property
    def scale_shape(self) -> Tuple[int, ...]:
        # Per-tensor scale shapes are (1,)
        return self._scale_shape

    def broadcast_initialized_params(self, src: int = 0):
        distributed.broadcast(self._num_bits, src=src)

    def set_export_mode(self, mode: QuantizerExportMode):
        self._export_mode = mode

    def _get_input_low_input_high(self):
        raise NotImplementedError

    def _prepare_export_quantization(self, x: torch.Tensor):
        raise NotImplementedError

    def _prepare_fq_export_quantization(self, x: torch.Tensor):
        x, level_high, level_low, input_low, input_high = self._prepare_export_quantization(x)
        with no_jit_trace():
            levels = level_high - level_low + 1
        return x, levels, input_low, input_high

    def _prepare_qdq_export_quantization(self, x: torch.Tensor):
        x, level_high, level_low, input_low, input_high = self._prepare_export_quantization(x)
        with no_jit_trace():
            y_scale, y_zero_point = get_scale_zp_from_input_low_input_high(level_low,
                                                                           level_high,
                                                                           input_low,
                                                                           input_high)
        return x, y_scale, y_zero_point

    def run_export_quantization(self, x: torch.Tensor):
        if self._export_mode == QuantizerExportMode.FAKE_QUANTIZE:
            x, levels, input_low, input_high = self._prepare_fq_export_quantization(x)
            return ExportQuantizeToFakeQuantize.apply(x, levels,
                                                      input_low,
                                                      input_high,
                                                      input_low,
                                                      input_high)
        if self._export_mode == QuantizerExportMode.ONNX_QUANTIZE_DEQUANTIZE_PAIRS:
            x, y_scale, y_zero_point = self._prepare_qdq_export_quantization(x)
            if self.per_channel and y_zero_point.numel() > 1:
                if torch.allclose(y_scale - y_scale[0], torch.zeros_like(y_scale)) and \
                        torch.allclose(y_zero_point - y_zero_point[0], torch.zeros_like(y_zero_point)):
                    y_scale, y_zero_point = y_scale[0], y_zero_point[0]
                    return ExportQuantizeToONNXQuantDequant.apply(x, y_scale, y_zero_point)
                raise RuntimeError("PyTorch export to ONNX using QuantizeLinear-DequantizeLinear "
                                   "doesn't support per channel quantization")
            return ExportQuantizeToONNXQuantDequant.apply(x, y_scale, y_zero_point)
        raise RuntimeError('Unknown export mode')

    def extra_repr(self):
        return 'bit={}, ch={}'.format(
            self.num_bits, self.per_channel)

    def get_quantizer_config(self) -> QuantizerConfig:
        raise NotImplementedError

    @property
    def per_channel(self) -> bool:
        numel = 1
        for el in self.scale_shape:
            numel *= el
        is_per_tensor = ((numel == 1) and (len(self.scale_shape) == 1))
        return not is_per_tensor


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

    def __init__(self, qspec: PTQuantizerSpec):
        super().__init__(qspec)
        self.signed_tensor = CompressionParameter(torch.IntTensor([0]), requires_grad=False,
                                                  compression_lr_multiplier=qspec.compression_lr_multiplier)
        self.collect_scale_statistics = False

        setattr(self, self._SCALE_PARAM_STORAGE_ATTR,
                CompressionParameter(torch.ones(self.scale_shape), requires_grad=True,
                                     compression_lr_multiplier=qspec.compression_lr_multiplier))
        if self._is_using_log_scale_storage:
            self._scale_param_storage.data.log_()
            self.eps = 0
        else:
            self.eps = 1e-16
        if qspec.signedness_to_force is not None:
            self.signed = int(qspec.signedness_to_force)
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
        """
        Need to handle the redirect-storage attributes (which are implemented using Python properties
        here) specially - otherwise the torch.nn.Module's __setattr__ will try to set them during
        assignment.
        """
        if key == self.SCALE_PARAM_NAME:
            object.__setattr__(self, key, value)
        else:
            super().__setattr__(key, value)

    def enable_gradients(self):
        self._scale_param_storage.requires_grad = True

    def disable_gradients(self):
        self._scale_param_storage.requires_grad = False

    def set_level_ranges(self):
        scaled_num_bits = 1 if self._half_range else 0
        self.level_low, self.level_high, self.levels = self.calculate_level_ranges(self.num_bits - scaled_num_bits,
                                                                                   self.signed)

    @staticmethod
    def calculate_level_ranges(num_bits, signed):
        return calculate_symmetric_level_ranges(num_bits, signed)

    @property
    def signed(self):
        return self.signed_tensor.item() == 1

    @signed.setter
    def signed(self, signed: bool):
        self.signed_tensor.fill_(signed)

    def quantize(self, x, execute_traced_op_as_identity: bool = False):
        return symmetric_quantize(x, self.levels, self.level_low, self.level_high, self.scale, self.eps,
                                  skip=execute_traced_op_as_identity)

    def get_trainable_params(self) -> Dict[str, torch.Tensor]:
        return {self.SCALE_PARAM_NAME: self.scale.detach()}

    def _apply_minmax_init(self, min_values, max_values, log_module_name: str = None):
        if torch.any(torch.eq(min_values, np.inf)) or torch.any(torch.eq(max_values, -np.inf)):
            raise AttributeError('Statistics is not collected for {}'.format(log_module_name))
        sign = torch.any(torch.lt(min_values, 0))
        if self._signedness_to_force is not None and sign != self._signedness_to_force:
            nncf_logger.warning("Forcing signed to {} for module {}".format(self._signedness_to_force, log_module_name))
            sign = self._signedness_to_force
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

    def _get_input_low_input_high(self, scale, level_low, level_high, eps):
        input_range = abs(scale) + eps
        input_low = input_range * level_low / level_high
        input_high = input_range
        return input_low, input_high

    def _prepare_export_quantization(self, x: torch.Tensor):
        with no_jit_trace():
            input_low, input_high = self._get_input_low_input_high(self.scale,
                                                                   self.level_low,
                                                                   self.level_high,
                                                                   self.eps)
            level_low = self.level_low
            level_high = self.level_high
            if self._half_range:
                if self.scale_shape[0] > 1:
                    for i in range(self.scale_shape[0]):
                        x[i] = torch.clamp(x[i], min=input_low[i].item(), max=input_high[i].item())
                else:
                    x.data = torch.clamp(x, min=input_low.item(), max=input_high.item())
                level_low = 2 * self.level_low
                level_high = 2 * self.level_high + 1
                input_low, input_high = self._get_input_low_input_high(level_high / self.level_high * self.scale,
                                                                       level_low,
                                                                       level_high,
                                                                       self.eps)
        return x, level_high, level_low, input_low, input_high

    def get_quantizer_config(self) -> QuantizerConfig:
        return QuantizerConfig(num_bits=self.num_bits,
                               mode=QuantizationMode.SYMMETRIC,
                               signedness_to_force=self.signed,
                               per_channel=self.per_channel)


@COMPRESSION_MODULES.register()
@QUANTIZATION_MODULES.register(QuantizationMode.ASYMMETRIC)
class AsymmetricQuantizer(BaseQuantizer):
    INPUT_LOW_PARAM_NAME = 'input_low'
    INPUT_RANGE_PARAM_NAME = 'input_range'
    _INPUT_RANGE_PARAM_STORAGE_ATTR = '_input_range_param_storage'

    def __init__(self, qspec: PTQuantizerSpec):
        super().__init__(qspec)
        self.input_low = CompressionParameter(torch.zeros(self.scale_shape), requires_grad=True,
                                              compression_lr_multiplier=qspec.compression_lr_multiplier)
        setattr(self, self._INPUT_RANGE_PARAM_STORAGE_ATTR,
                CompressionParameter(torch.ones(self.scale_shape), requires_grad=True,
                                     compression_lr_multiplier=qspec.compression_lr_multiplier))

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
        """
        Need to handle the redirect-storage attributes (which are implemented using Python properties
        here) specially - otherwise the torch.nn.Module's __setattr__ will try to set them during
        assignment.
        """
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
        scaled_num_bits = 1 if self._half_range else 0
        self.level_low, self.level_high, self.levels = self.calculate_level_ranges(self.num_bits - scaled_num_bits)

    @staticmethod
    def calculate_level_ranges(num_bits):
        return calculate_asymmetric_level_ranges(num_bits)

    def quantize(self, x, execute_traced_op_as_identity: bool = False):
        return asymmetric_quantize(x, self.levels, self.level_low, self.level_high, self.input_low, self.input_range,
                                   self.eps, skip=execute_traced_op_as_identity)

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

    def _get_input_low_input_high(self, input_range, input_low, levels, eps):
        input_range_safe = abs(input_range) + eps
        input_low, input_range_tuned = TuneRange.apply(input_low, input_range_safe, levels)
        input_high = input_low + input_range_tuned
        return input_low, input_high

    def _prepare_export_quantization(self, x: torch.Tensor):
        with no_jit_trace():
            input_low, input_high = self._get_input_low_input_high(self.input_range,
                                                                   self.input_low,
                                                                   self.levels,
                                                                   self.eps)
            level_low = self.level_low
            level_high = self.level_high
            if self._half_range:
                if self.scale_shape[0] > 1:
                    for i in range(self.scale_shape[0]):
                        x[i] = torch.clamp(x[i], min=input_low[i].item(), max=input_high[i].item())
                else:
                    x.data = torch.clamp(x, min=input_low.item(), max=input_high.item())
                level_low = 2 * level_low
                level_high = 2 * level_high + 1
                input_low, input_high = self._get_input_low_input_high(level_high / self.level_high * self.input_range,
                                                                       self.input_low,
                                                                       self.levels,
                                                                       self.eps)

        return x, level_high, level_low, input_low, input_high

    def get_quantizer_config(self) -> QuantizerConfig:
        return QuantizerConfig(num_bits=self.num_bits,
                               mode=QuantizationMode.ASYMMETRIC,
                               signedness_to_force=self.signed,
                               per_channel=self.per_channel)
