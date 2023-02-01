"""
 Copyright (c) 2023 Intel Corporation
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

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import torch
from nncf.common.quantization.structs import QuantizationPreset
from nncf.config import NNCFConfig
from nncf.config.structures import BNAdaptationInitArgs
from nncf.config.structures import QuantizationRangeInitArgs
from nncf.data import Dataset
from nncf.parameters import convert_ignored_scope_to_list
from nncf.parameters import IgnoredScope
from nncf.parameters import ModelType
from nncf.parameters import TargetDevice
from nncf.torch.dynamic_graph.context import no_nncf_trace
from nncf.torch.dynamic_graph.io_handling import replicate_same_tensors
from nncf.torch.dynamic_graph.io_handling import wrap_nncf_model_inputs_with_objwalk
from nncf.torch.dynamic_graph.io_handling import wrap_nncf_model_outputs_with_objwalk
from nncf.torch.initialization import PTInitializingDataLoader
from nncf.torch.model_creation import create_compressed_model
from nncf.torch.nested_objects_traversal import objwalk
from nncf.torch.utils import get_model_device
from nncf.torch.utils import is_tensor


# TODO(alexsu52): It is a workaround and should be removed.
class CalibrarionDataLoader(PTInitializingDataLoader):
    """
    This class wraps the nncf.Dataset.

    This is required for proper initialization of certain compression algorithms.
    """

    def __init__(self, data_loader: Dataset):
        super().__init__(data_loader)
        self._length = None

    @property
    def batch_size(self):
        data_source = getattr(self._data_loader, '_data_source')
        return getattr(data_source, 'batch_size', 1)

    def __iter__(self):
        return iter(self._data_loader.get_inference_data())

    def __len__(self):
        if self._length is None:
            data = self._data_loader.get_inference_data()
            self._length = CalibrarionDataLoader._get_length(data)
        return self._length

    def get_inputs(self, dataloader_output: Any) -> Tuple[Tuple, Dict]:
        if not isinstance(dataloader_output, tuple):
            dataloader_output = (dataloader_output, )
        return dataloader_output, {}

    @staticmethod
    def _get_length(iterable) -> int:
        length = 0
        for _ in iterable:
            length = length + 1

        return length


def _get_transformer_quantization_config(subset_size: int) -> Dict:
    """
    :return: The quantization config for transformer-based models.
    """
    return {
        'algorithm': 'quantization',
        'preset': 'mixed',
        'initializer': {
            'range': {'num_init_samples': subset_size, 'type': 'mean_min_max'},
            'batchnorm_adaptation': {'num_bn_adaptation_samples': 0},
        },
        'scope_overrides': {
            'activations': {'{re}.*matmul_0': {'mode': 'symmetric'}}},
        'ignored_scopes': [
            '{re}.*Embeddings.*',
            '{re}.*__add___[0-1]',
            '{re}.*layer_norm_0',
            '{re}.*matmul_1',
            '{re}.*__truediv__*',
        ],
        'overflow_fix': 'disable',
    }


def _get_default_quantization_config(preset: QuantizationPreset,
                                     subset_size: int) -> Dict:
    """
    :return: The default quantization config.
    """
    return {
        'algorithm': 'quantization',
        'preset': preset.value,
        'initializer': {
            'range': {'num_init_samples': subset_size},
            'batchnorm_adaptation': {'num_bn_adaptation_samples': 0}
        },
        'overflow_fix': 'first_layer_only'
    }


def _create_nncf_config(preset: QuantizationPreset,
                        target_device: TargetDevice,
                        subset_size: int,
                        model_type: Optional[ModelType],
                        ignored_scope: Optional[IgnoredScope]) -> NNCFConfig:
    """
    :return: The NNCFConfig for quantization method.
    """
    if model_type is None:
        compression_config = _get_default_quantization_config(preset, subset_size)
    elif model_type == ModelType.TRANSFORMER:
        compression_config = _get_transformer_quantization_config(subset_size)

    if ignored_scope is not None:
        _ignored_scope = convert_ignored_scope_to_list(ignored_scope)
        if 'ignored_scopes' in compression_config:
            compression_config['ignored_scopes'].extend(_ignored_scope)
        else:
            compression_config['ignored_scopes'] = _ignored_scope

    return NNCFConfig({
        'target_device': target_device.value,
        'compression': compression_config
    })


def quantize_impl(model: torch.nn.Module,
                  calibration_dataset: Dataset,
                  preset: QuantizationPreset,
                  target_device: TargetDevice,
                  subset_size: int,
                  fast_bias_correction: bool,
                  model_type: Optional[ModelType] = None,
                  ignored_scope: Optional[IgnoredScope] = None) -> torch.nn.Module:
    """
    Implementation of the `quantize()` method for the PyTorch backend.
    """
    if fast_bias_correction is False:
        raise ValueError(f'fast_bias_correction={fast_bias_correction} is not '
                          'supported')
    if ignored_scope is not None and ignored_scope.types is not None:
        raise RuntimeError('Quantization algorithm from the PyTorch backend '
                            'does not support operation types in the ignored '
                            'scopes yet')
    if target_device == TargetDevice.CPU_SPR:
        raise RuntimeError('target_device == CPU_SPR is not supported')

    nncf_config = _create_nncf_config(preset, target_device, subset_size,
                                      model_type, ignored_scope)

    calibration_data_loader = CalibrarionDataLoader(calibration_dataset)
    nncf_config.register_extra_structs(
        [
            QuantizationRangeInitArgs(data_loader=calibration_data_loader),
            BNAdaptationInitArgs(data_loader=calibration_data_loader)
        ]
    )

    def wrap_inputs(args, kwargs):
        return wrap_nncf_model_inputs_with_objwalk(args, kwargs)

    def wrap_outputs(retval):
        return wrap_nncf_model_outputs_with_objwalk(retval)

    def create_dummy_forward_fn(data_loader, device):
        def dummy_forward(model):
            with no_nncf_trace():
                data_item = next(iter(data_loader))
                args, kwargs = data_loader.get_inputs(data_item)

                def send_to_device(tensor):
                    return tensor.to(device)

                args = objwalk(args, is_tensor, send_to_device)
                kwargs = objwalk(kwargs, is_tensor, send_to_device)

            args, kwargs = wrap_inputs(args, kwargs)
            retval =  model(*args, **kwargs)
            retval = replicate_same_tensors(retval)
            return wrap_outputs(retval)

        return dummy_forward

    dummy_forward_fn = create_dummy_forward_fn(calibration_data_loader,
                                               get_model_device(model))

    clone_model = deepcopy(model)
    compression_ctrl, compressed_model = create_compressed_model(
        model=clone_model,
        config=nncf_config,
        dummy_forward_fn=dummy_forward_fn,
        wrap_inputs_fn=wrap_inputs,
        wrap_outputs_fn=wrap_outputs
    )
    compression_ctrl.prepare_for_export()
    compressed_model.disable_dynamic_graph_building()

    return compressed_model
