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

from copy import deepcopy
from typing import Optional, Any, Tuple, Dict

import torch

from nncf import Dataset
from nncf import NNCFConfig
from nncf import QuantizationPreset
from nncf import TargetDevice
from nncf.config.structures import BNAdaptationInitArgs
from nncf.config.structures import QuantizationRangeInitArgs
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
        return getattr(self._data_loader.data_source, 'batch_size', 1)

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


def quantize_impl(model: torch.nn.Module,
                  calibration_dataset: Dataset,
                  preset: QuantizationPreset,
                  target_device: TargetDevice,
                  subset_size: int,
                  fast_bias_correction: bool,
                  model_type: Optional[str] = None) -> torch.nn.Module:
    """
    Implementation of the `quantize()` method for the PyTorch backend.
    """
    if model_type is not None:
        raise ValueError(f'model_type={model_type} is not supported')
    if fast_bias_correction == False:
        raise ValueError(f'fast_bias_correction={fast_bias_correction} is not supported')

    nncf_config = NNCFConfig(
        {
            "target_device": target_device.value,
            "compression": {
                "algorithm": "quantization",
                "preset": preset.value,
                "initializer": {
                    "range": {
                        "num_init_samples": subset_size
                    },
                    "batchnorm_adaptation": {
                        "num_bn_adaptation_samples": 0
                    }
                },
                "overflow_fix": "first_layer_only"
            }
        }
    )

    calibration_data_loader = CalibrarionDataLoader(calibration_dataset)
    nncf_config.register_extra_structs(
        [
            QuantizationRangeInitArgs(data_loader=calibration_data_loader),
            BNAdaptationInitArgs(data_loader=calibration_data_loader)
        ]
    )

    def create_dummy_forward_fn(data_loader, device):
        data_item = next(iter(data_loader))
        args, kwargs = data_loader.get_inputs(data_item)

        def send_to_device(tensor):
            return tensor.to(device)

        args = objwalk(args, is_tensor, send_to_device)
        kwargs = objwalk(kwargs, is_tensor, send_to_device)

        def dummy_forward(model):
            return model(*args, **kwargs)

        return dummy_forward

    def wrap_inputs(args, kwargs):
        return wrap_nncf_model_inputs_with_objwalk(args, kwargs)

    def wrap_outputs(retval):
        return wrap_nncf_model_outputs_with_objwalk(retval)

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
