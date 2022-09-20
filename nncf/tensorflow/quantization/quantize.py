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

from typing import Optional

import tensorflow as tf

from nncf import Dataset
from nncf import NNCFConfig
from nncf import QuantizationPreset
from nncf import TargetDevice
from nncf.common.initialization.dataloader import NNCFDataLoader
from nncf.config.structures import BNAdaptationInitArgs
from nncf.config.structures import QuantizationRangeInitArgs
from nncf.data.dataset import DataProvider
from nncf.tensorflow.helpers.model_creation import create_compressed_model


# TODO(alexsu52): It is a workaround and should be removed.
class CalibrarionDataLoader(NNCFDataLoader):
    """
    This class wraps the nncf.Dataset.

    This is required for proper initialization of certain compression algorithms.
    """

    def __init__(self, dataset: Dataset):
        self._dataset = dataset

    @property
    def batch_size(self) -> int:
        data_source = self._dataset.data_source

        if not hasattr(data_source, '_batch_size'):
            return 1
        batch_size = getattr(data_source, '_batch_size')
        try:
            if isinstance(batch_size, tf.Tensor):
                batch_size = batch_size.numpy()
            batch_size = int(batch_size)
        except:
            batch_size = 1
        return batch_size

    def __iter__(self):
        def transform_fn(data_item):
            return data_item, None

        return iter(DataProvider(self._dataset.get_inference_data(), transform_fn))


def quantize_impl(model: tf.Module,
                  calibration_dataset: Dataset,
                  preset: QuantizationPreset,
                  target_device: TargetDevice,
                  subset_size: int,
                  fast_bias_correction: bool,
                  model_type: Optional[str] = None) -> tf.Module:
    """
    Implementation of the `quantize()` method for the TensorFlow backend.
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

    compression_ctrl, compressed_model = create_compressed_model(
        model=model,
        config=nncf_config
    )
    stripped_model = compression_ctrl.strip_model(compressed_model)

    return stripped_model
