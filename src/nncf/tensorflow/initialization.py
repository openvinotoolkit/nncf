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

import tensorflow as tf

from nncf.common.initialization.dataloader import NNCFDataLoader
from nncf.common.utils.api_marker import api
from nncf.config import NNCFConfig
from nncf.config.structures import BNAdaptationInitArgs
from nncf.config.structures import QuantizationRangeInitArgs


class TFInitializingDataLoader(NNCFDataLoader):
    """
    This class wraps the tf.data.Dataset class.

    This is required for proper initialization of certain compression algorithms.
    """

    def __init__(self, data_loader: tf.data.Dataset, batch_size: int):
        self._data_loader = data_loader
        self._batch_size = batch_size

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def __iter__(self):
        return iter(self._data_loader)


@api(canonical_alias="nncf.tensorflow.register_default_init_args")
def register_default_init_args(
    nncf_config: NNCFConfig, data_loader: tf.data.Dataset, batch_size: int, device: str = None
) -> NNCFConfig:
    """
    Register extra structures in the NNCFConfig. Initialization of some
    compression algorithms requires certain extra structures.

    :param nncf_config: An instance of the NNCFConfig class without extra structures.
    :type nncf_config: nncf.NNCFConfig
    :param data_loader: Dataset used for initialization.
    :param batch_size: Batch size used for initialization.
    :param device: Device to perform initialization. If `device` is `None` then the device
        of the model parameters will be used.
    :return: An instance of the NNCFConfig class with extra structures.
    :rtype: nncf.NNCFConfig
    """
    nncf_config.register_extra_structs(
        [
            QuantizationRangeInitArgs(data_loader=TFInitializingDataLoader(data_loader, batch_size), device=device),
            BNAdaptationInitArgs(data_loader=TFInitializingDataLoader(data_loader, batch_size), device=device),
        ]
    )
    return nncf_config
