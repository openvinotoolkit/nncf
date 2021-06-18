"""
 Copyright (c) 2020 Intel Corporation
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

from nncf.common.initialization.dataloader import NNCFDataLoader


class NNCFExtraConfigStruct:
    """
    This is the class from which all extra structures that define additional
    NNCFConfig arguments inherit.
    """

    @classmethod
    def get_id(cls) -> str:
        raise NotImplementedError


class QuantizationRangeInitArgs(NNCFExtraConfigStruct):
    """
    Stores additional arguments for quantization range initialization algorithms.
    """

    def __init__(self,
                 data_loader: NNCFDataLoader,
                 device: Optional[str] = None):
        """
        Initializes additional arguments for quantization range initialization
        algorithms.

        :param data_loader: Provides an iterable over the given dataset.
        :param device: Device to perform initialization. If `device` is `None`
            then the device of the model parameters will be used.
        """
        self._data_loader = data_loader
        self._device = device

    @property
    def data_loader(self) -> NNCFDataLoader:
        return self._data_loader

    @property
    def device(self) -> str:
        return self._device

    @classmethod
    def get_id(cls) -> str:
        return 'quantization_range_init_args'


class BNAdaptationInitArgs(NNCFExtraConfigStruct):
    """
    Stores additional arguments for batchnorm statistics adaptation algorithm.
    """

    def __init__(self,
                 data_loader: NNCFDataLoader,
                 device: Optional[str] = None):
        """
        Initializes additional arguments for batchnorm statistics adaptation
        algorithm.

        :param data_loader: Provides an iterable over the given dataset.
        :param device: Device to perform initialization. If `device` is `None`
            then the device of the model parameters will be used.
        """
        self._data_loader = data_loader
        self._device = device

    @property
    def data_loader(self) -> NNCFDataLoader:
        return self._data_loader

    @property
    def device(self) -> str:
        return self._device

    @classmethod
    def get_id(cls) -> str:
        return 'bn_adaptation_init_args'
