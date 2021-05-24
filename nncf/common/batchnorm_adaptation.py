"""
 Copyright (c) 2021 Intel Corporation
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
from abc import ABC, abstractmethod

import numpy as np

from nncf.api.compression import ModelType
from nncf.common.initialization import NNCFDataLoader
from nncf.common.utils.logger import logger as nncf_logger
import nncf.common.factory as factory


class BatchnormAdaptationAlgorithmImpl(ABC):
    """
    This is the class from which all framework-specific implementations of
    the batch-norm adaptation algorithm inherit.
    """

    def __init__(self,
                 data_loader: NNCFDataLoader,
                 num_bn_adaptation_steps: int,
                 num_bn_forget_steps: int,
                 device: Optional[str] = None):
        """
        Initializes the batch-norm adaptation algorithm implementation.
        """
        self._data_loader = data_loader
        self._num_bn_adaptation_steps = num_bn_adaptation_steps
        self._num_bn_forget_steps = num_bn_forget_steps
        self._device = device

    @abstractmethod
    def run(self, model: ModelType):
        """
        Runs the batch-norm adaptation algorithm. This method contains the implementation
        of the algorithm.
        """


class BatchnormAdaptationAlgorithm:
    """
    This algorithm updates the statistics of the batch normalization layers
    passing several batches of data through the model. This allows to correct
    the compression-induced bias in the model and reduce the corresponding
    accuracy drop even before model training.
    """

    def __init__(self,
                 data_loader: NNCFDataLoader,
                 num_bn_adaptation_samples: int = 2000,
                 num_bn_forget_samples: int = 1000,
                 device: Optional[str] = None):
        """
        Initializes the batch-norm adaptation algorithm.

        :param data_loader: NNCF data loader.
        :param num_bn_adaptation_samples: Number of samples from the training
            dataset to pass through the model at initialization in order to update
            batchnorm statistics of the original model. The actual number of samples
            will be a closest multiple of the batch size.
        :param num_bn_forget_samples: Number of samples from the training dataset to
            pass through the model at initialization in order to erase batchnorm
            statistics of the original model (using large momentum value for rolling
            mean updates). The actual number of samples will be a closest multiple of
            the batch size.
        :param device:
        """
        if num_bn_adaptation_samples <= 0:
            raise ValueError('Number of adaptation samples must be > 0')

        self._impl = None
        if data_loader:
            num_bn_adaptation_steps = np.ceil(num_bn_adaptation_samples / data_loader.batch_size)
            num_bn_forget_steps = np.ceil(num_bn_forget_samples / data_loader.batch_size)

            self._impl = factory.create_bn_adaptation_algorithm_impl(data_loader,
                                                                     num_bn_adaptation_steps,
                                                                     num_bn_forget_steps,
                                                                     device)

    def run(self, model: ModelType):
        """
        Runs the batch-norm adaptation algorithm.

        :param model: A model for which the algorithm will be applied.
        """
        if self._impl:
            self._impl.run(model)
        else:
            nncf_logger.warning(
                'Could not run batchnorm adaptation as the adaptation data loader is not provided as an extra struct. '
                'Refer to `NNCFConfig.register_extra_structs` and the `BNAdaptationInitArgs` class.'
            )
