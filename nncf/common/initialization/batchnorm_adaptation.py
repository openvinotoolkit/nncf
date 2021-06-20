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

import math

from nncf.api.compression import ModelType
from nncf.common.initialization.dataloader import NNCFDataLoader
from nncf.common.utils.backend import __nncf_backend__


class BatchnormAdaptationAlgorithmImpl(ABC):
    """
    This is the class from which all framework-specific implementations of
    the batch-norm statistics adaptation algorithm inherit.
    """

    def __init__(self,
                 data_loader: NNCFDataLoader,
                 num_bn_adaptation_steps: int,
                 num_bn_forget_steps: int,
                 device: Optional[str] = None):
        """
        Initializes the batch-norm statistics adaptation algorithm implementation.

        :param data_loader: NNCF data loader.
        :param num_bn_adaptation_steps: Number of batches from the training dataset to pass
            through the model at initialization in order to update batch-norm statistics of
            the original model.
        :param num_bn_forget_steps: Number of batches from the training dataset to pass
            through the model at initialization in order to erase batch-norm statistics of
            the original model.
        :param device: Device to perform initialization. If `device` is `None` then the device
            of the model parameters will be used.
        """
        self._data_loader = data_loader
        self._num_bn_adaptation_steps = num_bn_adaptation_steps
        self._num_bn_forget_steps = num_bn_forget_steps
        self._device = device

    @abstractmethod
    def run(self, model: ModelType) -> None:
        """
        Runs the batch-norm statistics adaptation algorithm. This method contains the implementation
        of the algorithm.
        """


def _create_bn_adaptation_algorithm_impl(data_loader: NNCFDataLoader,
                                         num_bn_adaptation_steps: int,
                                         num_bn_forget_steps: int,
                                         device: Optional[str] = None) -> BatchnormAdaptationAlgorithmImpl:
    """
    Factory for building a batchnorm adaptation algorithm implementation.

    :return: Implementation of the `BatchnormAdaptationAlgorithmImpl` class.
    """
    if __nncf_backend__ == 'Torch':
        from nncf.torch.batchnorm_adaptation import PTBatchnormAdaptationAlgorithmImpl
        bn_adaptation_algorithm_impl = PTBatchnormAdaptationAlgorithmImpl(data_loader,
                                                                          num_bn_adaptation_steps,
                                                                          num_bn_forget_steps,
                                                                          device)
    elif __nncf_backend__ == 'TensorFlow':
        from nncf.tensorflow.batchnorm_adaptation import TFBatchnormAdaptationAlgorithmImpl
        bn_adaptation_algorithm_impl = TFBatchnormAdaptationAlgorithmImpl(data_loader,
                                                                          num_bn_adaptation_steps,
                                                                          num_bn_forget_steps,
                                                                          device)

    return bn_adaptation_algorithm_impl


class BatchnormAdaptationAlgorithm:
    """
    This algorithm updates the statistics of the batch normalization layers
    passing several batches of data through the model. This allows to correct
    the compression-induced bias in the model and reduce the corresponding
    accuracy drop even before model training.
    """

    def __init__(self,
                 data_loader: NNCFDataLoader,
                 num_bn_adaptation_samples: int,
                 num_bn_forget_samples: int,
                 device: Optional[str] = None):
        """
        Initializes the batch-norm statistics adaptation algorithm.

        :param data_loader: NNCF data loader.
        :param num_bn_adaptation_samples: Number of samples from the training
            dataset to pass through the model at initialization in order to update
            batch-norm statistics of the original model. The actual number of samples
            will be a closest multiple of the batch size.
        :param num_bn_forget_samples: Number of samples from the training dataset to
            pass through the model at initialization in order to erase batch-norm
            statistics of the original model (using large momentum value for rolling
            mean updates). The actual number of samples will be a closest multiple of
            the batch size.
        :param device: Device to perform initialization. If `device` is `None` then the device
            of the model parameters will be used.
        """
        if num_bn_adaptation_samples < 0:
            raise ValueError('Number of adaptation samples must be >= 0')

        num_bn_adaptation_steps = math.ceil(num_bn_adaptation_samples / data_loader.batch_size)
        num_bn_forget_steps = math.ceil(num_bn_forget_samples / data_loader.batch_size)

        self._impl = _create_bn_adaptation_algorithm_impl(data_loader,
                                                          num_bn_adaptation_steps,
                                                          num_bn_forget_steps,
                                                          device)

    def run(self, model: ModelType) -> None:
        """
        Runs the batch-norm statistics adaptation algorithm.

        :param model: A model for which the algorithm will be applied.
        """
        self._impl.run(model)
