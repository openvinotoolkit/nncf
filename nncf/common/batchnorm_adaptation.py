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

from nncf.api.compression import ModelType
from nncf.config.structure import BNAdaptationInitArgs
import nncf.common.factory as factory


class BatchnormAdaptationAlgorithmImpl(ABC):
    """
    This is the class from which all framework-specific implementations of
    the batch-norm adaptation algorithm inherit.
    """

    def __init__(self,
                 num_bn_adaptation_samples: int,
                 num_bn_forget_samples: int,
                 extra_args: BNAdaptationInitArgs):
        """
        Initializes the batch-norm adaptation algorithm implementation.
        """
        self._num_bn_adaptation_samples = num_bn_adaptation_samples
        self._num_bn_forget_samples = num_bn_forget_samples
        self._extra_args = extra_args

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
                 num_bn_adaptation_samples: int = 2000,
                 num_bn_forget_samples: int = 1000,
                 extra_args: Optional[BNAdaptationInitArgs] = None):
        """
        Initializes the batch-norm adaptation algorithm.

        :param num_bn_adaptation_samples: Number of samples from the training
            dataset to pass through the model at initialization in order to update
            batchnorm statistics of the original model. The actual number of samples
            will be a closest multiple of the batch size.
        :param num_bn_forget_samples: Number of samples from the training dataset to
            pass through the model at initialization in order to erase batchnorm
            statistics of the original model (using large momentum value for rolling
            mean updates). The actual number of samples will be a closest multiple of
            the batch size.
        :param extra_args: Additional parameters for initialization.
        """
        if num_bn_adaptation_samples <= 0:
            raise ValueError('Number of adaptation samples must be > 0')

        self._impl = factory.create_bn_adaptation_algorithm_impl(num_bn_adaptation_samples,
                                                                 num_bn_forget_samples,
                                                                 extra_args)

    def run(self, model: ModelType):
        """
        Runs the batch-norm adaptation algorithm.

        :param model: A model for which the algorithm will be applied.
        """
        self._impl.run(model)
