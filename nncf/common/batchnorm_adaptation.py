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

from abc import ABC, abstractmethod

from nncf.api.compression import ModelType
from nncf.config import NNCFConfig
import nncf.common.factory as factory


class BatchnormAdaptationAlgorithmImpl(ABC):
    """
    This is the class from which all framework-specific implementations of
    the batch-norm adaptation algorithm inherit.
    """

    @abstractmethod
    def run(self, model: ModelType, config: NNCFConfig):
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

    def __init__(self):
        """
        Initializes the batch-norm adaptation algorithm.
        """
        self._impl = factory.create_bn_adaptation_algorithm_impl()  # type: BatchnormAdaptationAlgorithmImpl

    def run(self, model: ModelType, config: NNCFConfig):
        """
        Runs the batch-norm adaptation algorithm.

        :param model: A model for which the algorithm will be applied.
        :param config: NNCF config.
        """
        self._impl.run(model, config)
