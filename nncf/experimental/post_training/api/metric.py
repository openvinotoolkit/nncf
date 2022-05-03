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

from typing import TypeVar

from abc import ABC
from abc import abstractmethod

Output = TypeVar('Output')
Target = TypeVar('Target')


class Metric(ABC):
    """
    An abstract class representing an metric.
    """

    def __init__(self):
        self.reset()

    @property
    def value(self):
        """
        Computes metric value for the one dataset sample
        """
        raise NotImplementedError('The value() property should be implemented to use this metric '
                                  'with AccuracyAwareQuantization algorithm!')

    @property
    @abstractmethod
    def avg_value(self):
        """
        Computes metric value across dataset
        """

    @property
    def higher_better(self) -> bool:
        """
        Boolean attribute whether the metric should be increased
        """
        return True

    @abstractmethod
    def update(self, output: Output, target: Target) -> None:
        """
        Calculates and updates metric value
        
        :param output: model output
        :param target: annotation for metric calculation
        """

    @abstractmethod
    def reset(self) -> None:
        """ Reset metric """
