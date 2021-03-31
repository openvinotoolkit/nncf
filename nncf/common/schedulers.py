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

from typing import List, Optional
from bisect import bisect_right

import numpy as np


class PolynomialDecaySchedule:
    """
    This schedule applies a polynomial decay function to an epoch index.
    For more details about polynomial decay see the [paper](https://arxiv.org/abs/1710.01878).
    """

    def __init__(self, initial_value: float, target_value: float, target_epoch: int,
                 power: float, concave: bool):
        """
         Initializes a schedule with a polynomial decay function.

        :param initial_value: The initial value at which the schedule begins.
        :param target_value: The final value at which the schedule ends.
        :param target_epoch: Zero-based index of the epoch from which
            the function value will be equal to the `target_value` value.
        :param power: Exponent to be used in the polynomial decay function.
        :param concave: If true, then the `target_value` will be approached in a
            concave manner, and in a convex manner otherwise.
        """
        self.initial_value = initial_value
        self.target_value = target_value
        self.target_epoch = target_epoch
        self.power = power
        self.concave = concave

    def __call__(self, epoch: int, step: Optional[int] = None, steps_per_epoch: Optional[int] = None) -> float:
        """
        Calculates the value of the polynomial decay function. Two ways are possible:
            - Using epoch index only.
            - Using epoch index and step index. For this case, `steps_per_epoch` should
                be provided too.

        :param epoch: Zero-based epoch index for which the function value should be got.
        :param step: Local step index in `epoch` i.e. `step` should be in interval [0, steps_per_epoch).
            Used only if `steps_per_epoch` was provided.
        :param steps_per_epoch: A number of steps per epoch.
        """
        if self.target_epoch == 0:
            return self.target_value

        if step is not None and steps_per_epoch is not None:
            fractional_epoch = epoch + step / steps_per_epoch
            progress = fractional_epoch / self.target_epoch
        else:
            progress = epoch / self.target_epoch
        progress = min(1.0, max(0.0, progress))

        if self.concave:
            value = self.target_value - (self.target_value - self.initial_value) * np.power(1 - progress, self.power)
        else:
            value = self.initial_value + (self.target_value - self.initial_value) * np.power(progress, self.power)

        return value


class MultiStepSchedule:
    """
    This schedule applies a piecewise constant function to an epoch index
    """

    def __init__(self, boundaries: List[int], values: List[float]):
        """
        Initializes a schedule with a piecewise constant function.

        :param boundaries: List of zero-based epoch indices. Must be increasing.
        :param values: List of floats that specifies the values for the intervals
            defined by `boundaries`. It should have one more element than `boundaries`.
        :raises ValueError: If the number of elements in the `values` list does not
            equal to the number of elements in the `boundaries` list plus one.
        """
        if len(boundaries) + 1 != len(values):
            raise ValueError('The length of `values` should be 1 more than the length of `boundaries`')

        self.boundaries = boundaries
        self.values = values

    def __call__(self, epoch: int) -> float:
        """
        Calculates the value of the piecewise constant function for a given epoch index.
        The output of this call is `values[0]` when `epoch` < `boundaries[0]`,
        `values[1]` when  `boundaries[0]` <= `epoch` < `boundaries[1]`, ... , and
        `values[-1]` when `epoch` >= boundaries[-1].

        :param epoch: Zero-based epoch index for which the function value should be got.
        :return: The value of the piecewise constant function for a given epoch index.
        """
        pos = bisect_right(self.boundaries, epoch)
        return self.values[pos]


class ExponentialDecaySchedule:
    """
    This schedule applies an exponential decay function to an epoch index,
    considering a provided `initial_value` value. It is computed as:

        current_value = initial_value * decay_rate ^ (epoch / target_epoch),

    where `decay_rate` is equal to target_value / initial_value.
    """

    def __init__(self, initial_value: float, target_value: float, target_epoch: int):
        """
        Initializes a schedule with an exponential decay function.

        :param initial_value: The initial value at which the schedule begins.
        :param target_value: The final value at which the schedule end.
        :param target_epoch: Zero-based index of the epoch from which
            the function value will be equal to the `target_value` value.
        """
        self.initial_value = initial_value
        self.target_value = target_value
        self.target_epoch = target_epoch
        self.decay_rate = target_value / initial_value

    def __call__(self, epoch: int) -> float:
        """
        Calculates the value of the exponential decay function for a given epoch index.

        :param epoch: Zero-based epoch index for which the function value should be got.
        :return: The value of the exponential decay function for a given epoch index.
        """
        if self.target_epoch == 0:
            return self.target_value

        value = self.initial_value * np.power(self.decay_rate, epoch / self.target_epoch)
        return max(value, self.target_value)
