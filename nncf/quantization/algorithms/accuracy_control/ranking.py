"""
 Copyright (c) 2023 Intel Corporation
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

import operator
from typing import List

import numpy as np


def normalized_mse(x_ref: np.ndarray, x_approx: np.ndarray) -> float:
    """
    Calculates normalized mean square error between `x_ref` and `x_approx`.
    The normalized mean square error is defined as

        NMSE(x_ref, x_approx) = MSE(x_ref, x_approx) / MSE(x_ref, 0)

    :param x_ref: A 1-D array of (N,) shape. Represents the reference values.
    :param x_approx: A 1-D array of (N,) shape. Represents the measured values.
    :return: The normalized mean square error between `x_ref` and `x_approx`.
    """
    error = x_ref - x_approx
    nmse = np.dot(error, error) / np.dot(x_ref, x_ref)
    return nmse


def get_ranking_subset_indices(errors: List[float], subset_size: int) -> List[int]:
     """
     Returns `subset_size` indices of elements in the `errors` list
     that have the biggest error value. Returned indices are sorted in
     ascending order.

     :param errors: A list of errors.
     :param subset_size: A number of returned indices.
     :return: Indices of elements in the `errors` list which have the biggest error value.
     """
     ordered_indices = [
          idx for idx, _ in sorted(enumerate(errors), key=operator.itemgetter(1), reverse=True)
     ]
     end_index = min(subset_size, len(ordered_indices))
     return sorted(ordered_indices[:end_index])
