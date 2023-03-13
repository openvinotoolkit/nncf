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

from typing import Dict

import numpy as np


def normalized_mse(x_ref: Dict[str, np.ndarray], x_approx: Dict[str, np.ndarray]) -> float:
    """
    Calculates normalized mean square error between `x_ref` and `x_approx`.
    The normalized mean square error is defined as

    NMSE(x_ref, x_approx) = MSE(x_ref, x_approx) / MSE(x_ref, 0)

    :param x_ref: Dictionary of arrays. Represents the reference values.
    :param x_approx: Dictionary of arrays. Represents the measured values.
    :return: The normalized mean square error between `x_ref` and `x_approx`.
    """
    metrics = []
    for output_name in x_ref:
        error_flattened = (x_ref[output_name] - x_approx[output_name]).flatten()
        x_ref_flattened = x_ref[output_name].flatten()
        nmse = np.dot(error_flattened, error_flattened) / np.dot(x_ref_flattened, x_ref_flattened)
        metrics.append(nmse)
    nmse = sum(metrics) / len(metrics)
    return nmse
