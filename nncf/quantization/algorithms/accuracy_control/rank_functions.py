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

import numpy as np

from nncf.quantization.algorithms.accuracy_control.evaluator import Output


def normalized_mse(ref_outputs: Output, approx_outputs: Output) -> float:
    """
    Calculates normalized mean square error between `ref_outputs` and `approx_outputs`.
    The normalized mean square error is defined as

    NMSE(x_ref, x_approx) = MSE(x_ref, x_approx) / MSE(x_ref, 0)

    :param ref_outputs: Reference outputs.
    :param approx_outputs: Approximate outputs.
    :return: The normalized mean square error between `ref_outputs` and `approx_outputs`.
    """
    metrics = []
    for x_ref, x_approx in zip(ref_outputs, approx_outputs):
        error_flattened = (x_ref - x_approx).flatten()
        x_ref_flattened = x_ref.flatten()
        nmse = np.dot(error_flattened, error_flattened) / np.dot(x_ref_flattened, x_ref_flattened)
        metrics.append(nmse)
    nmse = sum(metrics) / len(metrics)
    return nmse
