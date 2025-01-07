# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, List, TypeVar

import numpy as np

import nncf
from nncf.common.utils.backend import BackendType

TTensor = TypeVar("TTensor")


def create_normalized_mse_func(backend: BackendType) -> Callable[[List[TTensor], List[TTensor]], float]:
    """
    Factory method to create backend-specific implementation of the normalized_nmse.

    :param backend: A backend type.
    :return: The backend-specific implementation of the normalized_nmse.
    """
    if backend == BackendType.OPENVINO:
        return normalized_mse

    raise nncf.UnsupportedBackendError(
        f"Could not create backend-specific implementation! {backend} backend is not supported!"
    )


def normalized_mse(ref_outputs: List[np.ndarray], approx_outputs: List[np.ndarray]) -> float:
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
