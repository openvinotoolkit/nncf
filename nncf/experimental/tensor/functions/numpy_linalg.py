# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Tuple, Union

import numpy as np
import scipy

from nncf.experimental.tensor.functions import linalg
from nncf.experimental.tensor.functions.dispatcher import register_numpy_types


@register_numpy_types(linalg.norm)
def _(
    a: Union[np.ndarray, np.generic],
    ord: Optional[Union[str, float, int]] = None,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> np.ndarray:
    return np.array(np.linalg.norm(a, ord=ord, axis=axis, keepdims=keepdims))


@register_numpy_types(linalg.cholesky)
def _(a: Union[np.ndarray, np.generic], upper: bool = False) -> np.ndarray:
    if a.ndim != 2:
        raise ValueError(f"Input tensor needs to be 2D but received a {a.ndim}d-tensor.")
    return scipy.linalg.cholesky(a, lower=not upper)


@register_numpy_types(linalg.cholesky_inverse)
def _(a: Union[np.ndarray, np.generic], upper: bool = False) -> np.ndarray:
    if a.ndim != 2:
        raise ValueError(f"Input tensor needs to be 2D but received a {a.ndim}d-tensor.")
    c = np.linalg.inv(a)
    if upper:
        return np.dot(c, c.T)
    return np.dot(c.T, c)


@register_numpy_types(linalg.inv)
def _(a: Union[np.ndarray, np.generic]) -> np.ndarray:
    return np.linalg.inv(a)
