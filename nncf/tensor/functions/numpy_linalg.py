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

from typing import Optional, Tuple, Union

import numpy as np
from scipy.linalg import lstsq

from nncf.tensor.functions import linalg
from nncf.tensor.functions.dispatcher import register_numpy_types


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
    lt = np.linalg.cholesky(a)
    if upper:
        return np.conjugate(np.swapaxes(lt, -2, -1))
    return lt


@register_numpy_types(linalg.cholesky_inverse)
def _(a: Union[np.ndarray, np.generic], upper: bool = False) -> np.ndarray:
    c = np.linalg.inv(a)
    ct = np.conjugate(np.swapaxes(c, -2, -1))
    if upper:
        return np.matmul(c, ct)
    return np.matmul(ct, c)


@register_numpy_types(linalg.inv)
def _(a: Union[np.ndarray, np.generic]) -> np.ndarray:
    return np.linalg.inv(a)


@register_numpy_types(linalg.pinv)
def _(a: Union[np.ndarray, np.generic]) -> np.ndarray:
    return np.linalg.pinv(a)


@register_numpy_types(linalg.lstsq)
def _(a: Union[np.ndarray, np.generic], b: Union[np.ndarray, np.generic], driver: Optional[str] = None) -> np.ndarray:
    return lstsq(a, b, lapack_driver=driver)[0]


@register_numpy_types(linalg.svd)
def _(a: Union[np.ndarray, np.generic], full_matrices: Optional[bool] = True) -> np.ndarray:
    return np.linalg.svd(a, compute_uv=True, full_matrices=full_matrices)
