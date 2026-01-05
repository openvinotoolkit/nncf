# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Literal, Optional, Union

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import lstsq

from nncf.tensor.definitions import T_AXIS
from nncf.tensor.functions import linalg

T_NUMPY_ARRAY = NDArray[Any]


@linalg.norm.register
def _(
    a: T_NUMPY_ARRAY,
    ord: Union[Literal["fro", "nuc"], float, None] = None,
    axis: T_AXIS = None,
    keepdims: bool = False,
) -> T_NUMPY_ARRAY:
    return np.array(np.linalg.norm(a, ord=ord, axis=axis, keepdims=keepdims))


@linalg.cholesky.register
def _(a: T_NUMPY_ARRAY, upper: bool = False) -> T_NUMPY_ARRAY:
    lt = np.linalg.cholesky(a)
    if upper:
        return np.conjugate(np.swapaxes(lt, -2, -1))
    return lt


@linalg.cholesky_inverse.register
def _(a: T_NUMPY_ARRAY, upper: bool = False) -> T_NUMPY_ARRAY:
    c = np.linalg.inv(a)
    ct = np.conjugate(np.swapaxes(c, -2, -1))
    if upper:
        return np.matmul(c, ct)
    return np.matmul(ct, c)


@linalg.inv.register
def _(a: T_NUMPY_ARRAY) -> T_NUMPY_ARRAY:
    return np.linalg.inv(a)


@linalg.pinv.register
def _(a: T_NUMPY_ARRAY) -> T_NUMPY_ARRAY:
    return np.linalg.pinv(a)


@linalg.lstsq.register
def _(a: T_NUMPY_ARRAY, b: T_NUMPY_ARRAY, driver: Optional[str] = None) -> T_NUMPY_ARRAY:
    return lstsq(a, b, lapack_driver=driver)[0]


@linalg.svd.register
def _(a: T_NUMPY_ARRAY, full_matrices: Optional[bool] = True) -> T_NUMPY_ARRAY:
    return np.linalg.svd(a, compute_uv=True, full_matrices=full_matrices)  # type: ignore[call-overload]
