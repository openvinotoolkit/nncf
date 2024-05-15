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

from nncf.experimental.tensor.functions import linalg


@linalg.norm.register
def _(
    a: Union[np.ndarray, np.generic],
    ord: Optional[Union[str, float, int]] = None,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> np.ndarray:
    return np.array(np.linalg.norm(a, ord=ord, axis=axis, keepdims=keepdims))


@linalg.cholesky.register
def _(a: Union[np.ndarray, np.generic], upper: bool = False) -> np.ndarray:
    lt = np.linalg.cholesky(a)
    if upper:
        return np.conjugate(np.swapaxes(lt, -2, -1))
    return lt


@linalg.cholesky_inverse.register
def _(a: Union[np.ndarray, np.generic], upper: bool = False) -> np.ndarray:
    c = np.linalg.inv(a)
    ct = np.conjugate(np.swapaxes(c, -2, -1))
    if upper:
        return np.matmul(c, ct)
    return np.matmul(ct, c)


@linalg.inv.register
def _(a: Union[np.ndarray, np.generic]) -> np.ndarray:
    return np.linalg.inv(a)
