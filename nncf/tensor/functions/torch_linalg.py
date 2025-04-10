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
from typing import Literal, Optional, Union

import torch

from nncf.tensor.definitions import T_AXIS
from nncf.tensor.functions import linalg


@linalg.norm.register
def _(
    a: torch.Tensor,
    ord: Union[Literal["fro", "nuc"], float, None] = None,
    axis: T_AXIS = None,
    keepdims: bool = False,
) -> torch.Tensor:
    return torch.linalg.norm(a, ord=ord, dim=axis, keepdims=keepdims)


@linalg.cholesky.register
def _(a: torch.Tensor, upper: bool = False) -> torch.Tensor:
    return torch.linalg.cholesky(a, upper=upper)


@linalg.cholesky_inverse.register
def _(a: torch.Tensor, upper: bool = False) -> torch.Tensor:
    return torch.cholesky_inverse(a, upper=upper)


@linalg.inv.register
def _(a: torch.Tensor) -> torch.Tensor:
    return torch.linalg.inv(a)


@linalg.pinv.register
def _(a: torch.Tensor) -> torch.Tensor:
    # Consider using torch.linalg.lstsq() if possible for multiplying a matrix on the left by the pseudo-inverse, as:
    # torch.linalg.lstsq(A, B).solution == A.pinv() @ B
    # It is always preferred to use lstsq() when possible, as it is faster and more numerically stable than computing
    # the pseudo-inverse explicitly.
    return torch.linalg.pinv(a)


@linalg.lstsq.register
def _(a: torch.Tensor, b: torch.Tensor, driver: Optional[str] = None) -> torch.Tensor:
    return torch.linalg.lstsq(a, b, driver=driver).solution


@linalg.svd.register
def _(a: torch.Tensor, full_matrices: Optional[bool] = True) -> torch.Tensor:
    return torch.linalg.svd(a, full_matrices=full_matrices)
