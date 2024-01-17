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

import torch

from nncf.experimental.tensor.functions import linalg


@linalg.norm.register(torch.Tensor)
def _(
    a: torch.Tensor,
    ord: Optional[Union[str, float, int]] = None,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> torch.Tensor:
    return torch.linalg.norm(a, ord=ord, dim=axis, keepdims=keepdims)
