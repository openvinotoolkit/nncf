# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from nncf.experimental.tensor import Tensor
from nncf.experimental.tensor import functions as fns


def mean_per_channel(x: Tensor, axis: int) -> Tensor:
    """
    Computes the mean of elements across given channel dimension of Tensor.

    :param x: Tensor to reduce.
    :param axis: The channel dimensions to reduce.
    :return: Reduced Tensor.
    """
    if len(x.shape) < 3:
        return fns.mean(x, axis=0)
    x = fns.moveaxis(x, axis, 1)
    t = x.reshape([x.shape[0], x.shape[1], -1])
    return fns.mean(t, axis=(0, 2))
