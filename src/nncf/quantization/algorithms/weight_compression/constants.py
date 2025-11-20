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

import numpy as np

from nncf.tensor import TensorDataType

NF4_QUANTILES = np.array(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ],
    dtype=np.float32,
)

F4E2M1_QUANTILES = np.array(
    [
        -6.0,
        -4.0,
        -3.0,
        -2.0,
        -1.5,
        -1.0,
        -0.5,
        -0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
    ],
    dtype=np.float32,
)


CB4_QUANTILES = np.array(
    [
        -3.5,
        -2.5,
        -1.875,
        -1.375,
        -1.0,
        -0.625,
        -0.3125,
        0.0,
        0.28125,
        0.5625,
        0.875,
        1.125,
        1.5,
        2.0,
        2.5,
        3.5,
    ],
    dtype=np.float32,
)


CENTER_OF_NF4_QUANTILES = np.array(
    [
        -0.84809643,
        -0.6106329,
        -0.45999527,
        -0.33967942,
        -0.2346074,
        -0.13791174,
        -0.045525018,
        0.03979015,
        0.120255254,
        0.20352125,
        0.29201376,
        0.38931254,
        0.5016634,
        0.6427869,
        0.8614784,
    ],
    dtype=np.float32,
)


CENTER_OF_F4E2M1_QUANTILES = (F4E2M1_QUANTILES[1:] + F4E2M1_QUANTILES[:-1]) / 2


FP_MAX_VALUES = {
    TensorDataType.nf4: 1.0,
    TensorDataType.f4e2m1: 6.0,
    TensorDataType.f8e4m3: 448.0,
}
