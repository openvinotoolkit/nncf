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
import warnings

import numpy as np
import pytest

from nncf.quantization.fake_quantize import tune_range
from nncf.tensor import Tensor


@pytest.mark.parametrize(
    "params",
    (
        (Tensor(np.array([0.0])), Tensor(np.array([1.0])), 8, False),
        (Tensor(np.array([-1.0])), Tensor(np.array([0.0])), 8, False),
    ),
)
def test_tune_range_zero_division_warning(params):
    with warnings.catch_warnings(record=True) as w:
        # Calling tune_range should not raise a warning
        tune_range(*params)
        assert len(w) == 0
