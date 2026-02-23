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

from typing import Any

import numpy as np
import pytest

from nncf.common.graph.layer_attributes import Dtype
from nncf.common.tensor_statistics.collectors import AbsMaxReducer
from nncf.common.tensor_statistics.collectors import AbsQuantileReducer
from nncf.common.tensor_statistics.collectors import BatchMeanReducer
from nncf.common.tensor_statistics.collectors import MaxReducer
from nncf.common.tensor_statistics.collectors import MaxVarianceReducer
from nncf.common.tensor_statistics.collectors import MeanPerChReducer
from nncf.common.tensor_statistics.collectors import MeanReducer
from nncf.common.tensor_statistics.collectors import MeanVarianceReducer
from nncf.common.tensor_statistics.collectors import MinReducer
from nncf.common.tensor_statistics.collectors import QuantileReducer
from nncf.tensor import Tensor
from tests.common.test_reducers_and_aggregators import TemplateTestReducersAggregators


class TestReducersAggregators(TemplateTestReducersAggregators):
    def get_nncf_tensor(self, x: np.array, dtype: Dtype | None = None):
        if dtype is Dtype.INTEGER:
            x = x.astype(np.int64)
        if dtype is Dtype.FLOAT:
            x = x.astype(np.float32)
        return Tensor(x)

    @pytest.fixture(scope="module")
    def reducers(self):
        return {
            "min": MinReducer,
            "max": MaxReducer,
            "abs_max": AbsMaxReducer,
            "mean": MeanReducer,
            "mean_variance": MeanVarianceReducer,
            "max_variance": MaxVarianceReducer,
            "quantile": QuantileReducer,
            "abs_quantile": AbsQuantileReducer,
            "batch_mean": BatchMeanReducer,
            "mean_per_ch": MeanPerChReducer,
        }

    def all_close(self, val, ref) -> bool:
        val_ = np.array(val)
        ref_ = np.array(ref)
        return np.allclose(val_, ref_) and val_.shape == ref_.shape

    def squeeze_tensor(self, ref_tensor: list[Any], axes: tuple[int] | None = None):
        return np.squeeze(np.array(ref_tensor), axes)

    def cast_tensor(self, tensor, dtype: Dtype):
        return tensor
