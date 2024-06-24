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

from typing import Any, List, Optional, Tuple

import numpy as np
import pytest

from nncf.common.graph.layer_attributes import Dtype
from nncf.onnx.statistics.collectors import ONNXAbsMaxReducer
from nncf.onnx.statistics.collectors import ONNXAbsQuantileReducer
from nncf.onnx.statistics.collectors import ONNXBatchMeanReducer
from nncf.onnx.statistics.collectors import ONNXMaxReducer
from nncf.onnx.statistics.collectors import ONNXMeanPerChanelReducer
from nncf.onnx.statistics.collectors import ONNXMeanReducer
from nncf.onnx.statistics.collectors import ONNXMinReducer
from nncf.onnx.statistics.collectors import ONNXQuantileReducer
from nncf.tensor import Tensor
from tests.common.experimental.test_reducers_and_aggregators import TemplateTestReducersAggregators


class TestReducersAggregators(TemplateTestReducersAggregators):

    def get_nncf_tensor(self, x: np.array, dtype: Optional[Dtype] = None):
        if dtype is Dtype.INTEGER:
            x = x.astype(np.int64)
        if dtype is Dtype.FLOAT:
            x = x.astype(np.float32)
        return Tensor(x)

    @pytest.fixture(scope="module")
    def reducers(self):
        return {
            "min": ONNXMinReducer,
            "max": ONNXMaxReducer,
            "abs_max": ONNXAbsMaxReducer,
            "mean": ONNXMeanReducer,
            "quantile": ONNXQuantileReducer,
            "abs_quantile": ONNXAbsQuantileReducer,
            "batch_mean": ONNXBatchMeanReducer,
            "mean_per_ch": ONNXMeanPerChanelReducer,
        }

    def all_close(self, val, ref) -> bool:
        val_ = np.array(val)
        ref_ = np.array(ref)
        return np.allclose(val_, ref_) and val_.shape == ref_.shape

    def squeeze_tensor(self, ref_tensor: List[Any], axes: Optional[Tuple[int]] = None):
        return np.squeeze(np.array(ref_tensor), axes)

    def cast_tensor(self, tensor, dtype: Dtype):
        return tensor
