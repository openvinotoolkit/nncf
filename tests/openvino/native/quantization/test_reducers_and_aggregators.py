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

from typing import Any, List, Optional, Tuple

import numpy as np
import openvino as ov
import openvino.runtime.opset13 as opset
import pytest

from nncf.common.graph.layer_attributes import Dtype
from nncf.common.utils.os import is_macos
from nncf.openvino.statistics.collectors import OVAbsMaxReducer
from nncf.openvino.statistics.collectors import OVAbsQuantileReducer
from nncf.openvino.statistics.collectors import OVBatchMeanReducer
from nncf.openvino.statistics.collectors import OVMaxReducer
from nncf.openvino.statistics.collectors import OVMaxVarianceReducer
from nncf.openvino.statistics.collectors import OVMeanAbsMaxReducer
from nncf.openvino.statistics.collectors import OVMeanPerChanelReducer
from nncf.openvino.statistics.collectors import OVMeanReducer
from nncf.openvino.statistics.collectors import OVMeanVarianceReducer
from nncf.openvino.statistics.collectors import OVMinReducer
from nncf.openvino.statistics.collectors import OVQuantileReducer
from nncf.openvino.statistics.collectors import OVShapeReducer
from nncf.tensor import Tensor
from tests.common.experimental.test_reducers_and_aggregators import TemplateTestReducersAggregators


class TestReducersAggregators(TemplateTestReducersAggregators):
    MIXED_PRECISION_REDUCERS_REF_VALUES = [
        (OVMeanVarianceReducer, (0, 1), np.array([695.375])),
        (OVMeanVarianceReducer, None, np.array([707.1875])),
        (OVMaxVarianceReducer, (0, 1), np.array([710.25])),
        (OVMaxVarianceReducer, None, np.array([707.1875])),
        (OVMeanAbsMaxReducer, (0, 1), np.array([87.0])),
        (OVMeanAbsMaxReducer, None, np.array([94.0])),
    ]

    def get_nncf_tensor(self, x: np.array, dtype: Optional[Dtype] = None):
        if dtype is Dtype.INTEGER:
            x = x.astype(np.int64)
        if dtype is Dtype.FLOAT:
            x = x.astype(np.float32)
        return Tensor(x)

    @pytest.fixture(scope="module")
    def reducers(self):
        return {
            "min": OVMinReducer,
            "max": OVMaxReducer,
            "abs_max": OVAbsMaxReducer,
            "mean": OVMeanReducer,
            "quantile": OVQuantileReducer,
            "abs_quantile": OVAbsQuantileReducer,
            "batch_mean": OVBatchMeanReducer,
            "mean_per_ch": OVMeanPerChanelReducer,
        }

    def all_close(self, val, ref) -> bool:
        val_ = np.array(val)
        ref_ = np.array(ref)
        return np.allclose(val_, ref_) and val_.shape == ref_.shape

    def squeeze_tensor(self, ref_tensor: List[Any], axes: Optional[Tuple[int]] = None):
        return np.squeeze(np.array(ref_tensor), axes)

    def cast_tensor(self, tensor, dtype: Dtype):
        return tensor

    @pytest.mark.parametrize("reducer_cls,reduction_axes,ref_value", MIXED_PRECISION_REDUCERS_REF_VALUES)
    def test_mixed_precision_reducers(self, reducer_cls, reduction_axes, ref_value):
        input_ = np.arange(2 * 4 * 8).reshape(2, 4, 8)
        input_[:, :2] *= 2

        reducer = reducer_cls(reduction_axes=reduction_axes, inplace=True)
        inplace_fn = reducer.get_inplace_fn()

        ov_model_input = opset.parameter(input_.shape)
        ov_model_output = inplace_fn(ov_model_input, 0, "reducer_output")
        ov_model = ov.Model([ov_model_output], [ov_model_input])
        compiled_ov_model = ov.compile_model(ov_model)

        reducer_output = compiled_ov_model(input_)[0]

        atol = 1.0e-8 if not is_macos() else 0.3
        assert np.allclose(reducer_output, ref_value, atol=atol)

    @pytest.mark.parametrize(
        "input_", [np.arange(2), np.arange(2 * 4 * 8).reshape(8, 8), np.arange(2 * 4 * 8).reshape(2, 4, 8)]
    )
    def test_inplace_shape_reducer(self, input_):
        reducer = OVShapeReducer()
        inplace_fn = reducer.get_inplace_fn()

        ov_model_input = opset.parameter(input_.shape)
        ov_model_output = inplace_fn(ov_model_input, 0, "reducer_output")
        ov_model = ov.Model([ov_model_output], [ov_model_input])
        compiled_ov_model = ov.compile_model(ov_model)

        reducer_output = compiled_ov_model(input_)[0]
        assert all([it[0] == it[1] for it in zip(input_.shape, reducer_output)])
