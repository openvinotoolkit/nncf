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

from abc import ABC
from typing import Any, List, Optional, Tuple

import numpy as np
import pytest
import torch

from nncf.common.graph.layer_attributes import Dtype
from nncf.experimental.common.tensor_statistics.collectors import AbsMaxReducer
from nncf.experimental.common.tensor_statistics.collectors import AbsQuantileReducer
from nncf.experimental.common.tensor_statistics.collectors import BatchMeanReducer
from nncf.experimental.common.tensor_statistics.collectors import MaxReducer
from nncf.experimental.common.tensor_statistics.collectors import MeanPerChReducer
from nncf.experimental.common.tensor_statistics.collectors import MeanReducer
from nncf.experimental.common.tensor_statistics.collectors import MinReducer
from nncf.experimental.common.tensor_statistics.collectors import NoopReducer
from nncf.experimental.common.tensor_statistics.collectors import QuantileReducer
from nncf.experimental.tensor import Tensor
from nncf.experimental.tensor import functions as fns
from nncf.torch.tensor_statistics.collectors import PTNoopReducer, PTMinReducer, PTMaxReducer, PTAbsMaxReducer, \
    PTMeanReducer, PTQuantileReducer, PTAbsQuantileReducer, PTBatchMeanReducer, PTMeanPerChanelReducer
from tests.common.experimental.test_reducers_and_aggregators import TemplateTestReducersAggreagtors


class BaseTestReducersAggregators(TemplateTestReducersAggreagtors, ABC):
    def _get_torch_tensor(self, x: np.ndarray, dtype: Optional[Dtype] = None):
        torch_tensor = torch.tensor(x)
        if dtype == Dtype.FLOAT:
            torch_tensor = torch_tensor.float()
        elif dtype == Dtype.INTEGER:
            torch_tensor = torch_tensor.int()
        return torch_tensor

    @pytest.fixture(scope="module")
    def reducers(self):
        return {
            "noop": PTNoopReducer,
            "min": PTMinReducer,
            "max": PTMaxReducer,
            "abs_max": PTAbsMaxReducer,
            "mean": PTMeanReducer,
            "quantile": PTQuantileReducer,
            "abs_quantile": PTAbsQuantileReducer,
            "batch_mean": PTBatchMeanReducer,
            "mean_per_ch": PTMeanPerChanelReducer,
        }

    def all_close(self, val, ref) -> bool:
        val_ = val
        ref_ = torch.tensor(ref).to(val_.device)
        return torch.allclose(val_, ref_) and val_.shape == ref_.shape

    def squeeze_tensor(self, ref_tensor: List[Any], axes: Optional[Tuple[int]] = None):
        if axes is None:
            return torch.tensor(ref_tensor).squeeze()
        return torch.tensor(ref_tensor).squeeze(axes)

    def cast_tensor(self, tensor, dtype: Dtype):
        tensor = torch.tensor(tensor)
        if dtype == Dtype.FLOAT:
            return tensor.float()
        if dtype == Dtype.INTEGER:
            return tensor.int()
        raise RuntimeError()

    def expand_dims(self, tensor, dims: Tuple[int, ...]):
        tensor_ = torch.tensor(tensor)
        shape = list(tensor_.shape)
        for dim in dims:
            shape.insert(dim, 1)
        return tensor_.view(shape)


class TestCPUReducersAggregators(BaseTestReducersAggregators):
    def get_nncf_tensor(self, x: np.array, dtype: Optional[Dtype] = None):
        return Tensor(self._get_torch_tensor(x, dtype=dtype).cpu())

    def all_close(self, val: torch.Tensor, ref) -> bool:
        assert not val.is_cuda
        return super().all_close(val, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Cuda is not available in current environment")
class TestCudaReducersAggregators(BaseTestReducersAggregators):
    def get_nncf_tensor(self, x: np.array, dtype: Optional[Dtype] = None):
        return Tensor(self._get_torch_tensor(x, dtype=dtype).cuda())

    def all_close(self, val: torch.Tensor, ref) -> bool:
        assert val.is_cuda
        return super().all_close(val, ref)


@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.parametrize("size,ref", [(16_000_000, 1_600_000.8750), (17_000_000, 1_700_000.7500)])
def test_quantile_function(device, size, ref):
    if not torch.cuda.is_available() and device == "cuda":
        pytest.skip("Cuda is not available in current environment")
    tensor = Tensor(torch.arange(1, size, 1).float().to(device))
    res_quantile = fns.quantile(tensor, [0.1], axis=0)
    assert len(res_quantile) == 1
    assert res_quantile[0].data == ref
    assert res_quantile[0].data.is_cuda == (device == "cuda")


@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.parametrize("size,ref", [(16_000_000, 8_000_000), (17_000_000, 8_500_000)])
def test_median_function(device, size, ref):
    if not torch.cuda.is_available() and device == "cuda":
        pytest.skip("Cuda is not available in current environment")
    tensor = Tensor(torch.arange(1, size, 1).float().to(device))
    res = fns.median(tensor, axis=0)
    assert res.data == ref
    assert res.data.is_cuda == (device == "cuda")
