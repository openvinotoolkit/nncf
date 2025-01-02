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

from abc import ABC
from typing import Any, List, Optional, Tuple

import numpy as np
import pytest
import torch

import nncf
from nncf.common.graph.layer_attributes import Dtype
from nncf.experimental.common.tensor_statistics.collectors import AbsMaxReducer
from nncf.experimental.common.tensor_statistics.collectors import AbsQuantileReducer
from nncf.experimental.common.tensor_statistics.collectors import BatchMeanReducer
from nncf.experimental.common.tensor_statistics.collectors import MaxReducer
from nncf.experimental.common.tensor_statistics.collectors import MeanPerChReducer
from nncf.experimental.common.tensor_statistics.collectors import MeanReducer
from nncf.experimental.common.tensor_statistics.collectors import MinReducer
from nncf.experimental.common.tensor_statistics.collectors import QuantileReducer
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.tensor import Tensor
from nncf.tensor import functions as fns
from nncf.torch.tensor_statistics.algo import create_register_input_hook
from tests.common.experimental.test_reducers_and_aggregators import TemplateTestReducersAggregators


class BaseTestReducersAggregators(TemplateTestReducersAggregators, ABC):

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
            "min": MinReducer,
            "max": MaxReducer,
            "abs_max": AbsMaxReducer,
            "mean": MeanReducer,
            "quantile": QuantileReducer,
            "abs_quantile": AbsQuantileReducer,
            "batch_mean": BatchMeanReducer,
            "mean_per_ch": MeanPerChReducer,
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
        raise nncf.ValidationError(f"Invalid dtype: {dtype}. Supported dtypes are {Dtype.FLOAT} and {Dtype.INTEGER}")


class TestCPUReducersAggregators(BaseTestReducersAggregators):
    def get_nncf_tensor(self, x: np.array, dtype: Optional[Dtype] = None):
        return Tensor(self._get_torch_tensor(x, dtype=dtype).cpu())

    def all_close(self, val: torch.Tensor, ref) -> bool:
        assert not val.is_cuda
        return super().all_close(val, ref)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Cuda is not available in current environment")
class TestCudaReducersAggregators(BaseTestReducersAggregators):
    def get_nncf_tensor(self, x: np.array, dtype: Optional[Dtype] = None):
        return Tensor(self._get_torch_tensor(x, dtype=dtype).cuda())

    def all_close(self, val: torch.Tensor, ref) -> bool:
        assert val.is_cuda
        return super().all_close(val, ref)


@pytest.mark.parametrize("size,ref", [(16_000_000, 1_600_000.8), (17_000_000, 1_700_000.8)])
def test_quantile_percentile_function(use_cuda, size, ref):
    if use_cuda and not torch.cuda.is_available():
        pytest.skip("Cuda is not available in current environment")
    device = "cuda" if use_cuda else "cpu"
    tensor = Tensor(torch.arange(1, size, 1).float().to(device))
    res_quantile = fns.quantile(tensor, [0.1], axis=0)
    res_percentile = fns.percentile(tensor, [10], axis=0)
    assert res_quantile.shape[0] == res_quantile.shape[0] == 1
    for tensor in [res_quantile[0].data, res_percentile[0].data]:
        assert tensor == ref
        assert tensor.is_cuda == (device == "cuda")


@pytest.mark.parametrize("size,ref", [(16_000_000, 8_000_000), (17_000_000, 8_500_000)])
def test_median_function(use_cuda, size, ref):
    if use_cuda and not torch.cuda.is_available():
        pytest.skip("Cuda is not available in current environment")
    device = "cuda" if use_cuda else "cpu"
    tensor = Tensor(torch.arange(1, size, 1).float().to(device))
    res = fns.median(tensor, axis=0)
    assert res.data == ref
    assert res.data.is_cuda == (device == "cuda")


def test_create_register_input_hook_with_return_type(mocker):
    collector = TensorCollector()
    collector.register_input_for_all_reducers = mocker.MagicMock()
    hook = create_register_input_hook(collector)
    input_ = torch.return_types.max([torch.tensor((1,))] * 2)
    output_ = hook(input_)
    assert input_ is output_
    mocker = collector.register_input_for_all_reducers
    mocker.assert_called_once()
    attr = mocker.call_args_list[0][0][0]
    assert isinstance(attr, Tensor)
    assert attr.data == torch.tensor(1)
