"""
 Copyright (c) 2022 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from dataclasses import dataclass
import math

import pytest
import torch

from nncf.torch.functions import STRound
from nncf.torch.functions import STThreshold


@dataclass
class STRoundTestCase:
    input_tensor: torch.Tensor
    ref_output_tensor: torch.Tensor


@dataclass
class STThresholdTestCase:
    input_tensor: torch.Tensor
    threshold: float
    ref_output_tensor: torch.Tensor


@pytest.mark.parametrize("use_cuda", [True, False])
@pytest.mark.parametrize("requires_grad", [True, False])
class TestAutogradFunction:
    @pytest.fixture(autouse=True)
    def check_cuda(self, use_cuda: bool):
        if use_cuda and (not torch.cuda.is_available()):
            pytest.skip("Skipping CUDA test cases for CPU only setups.")

    @pytest.mark.parametrize('test_case', [
        STRoundTestCase(
            input_tensor=torch.tensor([[1.2, -3.4], [5.6, 7.89]]),
            ref_output_tensor=torch.tensor([[1., -3.], [6., 8.]])
        ),
        STRoundTestCase(
            input_tensor=torch.tensor([[[1.5]]]),
            ref_output_tensor=torch.tensor([[[2.]]])
        ),
        STRoundTestCase(
            input_tensor=torch.tensor([2.5]),
            ref_output_tensor=torch.tensor([2.])
        ),
        STRoundTestCase(
            input_tensor=torch.tensor(4.2),  # scalar tensor
            ref_output_tensor=torch.tensor(4.)
        ),
        STRoundTestCase(
            input_tensor=torch.tensor([math.inf, 1.1, -math.inf]),
            ref_output_tensor=torch.tensor([math.inf, 1., -math.inf])
        ),
        STRoundTestCase(
            input_tensor=torch.tensor([math.nan, 2.1]),
            ref_output_tensor=torch.tensor([math.nan, 2.])
        )
    ])
    def test_STRound(self, test_case: STRoundTestCase, use_cuda: bool, requires_grad: bool):
        device = torch.device('cuda' if use_cuda else 'cpu')
        input_tensor = test_case.input_tensor.clone().to(device).requires_grad_(requires_grad)
        output_tensor = STRound.apply(input_tensor)
        ref_output_tensor = test_case.ref_output_tensor.clone().to(device)
        assert output_tensor.device == input_tensor.device
        assert output_tensor.requires_grad is requires_grad
        assert torch.allclose(output_tensor, ref_output_tensor, equal_nan=True)
        if requires_grad:
            assert output_tensor.grad_fn.name().startswith('STRoundBackward')
            output_tensor.sum().backward()
            ref_grad_tensor = torch.ones_like(input_tensor)
            assert torch.allclose(input_tensor.grad, ref_grad_tensor)

    @pytest.mark.parametrize('test_case', [
        STThresholdTestCase(
            input_tensor=torch.tensor([[1.2, -3.4], [5.6, 7.89]]),
            threshold=4.0,
            ref_output_tensor=torch.tensor([[0., 0.], [1., 1.]])
        ),
        STThresholdTestCase(
            input_tensor=torch.tensor([[[1.5]]]),
            threshold=1.,
            ref_output_tensor=torch.tensor([[[1.]]])
        ),
        STThresholdTestCase(
            input_tensor=torch.tensor([2.5]),
            threshold=-10.,
            ref_output_tensor=torch.tensor([1.])
        ),
        STThresholdTestCase(
            input_tensor=torch.tensor(4.2),  # scalar tensor
            threshold=4.,
            ref_output_tensor=torch.tensor(1.)
        ),
        STThresholdTestCase(
            input_tensor=torch.tensor([math.inf, 1.1, -math.inf]),
            threshold=2.,
            ref_output_tensor=torch.tensor([1., 0., 0.])
        ),
        STThresholdTestCase(
            input_tensor=torch.tensor([math.nan, 2.1]),
            threshold=2.,
            ref_output_tensor=torch.tensor([0., 1.])
        )
    ])
    def test_STThreshold(self, test_case: STThresholdTestCase, use_cuda: bool, requires_grad: bool):
        device = torch.device('cuda' if use_cuda else 'cpu')
        input_tensor = test_case.input_tensor.clone().to(device).requires_grad_(requires_grad)
        output_tensor = STThreshold.apply(input_tensor, test_case.threshold)
        ref_output_tensor = test_case.ref_output_tensor.clone().to(device)
        assert output_tensor.device == input_tensor.device
        assert output_tensor.requires_grad is requires_grad
        assert torch.allclose(output_tensor, ref_output_tensor)
        if requires_grad:
            assert output_tensor.grad_fn.name().startswith('STThresholdBackward')
            output_tensor.sum().backward()
            ref_grad_tensor = torch.ones_like(input_tensor)
            assert torch.allclose(input_tensor.grad, ref_grad_tensor)
