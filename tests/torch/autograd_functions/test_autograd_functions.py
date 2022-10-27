import math

import pytest
import torch
from nncf.torch.functions import STRound, STThreshold


@pytest.mark.parametrize("use_cuda", [True, False])
@pytest.mark.parametrize("requires_grad", [True, False])
class TestAutogradFunction:
    @staticmethod
    def get_device(use_cuda):
        return torch.device('cuda' if use_cuda else 'cpu')

    @pytest.mark.parametrize(("input_", "ref_output"), [
        ([[1.2, -3.4], [5.6, 7.89]], [[1., -3.], [6., 8.]]),
        ([[[1.5]]], [[[2.]]]),
        ([2.5], [2.]),
        (4.2, 4.),  # scalar tensor
        ([math.inf, 1.1, -math.inf], [math.inf, 1., -math.inf]),
        ([math.nan, 2.1], [math.nan, 2.])
    ])
    def test_STRound(self, input_, ref_output, use_cuda, requires_grad):
        if not torch.cuda.is_available() and use_cuda is True:
            pytest.skip("Skipping CUDA test cases for CPU only setups")
        device = self.get_device(use_cuda)
        input_tensor = torch.tensor(input_, device=device, requires_grad=requires_grad)
        output_tensor = STRound.apply(input_tensor)
        ref_output_tensor = torch.tensor(ref_output, device=device)
        assert output_tensor.device == input_tensor.device
        assert output_tensor.requires_grad is requires_grad
        assert torch.allclose(output_tensor, ref_output_tensor, equal_nan=True)
        if requires_grad:
            assert output_tensor.grad_fn.name().startswith('STRoundBackward')
            output_tensor.sum().backward()
            ref_grad_tensor = torch.ones_like(input_tensor)
            assert torch.allclose(input_tensor.grad, ref_grad_tensor)

    @pytest.mark.parametrize(("input_", "threshold", "ref_output"), [
        ([[1.2, -3.4], [5.6, 7.89]], 4.0, [[0., 0.], [1., 1.]]),
        ([[[1.5]]], 1., [[[1.]]]),
        ([2.5], -10., [1.]),
        (4.2, 4., 1.),  # scalar tensor
        ([math.inf, 1.1, -math.inf], 2., [1., 0., 0.]),
        ([math.nan, 2.1], 2., [0., 1.])
    ])
    def test_STThreshold(self, input_, threshold, ref_output, use_cuda, requires_grad):
        if not torch.cuda.is_available() and use_cuda is True:
            pytest.skip("Skipping CUDA test cases for CPU only setups")
        device = self.get_device(use_cuda)
        input_tensor = torch.tensor(input_, device=device, requires_grad=requires_grad)
        output_tensor = STThreshold.apply(input_tensor, threshold)
        ref_output_tensor = torch.tensor(ref_output, device=device)
        assert output_tensor.device == input_tensor.device
        assert output_tensor.requires_grad is requires_grad
        assert torch.allclose(output_tensor, ref_output_tensor)
        if requires_grad:
            assert output_tensor.grad_fn.name().startswith('STThresholdBackward')
            output_tensor.sum().backward()
            ref_grad_tensor = torch.ones_like(input_tensor)
            assert torch.allclose(input_tensor.grad, ref_grad_tensor)
