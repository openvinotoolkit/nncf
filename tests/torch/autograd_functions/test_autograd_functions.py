import math

import pytest
import torch

from nncf.torch.functions import STRound
from nncf.torch.functions import STThreshold


@pytest.mark.parametrize("use_cuda", [True, False])
@pytest.mark.parametrize("requires_grad", [True, False])
class TestAutogradFunction:
    @pytest.mark.parametrize(("input_", "ref_output"), [
        (torch.tensor([[1.2, -3.4], [5.6, 7.89]]), torch.tensor([[1., -3.], [6., 8.]])),
        (torch.tensor([[[1.5]]]), torch.tensor([[[2.]]])),
        (torch.tensor([2.5]), torch.tensor([2.])),
        (torch.tensor(4.2), torch.tensor(4.)),  # scalar tensor
        (torch.tensor([math.inf, 1.1, -math.inf]), torch.tensor([math.inf, 1., -math.inf])),
        (torch.tensor([math.nan, 2.1]), torch.tensor([math.nan, 2.]))
    ])
    def test_STRound(self, input_: torch.Tensor, ref_output: torch.Tensor,
                     use_cuda: bool, requires_grad: bool):
        if not torch.cuda.is_available() and use_cuda is True:
            pytest.skip("Skipping CUDA test cases for CPU only setups")
        device = torch.device('cuda' if use_cuda else 'cpu')
        input_tensor = input_.clone().to(device).requires_grad_(requires_grad)
        output_tensor = STRound.apply(input_tensor)
        ref_output_tensor = ref_output.clone().to(device)
        assert output_tensor.device == input_tensor.device
        assert output_tensor.requires_grad is requires_grad
        assert torch.allclose(output_tensor, ref_output_tensor, equal_nan=True)
        if requires_grad:
            assert output_tensor.grad_fn.name().startswith('STRoundBackward')
            output_tensor.sum().backward()
            ref_grad_tensor = torch.ones_like(input_tensor)
            assert torch.allclose(input_tensor.grad, ref_grad_tensor)

    @pytest.mark.parametrize(("input_", "threshold", "ref_output"), [
        (torch.tensor([[1.2, -3.4], [5.6, 7.89]]), 4.0, torch.tensor([[0., 0.], [1., 1.]])),
        (torch.tensor([[[1.5]]]), 1., torch.tensor([[[1.]]])),
        (torch.tensor([2.5]), -10., torch.tensor([1.])),
        (torch.tensor(4.2), 4., torch.tensor(1.)),  # scalar tensor
        (torch.tensor([math.inf, 1.1, -math.inf]), 2., torch.tensor([1., 0., 0.])),
        (torch.tensor([math.nan, 2.1]), 2., torch.tensor([0., 1.]))
    ])
    def test_STThreshold(self, input_: torch.Tensor, ref_output: torch.Tensor,
                         threshold: float, use_cuda: bool, requires_grad: bool):
        if not torch.cuda.is_available() and use_cuda is True:
            pytest.skip("Skipping CUDA test cases for CPU only setups")
        device = torch.device('cuda' if use_cuda else 'cpu')
        input_tensor = input_.clone().to(device).requires_grad_(requires_grad)
        output_tensor = STThreshold.apply(input_tensor, threshold)
        ref_output_tensor = ref_output.clone().to(device)
        assert output_tensor.device == input_tensor.device
        assert output_tensor.requires_grad is requires_grad
        assert torch.allclose(output_tensor, ref_output_tensor)
        if requires_grad:
            assert output_tensor.grad_fn.name().startswith('STThresholdBackward')
            output_tensor.sum().backward()
            ref_grad_tensor = torch.ones_like(input_tensor)
            assert torch.allclose(input_tensor.grad, ref_grad_tensor)
