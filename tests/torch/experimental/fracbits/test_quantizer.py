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

import pytest
import torch
from torch import nn

from nncf.common.utils.logger import logger as nncf_logger
from nncf.experimental.torch.fracbits.quantizer import FracBitsAsymmetricQuantizer, FracBitsSymmetricQuantizer
from nncf.experimental.torch.fracbits.structs import FracBitsQuantizationMode
from nncf.torch.quantization.layers import PTQuantizerSpec

#pylint: disable=redefined-outer-name


def set_manual_seed():
    torch.manual_seed(3003)


@pytest.fixture(scope="function")
def linear_problem(num_bits: int = 4, sigma: float = 0.2):
    set_manual_seed()

    levels = 2 ** num_bits
    w = 1 / levels * (torch.randint(0, levels, size=[100, 10]) - levels // 2)
    x = torch.randn([1000, 10])
    y = w.mm(x.t())
    y += sigma * torch.randn_like(y)

    return w, x, y, num_bits, sigma


@pytest.fixture()
def qspec(request):
    return PTQuantizerSpec(num_bits=8,
                           mode=request.param,
                           signedness_to_force=None,
                           scale_shape=(1, 1),
                           narrow_range=False,
                           half_range=False,
                           logarithm_scale=False)


@pytest.mark.parametrize("add_bitwidth_loss", [True, False])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("qspec",
                         [FracBitsQuantizationMode.ASYMMETRIC, FracBitsQuantizationMode.SYMMETRIC], indirect=["qspec"])
def test_quantization(linear_problem, qspec, device, add_bitwidth_loss):
    """
    Test quantization for the simple linear problem.
    The weight is filled with the random integer in range with [-2 ** (bit_width - 1), 2 ** (bit_width - 1) - 1],
    then scaled with 1 / (bit-width). Thus, it will finally be in [-0.5, 0.5].
    We initiate the quantizers input_low and input_high smaller than [-0.5, 0.5] by multiplying 0.1 to both limits.
    Let SGD optimizer to learn quantizer parameters with a MSE loss for the linear model.
    Check whether input_low and input_high is expanded to [-0.5, 0.5] to compensate quantization errors,
    and the MSE loss is also minimized. If we add target bit_bidth loss,
    we have to check whether our quantizer's learnable bit_width also goes to the target bit_width.
    """
    w, x, y, bit_width, sigma = linear_problem

    w, x, y = w.to(device=device), x.to(device=device), y.to(device=device)

    quant = FracBitsAsymmetricQuantizer(
        qspec) if qspec.mode == FracBitsQuantizationMode.ASYMMETRIC else FracBitsSymmetricQuantizer(qspec)

    init_input_low = torch.FloatTensor([w.min() * 0.1])
    init_input_high = torch.FloatTensor([w.max() * 0.1])

    quant.apply_minmax_init(init_input_low, init_input_high)
    quant = quant.to(w.device)
    criteria = nn.MSELoss()

    optim = torch.optim.SGD(quant.parameters(), lr=1e-1)

    for _ in range(100):
        optim.zero_grad()
        loss = criteria(y, quant(w).mm(x.t()))

        if add_bitwidth_loss:
            loss += criteria(bit_width *
                             torch.ones_like(quant.frac_num_bits), quant.frac_num_bits)

        loss.backward()
        optim.step()

    eps = 0.05
    ub_mse_loss = 1.1 * (sigma ** 2)
    ub_left_q_w = w.min() + eps
    lb_right_q_w = w.max() - eps

    with torch.no_grad():
        loss = criteria(y, quant(w).mm(x.t())).item()
        nncf_logger.debug(
            f"loss={loss:.3f} should be lower than ub_mse_loss={ub_mse_loss:.3f}.")
        assert loss < ub_mse_loss

    left_q_w, right_q_w = quant.get_input_range()
    left_q_w, right_q_w = left_q_w.item(), right_q_w.item()

    nncf_logger.debug(f"[left_q_w, right_q_w]^C [{left_q_w:.3f}, {right_q_w:.3f}]^C should be included in "
                      f"[ub_left_q_w, lb_right_q_w]^C = [{ub_left_q_w:.3f}, {lb_right_q_w:.3f}]^C.")

    assert left_q_w < ub_left_q_w
    assert lb_right_q_w < right_q_w

    if add_bitwidth_loss:
        assert quant.num_bits == bit_width
