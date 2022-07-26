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

from nncf.torch.quantization.layers import PTQuantizerSpec
from tests.torch.helpers import BasicConvTestModel, register_bn_adaptation_init_args
from tests.torch.quantization.test_quantization_helpers import get_empty_config


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


@pytest.fixture
def config(model_size: int = 4):
    config = get_empty_config(model_size)

    config["compression"] = {
        "algorithm": "fracbits_quantization",
        "initializer": {
            "range": {
                "num_init_samples": 0
            }
        },
        "freeze_epoch": -1,
        "loss": {
            "type": "model_size",
            "compression_rate": 1.5,
            "criteria": "L1"
        }
    }
    register_bn_adaptation_init_args(config)
    return config


@pytest.fixture
def conv_model():
    set_manual_seed()
    return BasicConvTestModel()


@pytest.fixture()
def lp_with_config_and_model(linear_problem, config):
    w, x, y, num_bits, sigma = linear_problem
    x = x.unsqueeze(0)
    y = y.unsqueeze(0).transpose(-2, -1)

    model = nn.Linear(in_features=w.shape[1], out_features=w.shape[0], bias=False)
    with torch.no_grad():
        model.weight.copy_(w)

    config["input_info"] = [{"sample_size": list(x.shape)}]
    return model, x, y, num_bits, sigma, nn.MSELoss(), config


@pytest.fixture
def conv_model_with_input_output(conv_model):
    with torch.no_grad():
        x = torch.randn([1, 1, 4, 4])
        y = conv_model(x)
        y += 0.1 * torch.randn_like(y)

    return conv_model, x, y
