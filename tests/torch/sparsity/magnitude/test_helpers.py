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

import pytest
import torch
from torch import nn

from nncf.config import NNCFConfig
from tests.torch.helpers import PTTensorListComparator
from tests.torch.helpers import create_conv

pytestmark = pytest.mark.legacy


class MagnitudeTestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = create_conv(1, 2, 2, 9, -2)
        self.conv2 = create_conv(2, 1, 3, -10, 0)

    def forward(self, x):
        return self.conv2(self.conv1(x))


def test_magnitude_model_has_expected_params():
    model = MagnitudeTestModel()
    act_weights_1 = model.conv1.weight.data
    act_weights_2 = model.conv2.weight.data
    act_bias_1 = model.conv1.bias.data
    act_bias_2 = model.conv2.bias.data

    sub_tensor = torch.tensor([[[[10.0, 9.0], [9.0, 10.0]]]])
    ref_weights_1 = torch.cat((sub_tensor, sub_tensor), 0)
    sub_tensor = torch.tensor([[[[-9.0, -10.0, -10.0], [-10.0, -9.0, -10.0], [-10.0, -10.0, -9.0]]]])
    ref_weights_2 = torch.cat((sub_tensor, sub_tensor), 1)

    PTTensorListComparator.check_equal(act_weights_1, ref_weights_1)
    PTTensorListComparator.check_equal(act_weights_2, ref_weights_2)

    PTTensorListComparator.check_equal(act_bias_1, torch.tensor([-2.0, -2]))
    PTTensorListComparator.check_equal(act_bias_2, torch.tensor([0]))


def get_basic_magnitude_sparsity_config(input_sample_size=None):
    if input_sample_size is None:
        input_sample_size = [1, 1, 4, 4]
    config = NNCFConfig()
    config.update(
        {
            "model": "basic_sparse_conv",
            "input_info": {
                "sample_size": input_sample_size,
            },
            "compression": {"algorithm": "magnitude_sparsity", "params": {}},
        }
    )
    return config
