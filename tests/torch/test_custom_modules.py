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

import torch
import torch.nn.functional

from nncf import NNCFConfig
from nncf.torch import register_module
from tests.torch.helpers import create_compressed_model_and_algo_for_test


@register_module()
class CustomConvModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones([1, 1, 1, 1]))

    def forward(self, x):
        return torch.nn.functional.conv2d(x, self.weight)


class ModelWithCustomConvModules(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.regular_conv = torch.nn.Conv2d(1, 1, 1)
        self.custom_conv = CustomConvModule()

    def forward(self, x):
        x = self.regular_conv(x)
        x = self.custom_conv(x)
        return x


def test_custom_module_processing():
    nncf_config = NNCFConfig.from_dict({"input_info": {"sample_size": [1, 1, 1, 1]}})

    # Should complete successfully without exceptions:
    create_compressed_model_and_algo_for_test(ModelWithCustomConvModules(), nncf_config)
