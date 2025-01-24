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
import itertools

import pytest
import torch

import nncf
from nncf import NNCFConfig
from nncf.torch.initialization import PTInitializingDataLoader
from tests.torch.helpers import TwoConvTestModel
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import register_bn_adaptation_init_args

SAMPLE_SIZE = [1, 1, 4, 4]


def get_config_for_logarithm_scale(logarithm_scale: bool, quantization_type: str) -> NNCFConfig:
    nncf_config = NNCFConfig()
    nncf_config.update(
        {
            "input_info": {"sample_size": SAMPLE_SIZE},
            "target_device": "TRIAL",
            "compression": {
                "algorithm": "quantization",
                "initializer": {
                    "range": {
                        "num_init_samples": 4,
                        "type": "percentile",
                        "params": {"min_percentile": 0.001, "max_percentile": 99.999},
                    }
                },
                "activations": {"mode": quantization_type, "logarithm_scale": logarithm_scale},
                "weights": {"mode": quantization_type, "signed": True, "logarithm_scale": logarithm_scale},
            },
        }
    )

    class RandDatasetMock:
        def __getitem__(self, index):
            return torch.rand(*SAMPLE_SIZE)

        def __len__(self):
            return 4

    data_loader = torch.utils.data.DataLoader(RandDatasetMock(), batch_size=1, shuffle=False, drop_last=True)

    class SquadInitializingDataloader(PTInitializingDataLoader):
        def get_inputs(self, dataloader_output):
            return dataloader_output, {}

        def get_target(self, dataloader_output):
            return None

    initializing_data_loader = SquadInitializingDataloader(data_loader)
    init_range = nncf.config.structures.QuantizationRangeInitArgs(initializing_data_loader)
    nncf_config.register_extra_structs([init_range])
    register_bn_adaptation_init_args(nncf_config)

    return nncf_config


@pytest.mark.parametrize(
    ["logarithm_scale_setting_1", "logarithm_scale_setting_2", "quantization_type"],
    list(itertools.product((True, False), (True, False), ("symmetric", "asymmetric"))),
)
def test_logarithm_scale_parameter(logarithm_scale_setting_1, logarithm_scale_setting_2, quantization_type):
    for logarithm_scales in [[False, True], [True, False]]:
        for symmetric in [False, True]:
            model0, _ = create_compressed_model_and_algo_for_test(
                TwoConvTestModel(),
                get_config_for_logarithm_scale(
                    logarithm_scale=logarithm_scale_setting_1, quantization_type=quantization_type
                ),
            )

            model1, _ = create_compressed_model_and_algo_for_test(
                TwoConvTestModel(),
                get_config_for_logarithm_scale(
                    logarithm_scale=logarithm_scale_setting_2, quantization_type=quantization_type
                ),
            )

            sd0 = model0.state_dict()
            model1.load_state_dict(sd0)
            sd1 = model1.state_dict()

            for k, v0 in sd0.items():
                v1 = sd1[k]
                diff = (v1 - v0).abs().sum().item() / v1.numel()
                assert diff < 1e-6, "symmetric {} logarithm_scales {} param {} is corrupted mean({}-{})={}".format(
                    symmetric, logarithm_scales, k, v0, v1, diff
                )
