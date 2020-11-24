"""
 Copyright (c) 2020 Intel Corporation
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
import torch
import nncf
from nncf import NNCFConfig
from tests.helpers import create_compressed_model_and_algo_for_test

from tests.test_helpers import TwoConvTestModel

SAMPLE_SIZE = [1, 1, 4, 4]

def get_config_for_scale_log(scale_log: bool, symmetric: bool) -> NNCFConfig:
    nncf_config = NNCFConfig()
    nncf_config.update({
        "input_info": {
            "sample_size": SAMPLE_SIZE
        },
        "target_device": 'NONE',
        "compression": {
            "algorithm": "quantization",
            "initializer": {
                "range":{
                    "num_init_steps": 4,
                    "type": "percentile",
                    "min_percentile": 0.001,
                    "max_percentile": 99.999
                }
            },
            "activations": {
                "mode": "symmetric" if symmetric else "asymmetric",
                "scale_log": scale_log
            },
            "weights": {
                "mode": "symmetric" if symmetric else "asymmetric",
                "signed": True,
                "scale_log": scale_log
            }
        }
    })

    class RandDatasetMock:
        def __getitem__(self, index):
            return torch.rand(*SAMPLE_SIZE)

        def __len__(self):
            return 4

    data_loader = torch.utils.data.DataLoader(RandDatasetMock(), batch_size=1, shuffle=False)

    class SquadInitializingDataloader(nncf.initialization.InitializingDataLoader):
        def get_inputs(self, batch):
            return batch, {}

    initializing_data_loader = SquadInitializingDataloader(data_loader)
    init_range = nncf.initialization.QuantizationRangeInitArgs(initializing_data_loader)
    nncf_config.register_extra_structs([init_range])

    return nncf_config

def test_scale_log_parameter():
    for scale_logs in [[False, True], [True, False]]:
        for symmetric in [False, True]:
            model0, _ = create_compressed_model_and_algo_for_test(
                TwoConvTestModel(),
                get_config_for_scale_log(scale_log=scale_logs[0], symmetric=symmetric))

            model1, _ = create_compressed_model_and_algo_for_test(
                TwoConvTestModel(),
                get_config_for_scale_log(scale_log=scale_logs[1], symmetric=symmetric))

            sd0 = model0.state_dict()
            model1.load_state_dict(sd0)
            sd1 = model1.state_dict()

            for k, v0 in sd0.items():
                v1 = sd1[k]
                diff = (v1-v0).abs().sum().item() / v1.numel()
                #print("symmetric {} scale_logs {} param {} diff {}".format(symmetric, scale_logs, k, diff))
                assert diff < 1e-6, "symmetric {} scale_logs {} param {} is corrupted mean({}-{})={}".format(
                    symmetric, scale_logs, k, v0, v1, diff)
