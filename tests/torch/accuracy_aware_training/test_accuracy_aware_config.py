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

import nncf
from nncf.common.accuracy_aware_training import create_accuracy_aware_training_loop
from nncf.torch.initialization import register_default_init_args
from tests.torch.helpers import LeNet
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import create_ones_mock_dataloader
from tests.torch.quantization.quantization_helpers import get_quantization_config_without_range_init


@pytest.mark.parametrize(
    ("aa_config", "must_raise"),
    (
        (
            {
                "compression": [
                    {
                        "algorithm": "filter_pruning",
                    },
                    {
                        "algorithm": "quantization",
                    },
                ]
            },
            False,
        ),
        (
            {
                "compression": {
                    "algorithm": "quantization",
                },
            },
            True,
        ),
        (
            {
                "compression": [
                    {
                        "algorithm": "filter_pruning",
                    },
                    {
                        "algorithm": "rb_sparsity",
                    },
                ]
            },
            True,
        ),
        (
            {
                "accuracy_aware_training": {
                    "mode": "early_exit",
                    "params": {"maximal_relative_accuracy_degradation": 1, "maximal_total_epochs": 1},
                },
                "compression": [
                    {
                        "algorithm": "filter_pruning",
                    },
                    {
                        "algorithm": "rb_sparsity",
                    },
                ],
            },
            False,
        ),
    ),
)
def test_accuracy_aware_config(aa_config, must_raise):
    def mock_validate_fn(model):
        pass

    config = get_quantization_config_without_range_init(LeNet.INPUT_SIZE[-1])

    config.update(
        {
            "accuracy_aware_training": {
                "mode": "adaptive_compression_level",
                "params": {
                    "maximal_relative_accuracy_degradation": 1,
                    "initial_training_phase_epochs": 1,
                    "patience_epochs": 10,
                },
            }
        }
    )

    config.update(aa_config)

    train_loader = create_ones_mock_dataloader(config, num_samples=10)
    model = LeNet()

    config = register_default_init_args(config, train_loader=train_loader, model_eval_fn=mock_validate_fn)
    model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    if must_raise:
        with pytest.raises(nncf.ValidationError):
            _ = create_accuracy_aware_training_loop(config, compression_ctrl, 0, dump_checkpoints=False)
    else:
        _ = create_accuracy_aware_training_loop(config, compression_ctrl, 0, dump_checkpoints=False)
