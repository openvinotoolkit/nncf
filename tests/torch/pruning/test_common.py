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

from nncf.common.pruning.schedulers import BaselinePruningScheduler
from nncf.common.pruning.schedulers import ExponentialPruningScheduler
from nncf.common.pruning.schedulers import ExponentialWithBiasPruningScheduler
from nncf.torch import register_default_init_args
from nncf.torch.structures import LeGRInitArgs
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import create_ones_mock_dataloader
from tests.torch.pruning.helpers import PruningTestModel
from tests.torch.pruning.helpers import get_basic_pruning_config


@pytest.mark.parametrize("algo", ("filter_pruning",))
@pytest.mark.parametrize(
    ("scheduler", "scheduler_class"),
    (
        ("baseline", BaselinePruningScheduler),
        ("exponential", ExponentialPruningScheduler),
        ("exponential_with_bias", ExponentialWithBiasPruningScheduler),
    ),
)
def test_can_choose_scheduler(algo, scheduler, scheduler_class):
    config = get_basic_pruning_config()
    config["compression"]["algorithm"] = algo
    config["compression"]["params"]["schedule"] = scheduler
    model = PruningTestModel()
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    scheduler = compression_ctrl.scheduler
    assert isinstance(scheduler, scheduler_class)


@pytest.mark.parametrize(
    ("algo", "ref_scheduler", "ref_scheduler_params"),
    (
        (
            "filter_pruning",
            ExponentialPruningScheduler,
            {"num_warmup_epochs": 0, "num_pruning_epochs": 100, "initial_level": 0, "target_level": 0.5},
        ),
    ),
)
def test_check_default_scheduler_params(algo, ref_scheduler, ref_scheduler_params):
    config = get_basic_pruning_config()
    config["compression"]["algorithm"] = algo
    model = PruningTestModel()
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    scheduler = compression_ctrl.scheduler
    assert isinstance(scheduler, ref_scheduler)
    for key, value in ref_scheduler_params.items():
        assert getattr(scheduler, key) == value


def test_default_legr_init_struct():
    config = get_basic_pruning_config()
    init_loader = create_ones_mock_dataloader(config)
    nncf_config = register_default_init_args(config, init_loader)

    with pytest.raises(KeyError):
        nncf_config.get_extra_struct(LeGRInitArgs)


def test_valid_legr_init_struct():
    config = get_basic_pruning_config()
    train_loader = create_ones_mock_dataloader(config)
    val_loader = create_ones_mock_dataloader(config)
    train_steps_fn = lambda *x: None
    validate_fn = lambda *x: (0, 0, 0)
    nncf_config = register_default_init_args(
        config, train_loader=train_loader, train_steps_fn=train_steps_fn, val_loader=val_loader, validate_fn=validate_fn
    )

    legr_init_args = config.get_extra_struct(LeGRInitArgs)
    assert legr_init_args.config == nncf_config
    assert legr_init_args.train_loader == train_loader
    assert legr_init_args.val_loader == val_loader
    assert legr_init_args.train_steps_fn == train_steps_fn
