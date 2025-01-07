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

from functools import partial

from torch import optim

from nncf.torch import register_default_init_args
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import create_ones_mock_dataloader
from tests.torch.pruning.filter_pruning.test_legr import create_default_legr_config
from tests.torch.pruning.helpers import PruningTestModel


def get_model_and_controller_for_legr_test():
    model = PruningTestModel()
    config = create_default_legr_config()
    train_loader = create_ones_mock_dataloader(config)
    val_loader = create_ones_mock_dataloader(config)
    train_steps_fn = lambda *x: None
    validate_fn = lambda *x: (0, 0)
    nncf_config = register_default_init_args(
        config, train_loader=train_loader, train_steps_fn=train_steps_fn, val_loader=val_loader, validate_fn=validate_fn
    )
    compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    return nncf_config, compressed_model, compression_ctrl


def test_evolution_optimizer_default_params():
    _, _, compression_ctrl = get_model_and_controller_for_legr_test()
    evo_optimizer = compression_ctrl.legr.agent

    assert evo_optimizer.population_size == 64
    assert evo_optimizer.num_generations == 400
    assert evo_optimizer.num_samples == 16
    assert evo_optimizer.mutate_percent == 0.1


def test_evolution_optimizer_interface():
    # That optimizer has all necessary functions: ask, tell
    _, _, compression_ctrl = get_model_and_controller_for_legr_test()
    evo_optimizer = compression_ctrl.legr.agent
    assert hasattr(evo_optimizer, "ask")
    assert hasattr(evo_optimizer, "tell")
    assert callable(evo_optimizer.ask)
    assert callable(evo_optimizer.tell)


def test_evolution_optimizer_reproducibility():
    # Check that optimizer generates same actions
    _, _, compression_ctrl_1 = get_model_and_controller_for_legr_test()
    evo_optimizer_1 = compression_ctrl_1.legr.agent

    _, _, compression_ctrl_2 = get_model_and_controller_for_legr_test()
    evo_optimizer_2 = compression_ctrl_2.legr.agent
    assert compression_ctrl_1.ranking_coeffs == compression_ctrl_2.ranking_coeffs
    assert evo_optimizer_1.population == evo_optimizer_2.population

    cur_episode = evo_optimizer_1.cur_episode
    assert cur_episode == evo_optimizer_2.cur_episode
    for _ in range(10):
        cur_episode += 1
        assert evo_optimizer_1.ask(cur_episode) == evo_optimizer_2.ask(cur_episode)


def test_evolution_env_default_params():
    model = PruningTestModel()
    config = create_default_legr_config()
    train_loader = create_ones_mock_dataloader(config)
    val_loader = create_ones_mock_dataloader(config)
    train_steps_fn = lambda *x: None
    validate_fn = lambda *x: (0, 0)
    nncf_config = register_default_init_args(
        config, train_loader=train_loader, train_steps_fn=train_steps_fn, val_loader=val_loader, validate_fn=validate_fn
    )
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    evolution_env = compression_ctrl.legr.env

    assert evolution_env.loss_as_reward is True
    assert evolution_env.prune_target == 0.5
    assert evolution_env.steps == 200

    assert evolution_env.train_loader == train_loader
    assert evolution_env.val_loader == val_loader
    assert evolution_env.train_fn == train_steps_fn
    assert evolution_env.validate_fn == validate_fn
    assert evolution_env.config == nncf_config


def test_evolution_env_setting_params():
    steps_ref = 100
    prune_target_ref = 0.1
    train_optimizer = partial(optim.Adam)

    model = PruningTestModel()
    config = create_default_legr_config()
    config["compression"]["params"]["legr_params"] = {}
    config["compression"]["params"]["legr_params"]["train_steps"] = steps_ref
    config["compression"]["params"]["legr_params"]["max_pruning"] = prune_target_ref
    train_loader = create_ones_mock_dataloader(config)
    val_loader = create_ones_mock_dataloader(config)
    train_steps_fn = lambda *x: None
    validate_fn = lambda *x: (0, 0)
    nncf_config = register_default_init_args(
        config,
        train_loader=train_loader,
        train_steps_fn=train_steps_fn,
        val_loader=val_loader,
        validate_fn=validate_fn,
        legr_train_optimizer=train_optimizer,
    )
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, nncf_config)
    evolution_env = compression_ctrl.legr.env

    assert evolution_env.prune_target == prune_target_ref
    assert evolution_env.steps == steps_ref
    assert evolution_env.train_optimizer == train_optimizer


def test_evolution_env_interface():
    _, _, compression_ctrl = get_model_and_controller_for_legr_test()
    evolution_env = compression_ctrl.legr.env

    assert hasattr(evolution_env, "reset")
    assert hasattr(evolution_env, "step")
    assert callable(evolution_env.reset)
    assert callable(evolution_env.step)


def test_pruner_default_params():
    _, _, compression_ctrl = get_model_and_controller_for_legr_test()
    legr_pruner = compression_ctrl.legr.pruner

    assert legr_pruner.filter_pruner == compression_ctrl
    assert isinstance(legr_pruner.scheduler, type(compression_ctrl.scheduler))


def test_pruner_interface():
    _, _, compression_ctrl = get_model_and_controller_for_legr_test()
    legr_pruner = compression_ctrl.legr.pruner

    assert hasattr(legr_pruner, "reset")
    assert hasattr(legr_pruner, "prune")
    assert hasattr(legr_pruner, "get_full_flops_number_in_model")
    assert callable(legr_pruner.reset)
    assert callable(legr_pruner.prune)
    assert callable(legr_pruner.get_full_flops_number_in_model)
