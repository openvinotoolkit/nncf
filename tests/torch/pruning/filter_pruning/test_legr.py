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
import json

import numpy as np

from nncf.torch.initialization import register_default_init_args
from nncf.torch.pruning.filter_pruning.functions import l2_filter_norm
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import create_ones_mock_dataloader
from tests.torch.pruning.helpers import PruningTestModel
from tests.torch.pruning.helpers import get_basic_pruning_config


def create_default_legr_config():
    config = get_basic_pruning_config()
    config["compression"]["algorithm"] = "filter_pruning"
    config["compression"]["params"]["interlayer_ranking_type"] = "learned_ranking"
    return config


def test_legr_coeffs_loading(tmp_path):
    file_name = tmp_path / "ranking_coeffs.json"
    model = PruningTestModel()
    # Generate random ranking coefficients and save them
    ranking_coeffs = {
        str(model.CONV_1_NODE_NAME): (np.random.normal(), np.random.normal()),
        str(model.CONV_2_NODE_NAME): (np.random.normal(), np.random.normal()),
    }
    with open(file_name, "w", encoding="utf8") as f:
        json.dump(ranking_coeffs, f)

    config = create_default_legr_config()
    config["compression"]["params"]["load_ranking_coeffs_path"] = str(file_name)

    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    assert compression_ctrl.ranking_coeffs == ranking_coeffs


def test_legr_coeffs_saving(tmp_path):
    file_name = tmp_path / "ranking_coeffs.json"
    model = PruningTestModel()
    ref_ranking_coeffs = {model.CONV_1_NODE_NAME: (1, 0), model.CONV_2_NODE_NAME: (1, 0)}

    config = get_basic_pruning_config()
    config["compression"]["algorithm"] = "filter_pruning"
    config["compression"]["params"]["prune_first_conv"] = True
    config["compression"]["params"]["save_ranking_coeffs_path"] = str(file_name)
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    assert compression_ctrl.ranking_coeffs == ref_ranking_coeffs

    # check that in specified file some coeffs are saved (1, 0 in case of not-legr)
    with open(file_name, "r", encoding="utf8") as f:
        saved_coeffs_in_file = json.load(f)
    assert all(ref_ranking_coeffs[key] == tuple(saved_coeffs_in_file[key]) for key in saved_coeffs_in_file)


def test_legr_coeffs_save_and_load(tmp_path):
    file_name_save = tmp_path / "save_ranking_coeffs.json"

    model = PruningTestModel()
    ref_ranking_coeffs = {
        str(model.CONV_1_NODE_NAME): (np.random.normal(), np.random.normal()),
        str(model.CONV_2_NODE_NAME): (np.random.normal(), np.random.normal()),
    }
    with open(file_name_save, "w", encoding="utf8") as f:
        json.dump(ref_ranking_coeffs, f)

    config = create_default_legr_config()
    config["compression"]["params"]["save_ranking_coeffs_path"] = str(file_name_save)
    config["compression"]["params"]["load_ranking_coeffs_path"] = str(file_name_save)

    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    assert compression_ctrl.ranking_coeffs == ref_ranking_coeffs
    with open(file_name_save, "r", encoding="utf8") as f:
        saved_coeffs_in_file = json.load(f)
    assert all(ref_ranking_coeffs[key] == tuple(saved_coeffs_in_file[key]) for key in saved_coeffs_in_file)


def test_default_pruning_params_for_legr(tmp_path):
    file_name_save = tmp_path / "ranking_coeffs.json"
    model = PruningTestModel()
    # Generate random ranking coefficients and save them
    ranking_coeffs = {str(model.CONV_1_NODE_NAME): (0, 1), str(model.CONV_2_NODE_NAME): (0, 1)}
    with open(file_name_save, "w", encoding="utf8") as f:
        json.dump(ranking_coeffs, f)

    config = create_default_legr_config()
    config["compression"]["params"]["load_ranking_coeffs_path"] = str(file_name_save)
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    assert compression_ctrl.prune_first is True
    assert compression_ctrl.all_weights is True
    assert compression_ctrl.prune_downsample_convs is True

    assert compression_ctrl.filter_importance is l2_filter_norm
    assert compression_ctrl.ranking_type == "learned_ranking"


def test_legr_class_default_params(tmp_path):
    model = PruningTestModel()
    config = create_default_legr_config()
    train_loader = create_ones_mock_dataloader(config)
    val_loader = create_ones_mock_dataloader(config)
    train_steps_fn = lambda *x: None
    validate_fn = lambda *x: (0, 0)
    nncf_config = register_default_init_args(
        config, train_loader=train_loader, train_steps_fn=train_steps_fn, val_loader=val_loader, validate_fn=validate_fn
    )
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, nncf_config)

    compression_ctrl.legr.num_generations = 400
    compression_ctrl.legr.max_pruning = 0.8
    compression_ctrl.legr._train_steps = 200
    compression_ctrl.legr.random_seed = 42


def test_legr_class_setting_params(tmp_path):
    generations_ref = 150
    train_steps_ref = 50
    max_pruning_ref = 0.1

    model = PruningTestModel()
    config = create_default_legr_config()
    config["compression"]["params"]["legr_params"] = {}
    config["compression"]["params"]["legr_params"]["generations"] = generations_ref
    config["compression"]["params"]["legr_params"]["train_steps"] = train_steps_ref
    config["compression"]["params"]["legr_params"]["max_pruning"] = max_pruning_ref
    config["compression"]["params"]["legr_params"]["random_seed"] = 1

    train_loader = create_ones_mock_dataloader(config)
    val_loader = create_ones_mock_dataloader(config)
    train_steps_fn = lambda *x: None
    validate_fn = lambda *x: (0, 0)
    nncf_config = register_default_init_args(
        config, train_loader=train_loader, train_steps_fn=train_steps_fn, val_loader=val_loader, validate_fn=validate_fn
    )
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, nncf_config)

    compression_ctrl.legr.num_generations = generations_ref
    compression_ctrl.legr.max_pruning = max_pruning_ref
    compression_ctrl.legr._train_steps = train_steps_ref
    compression_ctrl.legr.random_seed = 1


def test_legr_reproducibility():
    np.random.seed(42)
    config = create_default_legr_config()

    train_loader = create_ones_mock_dataloader(config)
    val_loader = create_ones_mock_dataloader(config)
    train_steps_fn = lambda *x: None
    validate_fn = lambda *x: (0, np.random.random())
    nncf_config = register_default_init_args(
        config, train_loader=train_loader, train_steps_fn=train_steps_fn, val_loader=val_loader, validate_fn=validate_fn
    )
    model_1 = PruningTestModel()
    _, compression_ctrl_1 = create_compressed_model_and_algo_for_test(model_1, nncf_config)

    model_2 = PruningTestModel()
    _, compression_ctrl_2 = create_compressed_model_and_algo_for_test(model_2, config)

    assert compression_ctrl_1.ranking_coeffs == compression_ctrl_2.ranking_coeffs
