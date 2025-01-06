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

from copy import deepcopy

import pytest
import torch
from pytest import approx
from torch import nn

from nncf.api.compression import CompressionStage
from nncf.common.sparsity.schedulers import PolynomialSparsityScheduler
from nncf.config import NNCFConfig
from nncf.torch.module_operations import UpdateWeight
from nncf.torch.sparsity.rb.algo import RBSparsityController
from nncf.torch.sparsity.rb.layers import RBSparsifyingWeight
from nncf.torch.sparsity.rb.loss import SparseLoss
from nncf.torch.sparsity.rb.loss import SparseLossForPerLayerSparsity
from tests.torch.helpers import BasicConvTestModel
from tests.torch.helpers import MockModel
from tests.torch.helpers import TwoConvTestModel
from tests.torch.helpers import check_correct_nncf_modules_replacement
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import get_empty_config


def get_basic_sparsity_config(
    input_sample_size=None, sparsity_init=0.02, sparsity_target=0.5, sparsity_target_epoch=2, sparsity_freeze_epoch=3
):
    if input_sample_size is None:
        input_sample_size = [1, 1, 4, 4]

    config = NNCFConfig()
    config.update(
        {
            "model": "basic_sparse_conv",
            "input_info": {
                "sample_size": input_sample_size,
            },
            "compression": {
                "algorithm": "rb_sparsity",
                "sparsity_init": sparsity_init,
                "params": {
                    "schedule": "polynomial",
                    "sparsity_target": sparsity_target,
                    "sparsity_target_epoch": sparsity_target_epoch,
                    "sparsity_freeze_epoch": sparsity_freeze_epoch,
                },
            },
        }
    )
    return config


def test_can_load_sparse_algo__with_defaults():
    model = BasicConvTestModel()
    config = get_basic_sparsity_config()
    sparse_model, compression_ctrl = create_compressed_model_and_algo_for_test(deepcopy(model), config)
    assert isinstance(compression_ctrl, RBSparsityController)

    _, sparse_model_conv = check_correct_nncf_modules_replacement(model, sparse_model)

    for sparse_module in sparse_model_conv.values():
        store = []
        for op in sparse_module.pre_ops.values():
            if isinstance(op, UpdateWeight) and isinstance(op.operand, RBSparsifyingWeight):
                assert torch.allclose(op.operand.binary_mask, torch.ones_like(sparse_module.weight))
                assert not op.operand.frozen
                assert op.__class__.__name__ not in store
                store.append(op.__class__.__name__)


def test_can_set_sparse_layers_to_loss():
    model = BasicConvTestModel()
    config = get_basic_sparsity_config()
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    loss = compression_ctrl.loss
    assert isinstance(loss, SparseLoss)

    for layer in loss._sparse_layers:
        assert isinstance(layer, RBSparsifyingWeight)


def test_sparse_algo_does_not_replace_not_conv_layer():
    class TwoLayersTestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 1, 1)
            self.bn = nn.BatchNorm2d(1)

        def forward(self, x):
            return self.bn(self.conv(x))

    model = TwoLayersTestModel()
    config = get_basic_sparsity_config()
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    assert isinstance(compression_ctrl, RBSparsityController)
    for m in compression_ctrl.sparsified_module_info:
        assert isinstance(m.operand, RBSparsifyingWeight)


def test_can_create_sparse_loss_and_scheduler():
    model = BasicConvTestModel()

    config = get_basic_sparsity_config()
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    scheduler = compression_ctrl.scheduler
    scheduler.epoch_step()
    loss = compression_ctrl.loss
    assert isinstance(loss, SparseLoss)
    assert not loss.disabled
    assert loss.target_sparsity_rate == approx(0.02)
    assert loss.p == approx(0.05)

    assert isinstance(scheduler, PolynomialSparsityScheduler)
    assert scheduler.current_sparsity_level == approx(0.02)
    assert scheduler.target_level == approx(0.5)
    assert scheduler.target_epoch == 2
    assert scheduler.freeze_epoch == 3


def test_sparse_algo_can_calc_sparsity_rate__for_basic_model():
    model = BasicConvTestModel()

    config = get_basic_sparsity_config()
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    nncf_stats = compression_ctrl.statistics()
    sparse_model_stats = nncf_stats.rb_sparsity.model_statistics

    assert sparse_model_stats.sparsity_level == (
        1 - (model.nz_weights_num + model.nz_bias_num) / (model.weights_num + model.bias_num)
    )
    assert sparse_model_stats.sparsity_level_for_layers == 1 - model.nz_weights_num / model.weights_num
    assert len(compression_ctrl.sparsified_module_info) == 1


def test_sparse_algo_can_collect_sparse_layers():
    model = TwoConvTestModel()

    config = get_basic_sparsity_config()
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    assert len(compression_ctrl.sparsified_module_info) == 2


def test_sparse_algo_can_calc_sparsity_rate__for_2_conv_model():
    model = TwoConvTestModel()

    config = get_basic_sparsity_config()
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    nncf_stats = compression_ctrl.statistics()
    sparse_model_stats = nncf_stats.rb_sparsity.model_statistics

    assert pytest.approx(sparse_model_stats.sparsity_level) == (
        1 - (model.nz_weights_num + model.nz_bias_num) / (model.weights_num + model.bias_num)
    )
    assert sparse_model_stats.sparsity_level_for_layers == 1 - model.nz_weights_num / model.weights_num


def test_scheduler_can_do_epoch_step__with_rb_algo():
    config = NNCFConfig()
    config["input_info"] = [{"sample_size": [1, 1, 32, 32]}]
    config["compression"] = {
        "algorithm": "rb_sparsity",
        "sparsity_init": 0.2,
        "params": {
            "schedule": "polynomial",
            "power": 1,
            "sparsity_target_epoch": 2,
            "sparsity_target": 0.6,
            "sparsity_freeze_epoch": 3,
        },
    }

    _, compression_ctrl = create_compressed_model_and_algo_for_test(BasicConvTestModel(), config)
    scheduler = compression_ctrl.scheduler
    loss = compression_ctrl.loss

    assert pytest.approx(loss.target_sparsity_rate) == 0.2
    assert not loss.disabled

    for module_info in compression_ctrl.sparsified_module_info:
        assert not module_info.operand.frozen
    scheduler.epoch_step()
    assert pytest.approx(loss.target_sparsity_rate, abs=1e-3) == 0.2
    assert pytest.approx(loss().item(), abs=1e-3) == 16
    assert not loss.disabled

    scheduler.epoch_step()
    assert pytest.approx(loss.target_sparsity_rate, abs=1e-3) == 0.4
    assert pytest.approx(loss().item(), abs=1e-3) == 64
    assert not loss.disabled

    scheduler.epoch_step()
    assert pytest.approx(loss.target_sparsity_rate, abs=1e-3) == 0.6
    assert pytest.approx(loss().item(), abs=1e-3) == 144
    assert not loss.disabled

    scheduler.epoch_step()
    assert loss.disabled
    assert loss.target_sparsity_rate == 0.6
    assert loss() == 0

    for module_info in compression_ctrl.sparsified_module_info:
        assert module_info.operand.frozen


def test_create_rb_algo_with_per_layer_loss():
    config = get_empty_config()
    config["compression"] = {"algorithm": "rb_sparsity", "params": {"sparsity_level_setting_mode": "local"}}
    _, compression_ctrl = create_compressed_model_and_algo_for_test(MockModel(), config)

    assert isinstance(compression_ctrl._loss, SparseLossForPerLayerSparsity)


def test_rb_sparsity__can_set_sparsity_level_for_module():
    config = get_empty_config()
    config["compression"] = {"algorithm": "rb_sparsity", "params": {"sparsity_level_setting_mode": "local"}}
    _, compression_ctrl = create_compressed_model_and_algo_for_test(BasicConvTestModel(), config)

    assert list(compression_ctrl._loss.per_layer_target.values())[0] == 1

    compression_ctrl.set_sparsity_level(0.7, compression_ctrl.sparsified_module_info[0])
    assert list(compression_ctrl._loss.per_layer_target.values())[0] == pytest.approx(0.3)


def test_create_rb_algo_with_local_sparsity_mode():
    config = get_empty_config()
    config["compression"] = {"algorithm": "rb_sparsity", "params": {"sparsity_level_setting_mode": "local"}}
    _, compression_ctrl = create_compressed_model_and_algo_for_test(MockModel(), config)
    assert compression_ctrl.compression_stage() == CompressionStage.FULLY_COMPRESSED


def test_can_set_compression_rate_for_rb_sparsity_algo():
    config = get_basic_sparsity_config()
    _, compression_ctrl = create_compressed_model_and_algo_for_test(BasicConvTestModel(), config)
    compression_ctrl.compression_rate = 0.65
    assert pytest.approx(compression_ctrl.compression_rate, 1e-2) == 0.65
