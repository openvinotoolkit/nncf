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
from typing import List

import pytest
import torch
from pytest import approx

from nncf.api.compression import CompressionStage
from nncf.torch.module_operations import UpdateWeight
from nncf.torch.sparsity.base_algo import SparseModuleInfo
from nncf.torch.sparsity.layers import BinaryMask
from nncf.torch.sparsity.magnitude.algo import MagnitudeSparsityController
from nncf.torch.sparsity.magnitude.functions import normed_magnitude
from tests.torch.helpers import BasicConvTestModel
from tests.torch.helpers import MockModel
from tests.torch.helpers import PTTensorListComparator
from tests.torch.helpers import check_correct_nncf_modules_replacement
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import get_empty_config
from tests.torch.sparsity.const.test_algo import ref_mask_1
from tests.torch.sparsity.const.test_algo import ref_mask_2
from tests.torch.sparsity.magnitude.test_helpers import MagnitudeTestModel
from tests.torch.sparsity.magnitude.test_helpers import get_basic_magnitude_sparsity_config


def test_can_create_magnitude_sparse_algo__with_defaults():
    model = MagnitudeTestModel()
    config = get_basic_magnitude_sparsity_config()
    config["compression"]["params"] = {"schedule": "multistep"}
    sparse_model, compression_ctrl = create_compressed_model_and_algo_for_test(deepcopy(model), config)

    assert isinstance(compression_ctrl, MagnitudeSparsityController)
    assert compression_ctrl.scheduler.current_sparsity_level == approx(0.1)
    assert len(list(sparse_model.modules())) == 12

    _, sparse_model_conv = check_correct_nncf_modules_replacement(model, sparse_model)

    nncf_stats = compression_ctrl.statistics()
    for layer_info in nncf_stats.magnitude_sparsity.thresholds:
        assert layer_info.threshold == approx(0.24, 0.1)

    assert isinstance(compression_ctrl._weight_importance_fn, type(normed_magnitude))

    for i, sparse_module in enumerate(sparse_model_conv.values()):
        store = []
        ref_mask = torch.ones_like(sparse_module.weight) if i == 0 else ref_mask_2
        for op in sparse_module.pre_ops.values():
            if isinstance(op, UpdateWeight) and isinstance(op.operand, BinaryMask):
                assert torch.allclose(op.operand.binary_mask, ref_mask)
                assert op.__class__.__name__ not in store
                store.append(op.__class__.__name__)


@pytest.mark.parametrize(
    ("weight_importance", "sparsity_level", "threshold"),
    (
        ("normed_abs", None, 0.219),
        ("abs", None, 9),
        ("normed_abs", 0.5, 0.243),
        ("abs", 0.5, 10),
    ),
)
def test_magnitude_sparse_algo_sets_threshold(weight_importance, sparsity_level, threshold):
    model = MagnitudeTestModel()
    config = get_basic_magnitude_sparsity_config()
    config["compression"]["params"] = {"schedule": "multistep", "weight_importance": weight_importance}
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    if sparsity_level:
        compression_ctrl.set_sparsity_level(sparsity_level)

    nncf_stats = compression_ctrl.statistics()
    for layer_info in nncf_stats.magnitude_sparsity.thresholds:
        assert layer_info.threshold == pytest.approx(threshold, 0.01)


def test_can_not_set_sparsity_more_than_one_for_magnitude_sparse_algo():
    config = get_basic_magnitude_sparsity_config()
    _, compression_ctrl = create_compressed_model_and_algo_for_test(MagnitudeTestModel(), config)
    with pytest.raises(AttributeError):
        compression_ctrl.set_sparsity_level(1)
        compression_ctrl.set_sparsity_level(1.2)


def test_can_not_create_magnitude_algo__without_steps():
    config = get_basic_magnitude_sparsity_config()
    config["compression"]["params"] = {"schedule": "multistep", "multistep_sparsity_levels": [0.1]}
    with pytest.raises(ValueError):
        _, _ = create_compressed_model_and_algo_for_test(MockModel(), config)


def test_can_create_magnitude_algo__without_levels():
    config = get_basic_magnitude_sparsity_config()
    config["compression"]["params"] = {"schedule": "multistep", "multistep_steps": [1]}
    _, compression_ctrl = create_compressed_model_and_algo_for_test(MockModel(), config)
    assert compression_ctrl.scheduler.current_sparsity_level == approx(0.1)


def test_can_not_create_magnitude_algo__with_not_matched_steps_and_levels():
    config = get_basic_magnitude_sparsity_config()
    config["compression"]["params"] = {
        "schedule": "multistep",
        "multistep_sparsity_levels": [0.1],
        "multistep_steps": [1, 2],
    }
    with pytest.raises(ValueError):
        _, _ = create_compressed_model_and_algo_for_test(MockModel(), config)


def test_magnitude_algo_set_binary_mask_on_forward():
    config = get_basic_magnitude_sparsity_config()
    config["compression"]["params"] = {"weight_importance": "abs"}
    sparse_model, compression_ctrl = create_compressed_model_and_algo_for_test(MagnitudeTestModel(), config)
    compression_ctrl.set_sparsity_level(0.3)
    with torch.no_grad():
        sparse_model(torch.ones([1, 1, 10, 10]))

    op = sparse_model.conv1.pre_ops["0"]
    PTTensorListComparator.check_equal(ref_mask_1, op.operand.binary_mask)

    op = sparse_model.conv2.pre_ops["0"]
    PTTensorListComparator.check_equal(ref_mask_2, op.operand.binary_mask)


def test_magnitude_algo_binary_masks_are_applied():
    model = BasicConvTestModel()
    config = get_empty_config()
    config["compression"] = {"algorithm": "magnitude_sparsity"}
    compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    minfo_list: List[SparseModuleInfo] = compression_ctrl.sparsified_module_info
    minfo: SparseModuleInfo = minfo_list[0]

    minfo.operand.binary_mask = torch.ones_like(minfo.module.weight)  # 1x1x2x2
    input_ = torch.ones(size=(1, 1, 5, 5))
    ref_output_1 = -4 * torch.ones(size=(2, 4, 4))
    output_1 = compressed_model(input_)
    assert torch.all(torch.eq(output_1, ref_output_1))

    minfo.operand.binary_mask[0][0][0][1] = 0
    minfo.operand.binary_mask[1][0][1][0] = 0
    ref_output_2 = -3 * torch.ones_like(ref_output_1)
    output_2 = compressed_model(input_)
    assert torch.all(torch.eq(output_2, ref_output_2))

    minfo.operand.binary_mask[1][0][0][1] = 0
    ref_output_3 = ref_output_2.clone()
    ref_output_3[1] = -2 * torch.ones_like(ref_output_1[1])
    output_3 = compressed_model(input_)
    assert torch.all(torch.eq(output_3, ref_output_3))


def test_magnitude_algo_set_independently_sparsity_level_for_one_module():
    module_name_conv1 = "MagnitudeTestModel/NNCFConv2d[conv1]/conv2d_0"
    module_name_conv2 = "MagnitudeTestModel/NNCFConv2d[conv2]/conv2d_0"
    config = get_basic_magnitude_sparsity_config()
    config["compression"]["params"] = {"sparsity_level_setting_mode": "local"}
    sparse_model, compression_ctrl = create_compressed_model_and_algo_for_test(MagnitudeTestModel(), config)
    sparse_info_conv1 = [
        sparse_info
        for sparse_info in compression_ctrl.sparsified_module_info
        if sparse_info.module_node_name == module_name_conv1
    ]
    sparse_info_conv2 = [
        sparse_info
        for sparse_info in compression_ctrl.sparsified_module_info
        if sparse_info.module_node_name == module_name_conv2
    ]

    compression_ctrl.set_sparsity_level(0.5, sparse_info_conv1[0])

    weights_conv1 = sparse_model.conv1.weight
    weights_conv2 = sparse_model.conv2.weight
    count_nonzero_conv1 = sparse_model.conv1.pre_ops["0"].operand.apply_binary_mask(weights_conv1).nonzero().size(0)
    count_param_conv1 = weights_conv1.view(-1).size(0)

    assert count_param_conv1 - count_nonzero_conv1 == 4  # 8 * 0.5

    compression_ctrl.set_sparsity_level(0.3, sparse_info_conv2[0])

    count_nonzero_conv1 = sparse_model.conv1.pre_ops["0"].operand.apply_binary_mask(weights_conv1).nonzero().size(0)
    count_param_conv1 = weights_conv1.view(-1).size(0)

    count_nonzero_conv2 = sparse_model.conv2.pre_ops["0"].operand.apply_binary_mask(weights_conv2).nonzero().size(0)
    count_param_conv2 = weights_conv2.view(-1).size(0)

    assert count_param_conv1 - count_nonzero_conv1 == 4  # 8 * 0.5
    assert count_param_conv2 - count_nonzero_conv2 == 6  # ~ 18 * 0.3


def test_can_do_sparsity_freeze_epoch():
    def compare_binary_mask(ref_sparse_module_info, sparse_module_info):
        for ref_sparse_layer, sparse_layer in zip(ref_sparse_module_info, sparse_module_info):
            if (ref_sparse_layer.operand.binary_mask != sparse_layer.operand.binary_mask).view(-1).sum() != 0:
                return False
        return True

    model = BasicConvTestModel()
    config = get_empty_config()
    config["compression"] = {
        "algorithm": "magnitude_sparsity",
        "sparsity_init": 0.1,
        "params": {"sparsity_target": 0.9, "sparsity_target_epoch": 3, "sparsity_freeze_epoch": 3},
    }
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    sparsified_minfo_before_update = deepcopy(compression_ctrl.sparsified_module_info)
    compression_ctrl.scheduler.epoch_step()  # update binary_masks
    compression_ctrl.scheduler.epoch_step()  # update binary_masks
    compression_ctrl.scheduler.epoch_step()  # update binary_masks, freeze binary_masks
    sparsified_minfo_after_update = deepcopy(compression_ctrl.sparsified_module_info)

    assert not compare_binary_mask(sparsified_minfo_after_update, sparsified_minfo_before_update)

    compression_ctrl.scheduler.epoch_step()  # don't update binary_masks
    sparsified_minfo_after_freeze = deepcopy(compression_ctrl.sparsified_module_info)

    assert compare_binary_mask(sparsified_minfo_after_update, sparsified_minfo_after_freeze)


def test_can_freeze_binary_masks():
    model = BasicConvTestModel()
    config = get_empty_config()
    config["compression"] = {"algorithm": "magnitude_sparsity"}
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    for sparse_layer in compression_ctrl.sparsified_module_info:
        assert not sparse_layer.operand.frozen

    compression_ctrl.freeze()

    for sparse_layer in compression_ctrl.sparsified_module_info:
        assert sparse_layer.operand.frozen


def test_create_magnitude_algo_with_local_sparsity_mode():
    config = get_empty_config()
    config["compression"] = {"algorithm": "magnitude_sparsity", "params": {"sparsity_level_setting_mode": "local"}}
    _, compression_ctrl = create_compressed_model_and_algo_for_test(MockModel(), config)
    assert compression_ctrl.compression_stage() == CompressionStage.FULLY_COMPRESSED


def test_magnitude_algo_can_calculate_correct_stats_for_local_mode():
    module_node_name_conv1 = "MagnitudeTestModel/NNCFConv2d[conv1]/conv2d_0"
    module_node_name_conv2 = "MagnitudeTestModel/NNCFConv2d[conv2]/conv2d_0"
    config = get_basic_magnitude_sparsity_config()
    config["compression"]["params"] = {"sparsity_level_setting_mode": "local"}
    _, compression_ctrl = create_compressed_model_and_algo_for_test(MagnitudeTestModel(), config)
    sparse_info_conv1 = [
        sparse_info
        for sparse_info in compression_ctrl.sparsified_module_info
        if sparse_info.module_node_name == module_node_name_conv1
    ]
    sparse_info_conv2 = [
        sparse_info
        for sparse_info in compression_ctrl.sparsified_module_info
        if sparse_info.module_node_name == module_node_name_conv2
    ]

    compression_ctrl.set_sparsity_level(0.5, sparse_info_conv1[0])

    compression_ctrl.set_sparsity_level(0.3, sparse_info_conv2[0])
    nncf_stats = compression_ctrl.statistics()

    expected = [(module_node_name_conv1, 0.3344823), (module_node_name_conv2, 0.2191864)]
    for idx, layer_info in enumerate(nncf_stats.magnitude_sparsity.thresholds):
        expected_name, expected_threshold = expected[idx]
        assert layer_info.name == expected_name
        assert pytest.approx(layer_info.threshold) == expected_threshold


def test_magnitude_algo_can_calculate_sparsity_rate_for_one_sparsified_module():
    module_node_name_conv1 = "MagnitudeTestModel/NNCFConv2d[conv1]/conv2d_0"
    config = get_basic_magnitude_sparsity_config()
    config["compression"]["params"] = {"sparsity_level_setting_mode": "local"}
    _, compression_ctrl = create_compressed_model_and_algo_for_test(MagnitudeTestModel(), config)
    sparse_info_conv1 = [
        sparse_info
        for sparse_info in compression_ctrl.sparsified_module_info
        if sparse_info.module_node_name == module_node_name_conv1
    ]

    compression_ctrl.set_sparsity_level(0.5, sparse_info_conv1[0])

    nncf_stats = compression_ctrl.statistics()
    sparse_model_stats = nncf_stats.magnitude_sparsity.model_statistics
    module_name_to_sparsity_level_map = {s.name: s.sparsity_level for s in sparse_model_stats.sparsified_layers_summary}

    module_name = sparse_info_conv1[0].module_node_name
    assert pytest.approx(module_name_to_sparsity_level_map[module_name], 1e-2) == 0.5


def test_can_set_compression_rate_for_magnitude_sparse_algo():
    config = get_basic_magnitude_sparsity_config()
    _, compression_ctrl = create_compressed_model_and_algo_for_test(MagnitudeTestModel(), config)
    compression_ctrl.compression_rate = 0.65
    assert pytest.approx(compression_ctrl.compression_rate, 1e-2) == 0.65
