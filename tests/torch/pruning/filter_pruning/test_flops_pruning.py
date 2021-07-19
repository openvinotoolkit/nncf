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
import pytest

from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.pruning.helpers import PruningTestModel
from tests.torch.pruning.helpers import PruningTestModelSharedConvs
from tests.torch.pruning.helpers import PruningTestWideModelConcat
from tests.torch.pruning.helpers import PruningTestWideModelEltwise
from tests.torch.pruning.helpers import get_basic_pruning_config


@pytest.mark.parametrize(
    ('pruning_target', 'pruning_flops_target', 'prune_flops_ref', 'pruning_target_ref'),
    [
        (0.3, None, False, 0.3),
        (None, 0.3, True, 0.3),
        (None, None, False, 0.5),
    ]

)
def test_prune_flops_param(pruning_target, pruning_flops_target, prune_flops_ref, pruning_target_ref):
    config = get_basic_pruning_config()
    config['compression']['algorithm'] = 'filter_pruning'
    if pruning_target:
        config['compression']['params']['pruning_target'] = pruning_target
    if pruning_flops_target:
        config['compression']['params']['pruning_flops_target'] = pruning_flops_target
    config['compression']['params']['prune_first_conv'] = True

    model = PruningTestModel()
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    assert compression_ctrl.prune_flops is prune_flops_ref
    assert compression_ctrl.scheduler.target_level == pruning_target_ref


def test_both_targets_assert():
    config = get_basic_pruning_config()
    config['compression']['algorithm'] = 'filter_pruning'
    config['compression']['params']['pruning_target'] = 0.3
    config['compression']['params']['pruning_flops_target'] = 0.5

    model = PruningTestModel()
    with pytest.raises(ValueError):
        create_compressed_model_and_algo_for_test(model, config)


@pytest.mark.parametrize(
    ("model", "ref_params"),
    ((PruningTestModel, {"_modules_in_channels": {PruningTestModel.CONV_1_NODE_NAME: 1,
                                                  PruningTestModel.CONV_2_NODE_NAME: 3},
                         "_modules_out_channels": {PruningTestModel.CONV_1_NODE_NAME: 3,
                                                   PruningTestModel.CONV_2_NODE_NAME: 1},
                         "nodes_flops": {PruningTestModel.CONV_1_NODE_NAME: 216,
                                         PruningTestModel.CONV_2_NODE_NAME: 54}}),)
)
def test_init_params_for_flops_calculation(model, ref_params):
    config = get_basic_pruning_config()
    config['compression']['algorithm'] = 'filter_pruning'
    config['compression']['params']['pruning_flops_target'] = 0.3
    config['compression']['params']['prune_first_conv'] = True

    model = model()
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    for key, value in ref_params.items():
        assert getattr(compression_ctrl, key) == value


@pytest.mark.parametrize(
    ("model", "all_weights", "ref_full_flops", "ref_current_flops", "ref_sizes"),
    (
        (PruningTestWideModelConcat, False, 671154176, 402311168, [400, 792, 792, 1584]),
        (PruningTestWideModelEltwise, False, 268500992, 157402112, [392, 784, 784, 784]),
        (PruningTestWideModelConcat, True, 671154176, 402578304, [441, 821, 822, 1473]),
        (PruningTestWideModelEltwise, True, 268500992, 161036544, [393, 855, 855, 685]),
        (PruningTestModelSharedConvs, True, 461438976, 276861184, [373, 827, 827]),
        (PruningTestModelSharedConvs, False, 461438976, 270498816, [392, 784, 784])
    )
)
def test_flops_calulation_for_spec_layers(model, all_weights, ref_full_flops, ref_current_flops, ref_sizes):
    # Need check models with large size of layers because in other case
    # different value of pruning rate give the same final size of model
    config = get_basic_pruning_config([1, 1, 8, 8])
    config['compression']['algorithm'] = 'filter_pruning'
    config['compression']['pruning_init'] = 0.4
    config['compression']['params']['pruning_flops_target'] = 0.4
    config['compression']['params']['prune_first_conv'] = True
    config['compression']['params']['prune_last_conv'] = True
    config['compression']['params']['all_weights'] = all_weights
    model = model()
    compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    assert compression_ctrl.full_flops == ref_full_flops
    assert compression_ctrl.current_flops == ref_current_flops

    for i, ref_size in enumerate(ref_sizes):
        node = getattr(compressed_model, f"conv{i+1}")
        op = list(node.pre_ops.values())[0]
        mask = op.operand.binary_filter_pruning_mask
        assert int(sum(mask)) == ref_size
