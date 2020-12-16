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

from tests.helpers import create_compressed_model_and_algo_for_test
from tests.pruning.helpers import get_basic_pruning_config, PruningTestModel


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
    assert compression_ctrl.scheduler.pruning_target == pruning_target_ref


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
    ((PruningTestModel, {"modules_in_channels": {1: 1, 3: 3}, "modules_out_channels": {1: 3, 3: 1},
                         "nodes_flops": {1: 243, 3: 55}, "nodes_flops_cost": {1: 81, 3: 18}}),)
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
