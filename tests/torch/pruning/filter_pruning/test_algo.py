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
import torch
import numpy as np

from examples.torch.common.optimizer import make_optimizer, get_parameter_groups
from nncf.torch.module_operations import UpdateWeight
from nncf.torch.pruning.filter_pruning.algo import FilterPruningController
from nncf.torch.pruning.filter_pruning.functions import l2_filter_norm
from nncf.torch.pruning.filter_pruning.layers import FilterPruningBlock
from nncf.torch.pruning.filter_pruning.layers import apply_filter_binary_mask
from nncf.common.pruning.schedulers import BaselinePruningScheduler
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import check_correct_nncf_modules_replacement
from tests.torch.helpers import create_ones_mock_dataloader
from tests.torch.pruning.helpers import gen_ref_masks
from tests.torch.pruning.helpers import get_basic_pruning_config
from tests.torch.pruning.helpers import PruningTestModel
from tests.torch.pruning.helpers import BigPruningTestModel
from tests.torch.pruning.helpers import TestModelMultipleForward
from tests.torch.pruning.helpers import PruningTestModelConcatBN


def create_pruning_algo_with_config(config):
    """
    Create filter_pruning with default params.
    :param config: config for the algorithm
    :return pruned model, pruning_algo, nncf_modules
    """
    config['compression']['algorithm'] = 'filter_pruning'
    model = BigPruningTestModel()
    pruned_model, pruning_algo = create_compressed_model_and_algo_for_test(BigPruningTestModel(), config)

    # Check that all modules was correctly replaced by NNCF modules and return this NNCF modules
    _, nncf_modules = check_correct_nncf_modules_replacement(model, pruned_model)
    return pruned_model, pruning_algo, nncf_modules


def test_check_default_algo_params():
    """
    Test for default algorithm params. Creating empty config and check for valid default
    parameters.
    """
    # Creating algorithm with empty config
    config = get_basic_pruning_config()
    config['compression']['algorithm'] = 'filter_pruning'
    model = PruningTestModel()
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    assert isinstance(compression_ctrl, FilterPruningController)
    scheduler = compression_ctrl.scheduler
    # Check default algo params
    assert compression_ctrl.prune_first is False
    assert compression_ctrl.prune_last is False
    assert compression_ctrl.prune_batch_norms is True
    assert compression_ctrl.filter_importance is l2_filter_norm

    assert compression_ctrl.all_weights is False
    assert compression_ctrl.zero_grad is True

    # Check default scheduler params
    assert isinstance(scheduler, BaselinePruningScheduler)


@pytest.mark.parametrize(
    ('prune_first', 'prune_last'),
    [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ]

)
def test_valid_modules_replacement_and_pruning(prune_first, prune_last):
    """
    Test that checks that all conv modules in model was replaced by nncf modules and
    pruning pre ops were added correctly.
    :param prune_first: whether to prune first convolution or not
    :param prune_last: whether to prune last convolution or not
    """

    def check_that_module_is_pruned(module):
        assert len(module.pre_ops.values()) == 1
        op = list(module.pre_ops.values())[0]
        assert isinstance(op, UpdateWeight)
        assert isinstance(op.operand, FilterPruningBlock)

    config = get_basic_pruning_config(input_sample_size=[1, 1, 8, 8])
    config['compression']['params']['prune_first_conv'] = prune_first
    config['compression']['params']['prune_last_conv'] = prune_last

    pruned_model, pruning_algo, nncf_modules = create_pruning_algo_with_config(config)
    pruned_module_info = pruning_algo.pruned_module_groups_info.get_all_nodes()
    pruned_modules = [minfo.module for minfo in pruned_module_info]

    # Check for conv1
    conv1 = pruned_model.conv1
    if prune_first:
        assert conv1 in pruned_modules
        assert conv1 in nncf_modules.values()
        check_that_module_is_pruned(conv1)

    # Check for conv2
    conv2 = pruned_model.conv2
    assert conv2 in pruned_modules
    assert conv2 in nncf_modules.values()
    check_that_module_is_pruned(conv2)

    # Check for conv3
    up = pruned_model.up
    assert up in pruned_modules
    assert up in nncf_modules.values()
    check_that_module_is_pruned(up)

    # Check for conv3W
    conv3 = pruned_model.conv3
    if prune_last:
        assert conv3 in pruned_modules
        assert conv3 in nncf_modules.values()
        check_that_module_is_pruned(conv3)


@pytest.mark.parametrize(('all_weights', 'pruning_flops_target', 'prune_first', 'ref_masks'),
                         [
                             (False, None, True, gen_ref_masks([(8, 8), (16, 16), (32, 32)])),
                             (True, None, True, gen_ref_masks([(5, 11), (9, 23), (42, 22)])),
                             (False, None, False, gen_ref_masks([(16, 16), (32, 32)])),
                             (True, None, False, gen_ref_masks([(8, 24), (40, 24)])),
                             # Flops pruning cases
                             (False, 0.5, True, gen_ref_masks([(0, 16), (8, 24), (24, 40)])),
                             (False, 0.5, False, gen_ref_masks([(8, 24), (24, 40)])),
                             (True, 0.5, True, gen_ref_masks([(4, 12), (3, 29), (30, 34)])),
                             (True, 0.5, False, gen_ref_masks([(3, 29), (31, 33)])),
                         ]
                         )
def test_pruning_masks_correctness(all_weights, pruning_flops_target, prune_first, ref_masks):
    """
    Test for pruning masks check (_set_binary_masks_for_filters, _set_binary_masks_for_all_filters_together).
    :param all_weights: whether mask will be calculated for all weights in common or not
    :param pruning_flops_target: prune model by flops, if None then by number of channels
    :param prune_first: whether to prune first convolution or not
    :param ref_masks: reference masks values
    """

    def check_mask(module, num):
        op = list(module.pre_ops.values())[0]
        assert hasattr(op.operand, 'binary_filter_pruning_mask')
        assert torch.allclose(op.operand.binary_filter_pruning_mask, ref_masks[num])

    config = get_basic_pruning_config(input_sample_size=[1, 1, 8, 8])
    config['compression']['params']['all_weights'] = all_weights
    config['compression']['params']['prune_first_conv'] = prune_first
    config['compression']['pruning_init'] = 0.5
    if pruning_flops_target:
        config['compression']['params']['pruning_flops_target'] = pruning_flops_target

    pruned_model, pruning_algo, _ = create_pruning_algo_with_config(config)
    pruned_module_info = pruning_algo.pruned_module_groups_info.get_all_nodes()
    pruned_modules = [minfo.module for minfo in pruned_module_info]
    assert pruning_algo.pruning_rate == 0.5
    assert pruning_algo.all_weights is all_weights

    i = 0
    # ref_masks Check for conv1
    conv1 = pruned_model.conv1
    if prune_first:
        assert conv1 in pruned_modules
        check_mask(conv1, i)
        i += 1

    # Check for conv2
    conv2 = pruned_model.conv2
    assert conv2 in pruned_modules
    check_mask(conv2, i)
    i += 1

    # Check for conv3
    up = pruned_model.up
    assert up in pruned_modules
    check_mask(up, i)


@pytest.mark.parametrize('prune_bn',
                         [False,
                          True]
                         )
def test_applying_masks(prune_bn):
    config = get_basic_pruning_config(input_sample_size=[1, 1, 8, 8])
    config['compression']['params']['prune_batch_norms'] = prune_bn
    config['compression']['params']['prune_first_conv'] = True
    config['compression']['params']['prune_last_conv'] = True
    config['compression']['pruning_init'] = 0.5

    pruned_model, pruning_algo, nncf_modules = create_pruning_algo_with_config(config)
    pruned_module_info = pruning_algo.pruned_module_groups_info.get_all_nodes()
    pruned_modules = [minfo.module for minfo in pruned_module_info]

    assert len(pruned_modules) == len(nncf_modules)

    for module in pruned_modules:
        op = list(module.pre_ops.values())[0]
        mask = op.operand.binary_filter_pruning_mask
        masked_weight = apply_filter_binary_mask(mask, module.weight, dim=module.target_weight_dim_for_compression)
        masked_bias = apply_filter_binary_mask(mask, module.bias)
        assert torch.allclose(module.weight, masked_weight)
        assert torch.allclose(module.bias, masked_bias)

    # Have only one BN node in graph
    bn_module = pruned_model.bn
    conv_for_bn = pruned_model.conv2
    bn_mask = list(conv_for_bn.pre_ops.values())[0].operand.binary_filter_pruning_mask
    if prune_bn:
        masked_bn_weight = apply_filter_binary_mask(bn_mask, bn_module.weight)
        masked_bn_bias = apply_filter_binary_mask(bn_mask, bn_module.bias)
        assert torch.allclose(bn_module.weight, masked_bn_weight)
        assert torch.allclose(bn_module.bias, masked_bn_bias)
    else:
        assert sum(bn_module.weight) == len(bn_module.weight)
        # Can not check bias because bias initialized with zeros


@pytest.mark.parametrize('prune_bn',
                         (False,
                          True)
                         )
def test_applying_masks_for_bn_after_concat(prune_bn):
    config = get_basic_pruning_config(input_sample_size=[1, 1, 8, 8])
    config['compression']['algorithm'] = 'filter_pruning'
    config['compression']['params']['prune_batch_norms'] = prune_bn
    config['compression']['params']['prune_first_conv'] = True
    config['compression']['params']['prune_last_conv'] = True
    config['compression']['pruning_init'] = 0.5
    model = PruningTestModelConcatBN()
    pruned_model, _ = create_compressed_model_and_algo_for_test(model, config)

    bn_modules = [pruned_model.bn, pruned_model.bn1, pruned_model.bn2]
    for bn_module in bn_modules:
        if prune_bn:
            # Check that mask was applied for batch_norm module
            assert sum(bn_module.weight) == len(bn_module.weight) * 0.5
            assert sum(bn_module.bias) == len(bn_module.bias) * 0.5
        else:
            # Check that mask was not applied for batch_norm module
            assert sum(bn_module.weight) == len(bn_module.weight)
            assert sum(bn_module.bias) == len(bn_module.bias)

    # Check output mask of concat layers
    ref_concat_masks = [
        [0] * 8 + [1] * 8 + [0] * 8 + [1] * 8,
        [1] * 8 + [0] * 16 + [1] * 8 + [0] * 8 + [1] * 8
    ]
    graph = pruned_model.get_original_graph()
    for i, node in enumerate(graph.get_nodes_by_types(['cat'])):
        assert np.allclose(node.data['output_mask'].numpy(), ref_concat_masks[i])


@pytest.mark.parametrize('zero_grad',
                         [True, False])
def test_zeroing_gradients(zero_grad):
    """
    Test for zeroing gradients functionality (zero_grads_for_pruned_modules in base algo)
    :param zero_grad: zero grad or not
    """
    config = get_basic_pruning_config(input_sample_size=[2, 1, 8, 8])
    config['compression']['params']['prune_first_conv'] = True
    config['compression']['params']['prune_last_conv'] = True
    config['compression']['params']['zero_grad'] = zero_grad

    pruned_model, pruning_algo, _ = create_pruning_algo_with_config(config)
    assert pruning_algo.zero_grad is zero_grad

    pruned_module_info = pruning_algo.pruned_module_groups_info.get_all_nodes()
    pruned_modules = [minfo.module for minfo in pruned_module_info]

    device = next(pruned_model.parameters()).device
    data_loader = create_ones_mock_dataloader(config)

    params_to_optimize = get_parameter_groups(pruned_model, config)
    optimizer, lr_scheduler = make_optimizer(params_to_optimize, config)

    lr_scheduler.step(0)

    pruned_model.train()
    for input_, target in data_loader:
        input_ = input_.to(device)
        target = target.to(device).view(1)

        output = pruned_model(input_)

        loss = torch.sum(target.to(torch.float32) - output)
        optimizer.zero_grad()
        loss.backward()

        # In case of zero_grad = True gradients should be masked
        if zero_grad:
            for module in pruned_modules:
                op = list(module.pre_ops.values())[0]
                mask = op.operand.binary_filter_pruning_mask
                grad = module.weight.grad
                masked_grad = apply_filter_binary_mask(mask, grad, dim=module.target_weight_dim_for_compression)
                assert torch.allclose(masked_grad, grad)


@pytest.mark.parametrize(('all_weights', 'pruning_flops_target', 'ref_flops', 'ref_params_num'),
                         [
                             (False, None, 1315008, 7776),
                             (True, None, 1492400, 9304),
                             (False, 0.5, 2367952, 13160),
                             (True, 0.5, 2380268, 13678),
                         ]
                         )
def test_calculation_of_flops(all_weights, pruning_flops_target, ref_flops, ref_params_num):
    """
    Test for pruning masks check (_set_binary_masks_for_filters, _set_binary_masks_for_all_filters_together).
    :param all_weights: whether mask will be calculated for all weights in common or not
    :param pruning_flops_target: prune model by flops, if None then by number of channels
    :param ref_flops: reference size of model
    """
    config = get_basic_pruning_config(input_sample_size=[1, 1, 8, 8])
    config['compression']['params']['all_weights'] = all_weights
    config['compression']['pruning_init'] = 0.5
    if pruning_flops_target:
        config['compression']['params']['pruning_flops_target'] = pruning_flops_target

    _, pruning_algo, _ = create_pruning_algo_with_config(config)

    assert pruning_algo.current_flops == ref_flops
    assert pruning_algo.current_params_num == ref_params_num
    # pylint:disable=protected-access
    assert pruning_algo._calculate_flops_and_weights_pruned_model_by_masks() == (ref_flops, ref_params_num)


def test_clusters_for_multiple_forward():
    config = get_basic_pruning_config(input_sample_size=[1, 2, 8, 8])
    config['compression']['algorithm'] = 'filter_pruning'
    config['compression']['params']['all_weights'] = False
    config['compression']['params']['prune_first_conv'] = True
    config['compression']['params']['prune_last_conv'] = True
    config['compression']['pruning_init'] = 0.5
    model = TestModelMultipleForward()
    _, pruning_algo = create_compressed_model_and_algo_for_test(model, config)

    clusters = pruning_algo.pruned_module_groups_info.clusters
    assert len(clusters) == 2
    # Convolutions before one node that forwards several times should be in one cluster
    assert sorted([n.nncf_node_id for n in clusters[0].elements]) == [1, 2, 3]
    # Nodes that associate with one module should be in one cluster
    assert sorted([n.nncf_node_id for n in clusters[1].elements]) == [4, 5, 6]
