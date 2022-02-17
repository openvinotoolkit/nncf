"""
 Copyright (c) 2022 Intel Corporation
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

from copy import deepcopy

import pytest
import torch
import numpy as np

from nncf.torch.module_operations import UpdateWeightAndBias
from nncf.torch.layers import NNCF_LINEAR_MODULES_DICT
from nncf.torch.pruning.filter_pruning.algo import FilterPruningController
from nncf.torch.pruning.filter_pruning.functions import l2_filter_norm
from nncf.torch.pruning.filter_pruning.layers import FilterPruningMask
from nncf.torch.pruning.filter_pruning.layers import apply_filter_binary_mask
from nncf.common.pruning.utils import calculate_in_out_channels_by_masks
from nncf.common.pruning.utils import count_flops_and_weights
from nncf.common.pruning.schedulers import ExponentialPruningScheduler
from nncf.torch.pruning.filter_pruning.algo import GENERAL_CONV_LAYER_METATYPES
from nncf.torch.pruning.filter_pruning.algo import LINEAR_LAYER_METATYPES
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import check_correct_nncf_modules_replacement
from tests.torch.pruning.helpers import gen_ref_masks
from tests.torch.pruning.helpers import get_basic_pruning_config
from tests.torch.pruning.helpers import PruningTestModel
from tests.torch.pruning.helpers import BigPruningTestModel
from tests.torch.pruning.helpers import MultipleForwardModel
from tests.torch.pruning.helpers import PruningTestModelConcatBN
from tests.torch.pruning.helpers import DisconectedGraphModel


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
    assert compression_ctrl.prune_batch_norms is True
    assert compression_ctrl.prune_downsample_convs is False
    assert compression_ctrl.filter_importance is l2_filter_norm
    assert compression_ctrl.ranking_type == 'unweighted_ranking'
    assert compression_ctrl.pruning_quota == 0.9

    assert compression_ctrl.all_weights is False

    # Check default scheduler params
    assert isinstance(scheduler, ExponentialPruningScheduler)


@pytest.mark.parametrize('prune_first', [False, True])
@pytest.mark.parametrize('prune_batch_norms', [True, False])
def test_valid_modules_replacement_and_pruning(prune_first, prune_batch_norms):
    """
    Test that checks that all conv modules in model was replaced by nncf modules and
    pruning pre ops were added correctly.
    :param prune_first: whether to prune first convolution or not
    :param prune_batch_norms: whether to prune batch norm layers or not.
    """

    def check_that_module_is_pruned(module):
        assert len(module.pre_ops.values()) == 1
        pre_ops = list(module.pre_ops.values())
        assert isinstance(pre_ops[0], UpdateWeightAndBias)
        pruning_op = pre_ops[0].operand
        assert isinstance(pruning_op, FilterPruningMask)

    def check_that_module_is_not_pruned(module):
        assert len(module.pre_ops) == 0
        assert len(module.post_ops) == 0

    config = get_basic_pruning_config(input_sample_size=[1, 1, 8, 8])
    config['compression']['params']['prune_first_conv'] = prune_first
    config['compression']['params']['prune_batch_norms'] = prune_batch_norms

    pruned_model, pruning_algo, nncf_modules = create_pruning_algo_with_config(config)
    pruned_module_info = pruning_algo.pruned_module_groups_info.get_all_nodes()
    pruned_modules = [minfo.module for minfo in pruned_module_info]

    # Check for conv1 and conv_depthwise
    conv1 = pruned_model.conv1
    conv_depthwise = pruned_model.conv_depthwise
    if prune_first:
        assert conv1 in pruned_modules
        assert conv1 in nncf_modules.values()
        check_that_module_is_pruned(conv1)

        assert conv_depthwise in nncf_modules.values()
        check_that_module_is_pruned(conv_depthwise)
    else:
        check_that_module_is_not_pruned(conv1)
        check_that_module_is_not_pruned(conv_depthwise)

    # Check for bn1
    bn1 = pruned_model.bn1
    if prune_first and prune_batch_norms:
        assert bn1 in nncf_modules.values()
        check_that_module_is_pruned(bn1)
    else:
        check_that_module_is_not_pruned(bn1)

    # Check for conv2
    conv2 = pruned_model.conv2
    assert conv2 in pruned_modules
    assert conv2 in nncf_modules.values()
    check_that_module_is_pruned(conv2)

    # Check for bn2
    bn2 = pruned_model.bn2
    if prune_batch_norms:
        assert bn2 in nncf_modules.values()
        check_that_module_is_pruned(bn2)
    else:
        check_that_module_is_not_pruned(bn2)

    # Check for conv3
    up = pruned_model.up
    assert up in pruned_modules
    assert up in nncf_modules.values()
    check_that_module_is_pruned(up)

    # Check for conv3W
    conv3 = pruned_model.conv3
    check_that_module_is_not_pruned(conv3)


BIG_PRUNING_MODEL_TEST_PARAMS = ('all_weights', 'pruning_flops_target', 'prune_first', 'ref_masks')
BIG_PRUNING_MODEL_TEST_PARAMS_VALUES = \
[
    (False, None, True, gen_ref_masks([(8, 8), (16, 16), (32, 32), (64, 64)])),
    (True, None, True, gen_ref_masks([(2, 14), (2, 30), (29, 35), (87, 41)])),
    (False, None, False, gen_ref_masks([(16, 16), (32, 32), (64, 64)])),
    (True, None, False, gen_ref_masks([(1, 31), (27, 37), (84, 44)])),
    # Flops pruning cases
    (False, 0.5, True, gen_ref_masks([(8, 8), (16, 16), (32, 32), (64, 64)])),
    (False, 0.5, False, gen_ref_masks([(16, 16), (32, 32), (64, 64)])),
    (True, 0.5, True, gen_ref_masks([(3, 13), (8, 24), (41, 23), (113, 15)])),
    (True, 0.5, False, gen_ref_masks([(9, 23), (41, 23), (113, 15)])),
]


@pytest.mark.parametrize(BIG_PRUNING_MODEL_TEST_PARAMS, BIG_PRUNING_MODEL_TEST_PARAMS_VALUES)
def test_pruning_masks_correctness(all_weights, pruning_flops_target, prune_first, ref_masks):
    """
    Test for pruning masks check (_set_binary_masks_for_filters, _set_binary_masks_for_all_filters_together).
    :param all_weights: whether mask will be calculated for all weights in common or not
    :param pruning_flops_target: prune model by flops, if None then by number of channels
    :param prune_first: whether to prune first convolution or not
    :param ref_masks: reference masks values
    """

    def check_mask(module, num):
        pruning_op = list(module.pre_ops.values())[0].operand
        assert hasattr(pruning_op, 'binary_filter_pruning_mask')
        #x = torch.sum((pruning_op.binary_filter_pruning_mask == 0.).int())
        #y = pruning_op.binary_filter_pruning_mask.shape[0]
        #y_minus_x = y - x
        #print(x, y)
        assert torch.allclose(pruning_op.binary_filter_pruning_mask, ref_masks[num])

    config = get_basic_pruning_config(input_sample_size=[1, 1, 8, 8])
    config['compression']['params']['all_weights'] = all_weights
    config['compression']['params']['prune_first_conv'] = prune_first
    config['compression']['pruning_init'] = 0.5
    if pruning_flops_target:
        config['compression']['params']['pruning_flops_target'] = pruning_flops_target

    pruned_model, pruning_algo, _ = create_pruning_algo_with_config(config)
    pruned_module_info = pruning_algo.pruned_module_groups_info.get_all_nodes()
    pruned_modules = [minfo.module for minfo in pruned_module_info]
    assert pruning_algo.pruning_level == 0.5
    assert pruning_algo.all_weights is all_weights

    i = 0
    # ref_masks Check for conv1
    conv1 = pruned_model.conv1
    conv_depthwise = pruned_model.conv_depthwise
    if prune_first:
        assert conv1 in pruned_modules
        assert conv_depthwise in pruned_modules

        check_mask(conv1, i)
        check_mask(conv_depthwise, i)
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
    i += 1

    # Check for linear
    linear = pruned_model.linear
    assert linear in pruned_modules
    check_mask(linear, i)


@pytest.mark.parametrize(BIG_PRUNING_MODEL_TEST_PARAMS, BIG_PRUNING_MODEL_TEST_PARAMS_VALUES )
def test_pruning_masks_applying_correctness(all_weights, pruning_flops_target, prune_first, ref_masks):
    """
    Test for pruning masks check (_set_binary_masks_for_filters, _set_binary_masks_for_all_filters_together).
    :param all_weights: whether mask will be calculated for all weights in common or not.
    :param pruning_flops_target: prune model by flops, if None then by number of channels.
    :param prune_first: whether to prune first convolution or not.
    :param ref_masks: reference masks values.
    """
    input_shapes = {'conv1': [1, 1, 8, 8],
                    'conv_depthwise': [1, 16, 7, 7],
                    'conv2': [1, 16, 8, 8],
                    'bn1': [1, 16, 8, 8],
                    'bn2': [1, 32, 8, 8],
                    'up': [1, 32, 8, 8],
                    'linear': [1, 3136]}

    def check_mask(module, num):
        # Mask for weights
        pruning_op = list(module.pre_ops.values())[0].operand
        assert hasattr(pruning_op, 'binary_filter_pruning_mask')
        assert torch.allclose(pruning_op.binary_filter_pruning_mask, ref_masks[num])

        # Mask for bias
        # pruning_op = list(module.pre_ops.values())[1].operand
        # assert hasattr(pruning_op, 'binary_filter_pruning_mask')
        # assert torch.allclose(pruning_op.binary_filter_pruning_mask, ref_masks[num])

    def check_module_output(module, name, num):
        """
        Checks that output of module are masked.
        """
        mask = ref_masks[num]
        input_ = torch.ones(input_shapes[name])
        output = module(input_)
        ref_output = apply_filter_binary_mask(mask, output, dim=1)
        assert torch.allclose(output, ref_output)

    def check_model_weights(model_state_dict, ref_state_dict):
        for key in ref_state_dict.keys():
            assert torch.allclose(model_state_dict['nncf_module.' + key], ref_state_dict[key])

    config = get_basic_pruning_config(input_sample_size=[1, 1, 8, 8])
    config['compression']['algorithm'] = 'filter_pruning'
    config['compression']['params']['all_weights'] = all_weights
    config['compression']['params']['prune_first_conv'] = prune_first
    config['compression']['pruning_init'] = 0.5
    if pruning_flops_target:
        config['compression']['params']['pruning_flops_target'] = pruning_flops_target

    model = BigPruningTestModel()
    ref_state_dict = deepcopy(model.state_dict())
    pruned_model, pruning_algo = create_compressed_model_and_algo_for_test(BigPruningTestModel(), config)

    pruned_module_info = pruning_algo.pruned_module_groups_info.get_all_nodes()
    pruned_modules = [minfo.module for minfo in pruned_module_info]
    assert pruning_algo.pruning_level == 0.5
    assert pruning_algo.all_weights is all_weights

    # Checking that model weights remain unchanged
    check_model_weights(pruned_model.state_dict(), ref_state_dict)

    i = 0
    # ref_masks Check for conv1
    conv1 = pruned_model.conv1
    conv_depthwise = pruned_model.conv_depthwise
    if prune_first:
        assert conv1 in pruned_modules
        assert conv_depthwise in pruned_modules

        check_mask(conv1, i)
        check_mask(conv_depthwise, i)

        check_module_output(conv1, 'conv1', i)
        check_module_output(conv_depthwise, 'conv_depthwise', i)

    # Check for bn1
    bn1 = pruned_model.bn1
    if prune_first:
        check_mask(bn1, i)
        check_module_output(bn1, 'bn1', i)
        i += 1

    # Check for conv2
    conv2 = pruned_model.conv2
    assert conv2 in pruned_modules
    check_mask(conv2, i)
    check_module_output(conv2, 'conv2', i)

    # Check for bn2
    bn2 = pruned_model.bn2
    check_mask(bn2, i)
    check_module_output(bn2, 'bn2', i)
    i += 1

    # Check for up conv
    up = pruned_model.up
    assert up in pruned_modules
    check_mask(up, i)
    check_module_output(up, 'up', i)
    i += 1

    # Check for linear
    linear = pruned_model.linear
    assert linear in pruned_modules
    check_mask(linear, i)
    check_module_output(linear, 'linear', i)


@pytest.mark.parametrize('prune_bn',
                         (False,
                          True)
                         )
def test_valid_masks_for_bn_after_concat(prune_bn):
    config = get_basic_pruning_config(input_sample_size=[1, 1, 8, 8])
    config['compression']['algorithm'] = 'filter_pruning'
    config['compression']['params']['prune_batch_norms'] = prune_bn
    config['compression']['params']['prune_first_conv'] = True
    config['compression']['pruning_init'] = 0.5
    model = PruningTestModelConcatBN()
    pruned_model, _ = create_compressed_model_and_algo_for_test(model, config)

    bn_modules = [pruned_model.bn, pruned_model.bn1, pruned_model.bn2]
    for bn_module in bn_modules:
        if prune_bn:
            # Check that mask was applied for batch_norm module
            mask = bn_module.pre_ops['0'].op.binary_filter_pruning_mask
            assert sum(mask) == len(mask) * 0.5
        else:
            # Check that no mask was added to the layer
            assert len(bn_module.pre_ops) == 0

    # Check output mask of concat layers
    ref_concat_masks = [
        [0] * 8 + [1] * 8 + [0] * 8 + [1] * 8,
        [1] * 8 + [0] * 16 + [1] * 8 + [0] * 8 + [1] * 8
    ]
    graph = pruned_model.get_original_graph()
    for i, node in enumerate(graph.get_nodes_by_types(['cat'])):
        assert np.allclose(node.data['output_mask'].tensor.numpy(), ref_concat_masks[i])


@pytest.mark.parametrize(('all_weights', 'pruning_flops_target', 'ref_flops', 'ref_params_num'),
                         [
                             (False, None, 1282000, 407336),
                             (True, None, 1808826, 414861),
                             (False, 0.5, 1282000, 407336),
                             (True, 0.5, 1351200, 409368),
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
    config['compression']['params']['prune_first_conv'] = True
    config['compression']['pruning_init'] = 0.5
    if pruning_flops_target:
        config['compression']['params']['pruning_flops_target'] = pruning_flops_target

    _, pruning_algo, _ = create_pruning_algo_with_config(config)

    assert pruning_algo.current_flops == ref_flops
    assert pruning_algo.current_params_num == ref_params_num
    # pylint:disable=protected-access
    tmp_in_channels, tmp_out_channels = calculate_in_out_channels_by_masks(
        pruning_algo.pruned_module_groups_info.get_all_clusters(),
        num_of_sparse_elements_by_node=pruning_algo._calculate_num_of_sparse_elements_by_node(),
        full_input_channels=pruning_algo._modules_in_channels,
        full_output_channels=pruning_algo._modules_out_channels,
        pruning_groups_next_nodes=pruning_algo.next_nodes)

    cur_flops, cur_params_num = count_flops_and_weights(
        pruning_algo._model.get_original_graph(),
        pruning_algo._modules_in_shapes,
        pruning_algo._modules_out_shapes,
        input_channels=tmp_in_channels,
        output_channels=tmp_out_channels,
        conv_op_metatypes=GENERAL_CONV_LAYER_METATYPES,
        linear_op_metatypes=LINEAR_LAYER_METATYPES)
    assert (cur_flops, cur_params_num) == (ref_flops, ref_params_num)


@pytest.mark.parametrize('repeat_seq_of_shared_convs,ref_second_cluster', [(True, [4, 5, 6, 7, 8, 9]),
                                                                           (False, [4, 5, 6])])
@pytest.mark.parametrize('additional_last_shared_layers', [True, False])
def test_clusters_for_multiple_forward(repeat_seq_of_shared_convs,
                                       ref_second_cluster,
                                       additional_last_shared_layers):
    config = get_basic_pruning_config(input_sample_size=[1, 2, 8, 8])
    config['compression']['algorithm'] = 'filter_pruning'
    config['compression']['params']['all_weights'] = False
    config['compression']['params']['prune_first_conv'] = True
    config['compression']['pruning_init'] = 0.5
    model = MultipleForwardModel(repeat_seq_of_shared_convs, additional_last_shared_layers)
    _, pruning_algo = create_compressed_model_and_algo_for_test(model, config)

    clusters = pruning_algo.pruned_module_groups_info.clusters
    ref_num_clusters = 2 if additional_last_shared_layers else 1
    assert len(clusters) == ref_num_clusters
    # Convolutions before one node that forwards several times should be in one cluster
    assert sorted([n.nncf_node_id for n in clusters[0].elements]) == [1, 2, 3]
    # In case of two clusters
    if additional_last_shared_layers:
        # Nodes that associate with one module should be in one cluster
        assert sorted([n.nncf_node_id for n in clusters[1].elements]) == ref_second_cluster


@pytest.mark.parametrize(
    ("model"),
    (
        BigPruningTestModel,
        PruningTestModelConcatBN
    )
)
def test_func_calulation_flops_for_conv(model):
    # Check _calculate_output_shape that used for disconnected graph
    config = get_basic_pruning_config([1, 1, 8, 8])
    config['compression']['algorithm'] = 'filter_pruning'
    config['compression']['pruning_init'] = 0.4
    config['compression']['params']['pruning_flops_target'] = 0.4
    model = model()
    pruned_model, pruning_algo = create_compressed_model_and_algo_for_test(model, config)

    graph = pruned_model.get_original_graph()

    # pylint:disable=protected-access
    for node_name, ref_shape in pruning_algo._modules_out_shapes.items():
        # ref_shape get from tracing graph
        node = graph.get_node_by_name(node_name)
        if node.node_type in [v.op_func_name for v in NNCF_LINEAR_MODULES_DICT]:
            continue

        shape = pruning_algo._calculate_output_shape(graph, node)
        assert ref_shape == shape, f"Incorrect calculation output name for {node_name}"


def test_disconnected_graph():
    config = get_basic_pruning_config([1, 1, 8, 8])
    config['compression']['algorithm'] = 'filter_pruning'
    config['compression']['pruning_init'] = 0.5
    config['compression']['params']['pruning_target'] = 0.5
    config['compression']['params']['prune_first_conv'] = True
    model = DisconectedGraphModel()
    pruned_model, _ = create_compressed_model_and_algo_for_test(model, config)
    graph = pruned_model.get_original_graph()

    conv1 = graph.get_node_by_name('DisconectedGraphModel/NNCFConv2d[conv1]/conv2d_0')
    conv2 = graph.get_node_by_name('DisconectedGraphModel/NNCFConv2d[conv2]/conv2d_0')

    assert sum(conv1.data['output_mask'].tensor) == 8
    assert sum(conv2.data['output_mask'].tensor) == 8
