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
from functools import partial

import numpy as np
import pytest
import torch

from nncf.api.compression import CompressionStage
from nncf.common.pruning.schedulers import ExponentialPruningScheduler
from nncf.common.pruning.shape_pruning_processor import ShapePruningProcessor
from nncf.common.pruning.weights_flops_calculator import WeightsFlopsCalculator
from nncf.torch.layers import NNCF_PRUNING_MODULES_DICT
from nncf.torch.module_operations import UpdateWeightAndBias
from nncf.torch.pruning.filter_pruning.algo import GENERAL_CONV_LAYER_METATYPES
from nncf.torch.pruning.filter_pruning.algo import LINEAR_LAYER_METATYPES
from nncf.torch.pruning.filter_pruning.algo import FilterPruningController
from nncf.torch.pruning.filter_pruning.functions import l2_filter_norm
from nncf.torch.pruning.filter_pruning.layers import FilterPruningMask
from nncf.torch.pruning.filter_pruning.layers import apply_filter_binary_mask
from nncf.torch.pruning.operations import PT_PRUNING_OPERATOR_METATYPES
from nncf.torch.pruning.utils import _calculate_output_shape
from nncf.torch.pruning.utils import collect_output_shapes
from tests.torch.helpers import check_correct_nncf_modules_replacement
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.pruning.helpers import BigPruningTestModel
from tests.torch.pruning.helpers import DisconectedGraphModel
from tests.torch.pruning.helpers import MultipleForwardModel
from tests.torch.pruning.helpers import PruningTestBatchedLinear
from tests.torch.pruning.helpers import PruningTestModel
from tests.torch.pruning.helpers import PruningTestModelBroadcastedLinear
from tests.torch.pruning.helpers import PruningTestModelBroadcastedLinearWithConcat
from tests.torch.pruning.helpers import PruningTestModelConcatBN
from tests.torch.pruning.helpers import PruningTestModelConcatWithLinear
from tests.torch.pruning.helpers import PruningTestModelDiffChInPruningCluster
from tests.torch.pruning.helpers import gen_ref_masks
from tests.torch.pruning.helpers import get_basic_pruning_config


def create_pruning_algo_with_config(config, dim=2):
    """
    Create filter_pruning with default params.
    :param config: config for the algorithm
    :return pruned model, pruning_algo, nncf_modules
    """
    config["compression"]["algorithm"] = "filter_pruning"
    model = BigPruningTestModel(dim)
    pruned_model, pruning_algo = create_compressed_model_and_algo_for_test(BigPruningTestModel(dim), config)

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
    config["compression"]["algorithm"] = "filter_pruning"
    model = PruningTestModel()
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    assert isinstance(compression_ctrl, FilterPruningController)
    scheduler = compression_ctrl.scheduler
    # Check default algo params
    assert compression_ctrl.prune_first is False
    assert compression_ctrl.prune_batch_norms is True
    assert compression_ctrl.prune_downsample_convs is False
    assert compression_ctrl.filter_importance is l2_filter_norm
    assert compression_ctrl.ranking_type == "unweighted_ranking"
    assert compression_ctrl.pruning_quota == 0.9

    assert compression_ctrl.all_weights is False

    # Check default scheduler params
    assert isinstance(scheduler, ExponentialPruningScheduler)


@pytest.mark.parametrize("prune_first", [False, True])
@pytest.mark.parametrize("prune_batch_norms", [True, False])
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
    config["compression"]["params"]["prune_first_conv"] = prune_first
    config["compression"]["params"]["prune_batch_norms"] = prune_batch_norms

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


BIG_PRUNING_MODEL_TEST_PARAMS = ("all_weights", "prune_by_flops", "pruning_init", "prune_first", "ref_masks")
BIG_PRUNING_MODEL_TEST_PARAMS_VALUES = \
[
    (False, False, 0.5, True, {1 : gen_ref_masks([(8, 8), (16, 16), (32, 32), (64, 64)]),
                         2 : gen_ref_masks([(8, 8), (16, 16), (32, 32), (64, 64)]),
                         3 : gen_ref_masks([(8, 8), (16, 16), (32, 32), (64, 64)])}),
    (True, False, 0.5, True, {1 : gen_ref_masks([(2, 14), (2, 30), (29, 35), (87, 41)]),
                        2 : gen_ref_masks([(2, 14), (2, 30), (29, 35), (87, 41)]),
                        3 : gen_ref_masks([(2, 14), (2, 30), (29, 35), (87, 41)])}),
    (False, False, 0.5, False, { 1 : gen_ref_masks([(16, 16), (32, 32), (64, 64)]),
                           2 : gen_ref_masks([(16, 16), (32, 32), (64, 64)]),
                           3 : gen_ref_masks([(16, 16), (32, 32), (64, 64)])}),

    (True, False, 0.5, False, { 1 : gen_ref_masks([(1, 31), (27, 37), (84, 44)]),
                          2 : gen_ref_masks([(1, 31), (27, 37), (84, 44)]),
                          3: gen_ref_masks([(1, 31), (27, 37), (84, 44)])}),
    # Flops pruning cases
    (False, True, 0.7, True, { 1 : gen_ref_masks([(8, 8), (16, 16), (32, 32), (64, 64)]),
                         2 : gen_ref_masks([(8, 8), (16, 16), (32, 32), (64, 64)]),
                         3 : gen_ref_masks([(8, 8), (16, 16), (32, 32), (64, 64)])}),
    (False, True, 0.7, False, {1 : gen_ref_masks([(16, 16), (32, 32), (64, 64)]),
                         2 : gen_ref_masks([(16, 16), (32, 32), (64, 64)]),
                         3 : gen_ref_masks([(16, 16), (32, 32), (64, 64)])}),
    (True, True, 0.7, True, { 1: gen_ref_masks([(2, 14), (4, 28), (31, 33), (93, 35)]),
                        2: gen_ref_masks([(2, 14), (6, 26), (35, 29), (102, 26)]),
                        3: gen_ref_masks([(2, 14), (7, 25), (38, 26), (106, 22)])}),
    (True, True, 0.7, False, { 1 : gen_ref_masks([(4, 28), (32, 32), (93, 35)]),
                         2 : gen_ref_masks([(6, 26), (36, 28), (102, 26)]),
                         3 : gen_ref_masks([(7, 25), (38, 26), (106, 22)])}),
]  # fmt: skip


@pytest.mark.parametrize(BIG_PRUNING_MODEL_TEST_PARAMS, BIG_PRUNING_MODEL_TEST_PARAMS_VALUES)
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_pruning_masks_correctness(all_weights, prune_by_flops, pruning_init, prune_first, ref_masks, dim):
    """
    Test for pruning masks check (_set_binary_masks_for_filters, _set_binary_masks_for_all_filters_together).
    :param all_weights: whether mask will be calculated for all weights in common or not
    :param pruning_flops_target: prune model by flops, if None then by number of channels
    :param prune_first: whether to prune first convolution or not
    :param ref_masks: reference masks values
    :param dim: dimension of the model
    """

    def check_mask(module, num):
        pruning_op = list(module.pre_ops.values())[0].operand
        assert hasattr(pruning_op, "binary_filter_pruning_mask")
        # x = torch.sum((pruning_op.binary_filter_pruning_mask == 0.).int())
        # y = pruning_op.binary_filter_pruning_mask.shape[0]
        # y_minus_x = y - x
        # print(x, y)
        assert torch.allclose(pruning_op.binary_filter_pruning_mask, ref_masks[dim][num])

    config = get_basic_pruning_config(input_sample_size=[1, 1] + [8] * dim)
    config["compression"]["params"]["all_weights"] = all_weights
    config["compression"]["params"]["prune_first_conv"] = prune_first

    config["compression"]["pruning_init"] = pruning_init
    if prune_by_flops:
        config["compression"]["params"]["pruning_flops_target"] = pruning_init

    pruned_model, pruning_algo, _ = create_pruning_algo_with_config(config, dim)
    pruned_module_info = pruning_algo.pruned_module_groups_info.get_all_nodes()
    pruned_modules = [minfo.module for minfo in pruned_module_info]
    assert pruning_algo.pruning_level == pruning_init
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


@pytest.mark.parametrize(BIG_PRUNING_MODEL_TEST_PARAMS, BIG_PRUNING_MODEL_TEST_PARAMS_VALUES)
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_pruning_masks_applying_correctness(all_weights, prune_by_flops, pruning_init, prune_first, ref_masks, dim):
    """
    Test for pruning masks check (_set_binary_masks_for_filters, _set_binary_masks_for_all_filters_together).
    :param all_weights: whether mask will be calculated for all weights in common or not.
    :param pruning_flops_target: prune model by flops, if None then by number of channels.
    :param prune_first: whether to prune first convolution or not.
    :param ref_masks: reference masks values.
    """
    input_shapes = {
        "conv1": [1, 1] + [8] * dim,
        "conv_depthwise": [1, 16] + [7] * dim,
        "conv2": [1, 16] + [8] * dim,
        "bn1": [1, 16] + [8] * dim,
        "bn2": [1, 32] + [8] * dim,
        "up": [1, 32] + [8] * dim,
        "linear": [1, 448 * 7 ** (dim - 1)],
        "layernorm": [1, 128],
    }

    def check_mask(module, num):
        # Mask for weights
        pruning_op = list(module.pre_ops.values())[0].operand
        assert hasattr(pruning_op, "binary_filter_pruning_mask")
        assert torch.allclose(pruning_op.binary_filter_pruning_mask, ref_masks[dim][num])

        # Mask for bias
        # pruning_op = list(module.pre_ops.values())[1].operand
        # assert hasattr(pruning_op, 'binary_filter_pruning_mask')
        # assert torch.allclose(pruning_op.binary_filter_pruning_mask, ref_masks[num])

    def check_module_output(module, name, num):
        """
        Checks that output of module are masked.
        """
        mask = ref_masks[dim][num]
        input_ = torch.ones(input_shapes[name])
        output = module(input_)
        ref_output = apply_filter_binary_mask(mask, output, dim=1)
        assert torch.allclose(output, ref_output)

    def check_model_weights(model_state_dict, ref_state_dict):
        for key in ref_state_dict:
            assert torch.allclose(model_state_dict[key], ref_state_dict[key])

    config = get_basic_pruning_config(input_sample_size=[1, 1] + [8] * dim)
    config["compression"]["algorithm"] = "filter_pruning"
    config["compression"]["params"]["all_weights"] = all_weights
    config["compression"]["params"]["prune_first_conv"] = prune_first
    config["compression"]["pruning_init"] = pruning_init
    if prune_by_flops:
        config["compression"]["params"]["pruning_flops_target"] = pruning_init

    model = BigPruningTestModel(dim)
    ref_state_dict = deepcopy(model.state_dict())
    pruned_model, pruning_algo = create_compressed_model_and_algo_for_test(BigPruningTestModel(dim), config)

    pruned_module_info = pruning_algo.pruned_module_groups_info.get_all_nodes()
    pruned_modules = [minfo.module for minfo in pruned_module_info]
    assert pruning_algo.pruning_level == pruning_init
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

        check_module_output(conv1, "conv1", i)
        check_module_output(conv_depthwise, "conv_depthwise", i)

    # Check for bn1
    bn1 = pruned_model.bn1
    if prune_first:
        check_mask(bn1, i)
        check_module_output(bn1, "bn1", i)
        i += 1

    # Check for conv2
    conv2 = pruned_model.conv2
    assert conv2 in pruned_modules
    check_mask(conv2, i)
    check_module_output(conv2, "conv2", i)

    # Check for bn2
    bn2 = pruned_model.bn2
    check_mask(bn2, i)
    check_module_output(bn2, "bn2", i)
    i += 1

    # Check for up conv
    up = pruned_model.up
    assert up in pruned_modules
    check_mask(up, i)
    check_module_output(up, "up", i)
    i += 1

    # Check for linear
    linear = pruned_model.linear
    assert linear in pruned_modules
    check_mask(linear, i)
    check_module_output(linear, "linear", i)

    # Check for layernorm
    check_mask(pruned_model.layernorm, i)
    check_module_output(pruned_model.layernorm, "layernorm", i)


@pytest.mark.parametrize("prune_bn", (False, True))
def test_valid_masks_for_bn_after_concat(prune_bn):
    config = get_basic_pruning_config(input_sample_size=[1, 1, 8, 8])
    config["compression"]["algorithm"] = "filter_pruning"
    config["compression"]["params"]["prune_batch_norms"] = prune_bn
    config["compression"]["params"]["prune_first_conv"] = True
    config["compression"]["pruning_init"] = 0.5
    model = PruningTestModelConcatBN()
    pruned_model, _ = create_compressed_model_and_algo_for_test(model, config)

    bn_modules = [pruned_model.bn, pruned_model.bn1, pruned_model.bn2]
    for bn_module in bn_modules:
        if prune_bn:
            # Check that mask was applied for batch_norm module
            mask = bn_module.pre_ops["0"].op.binary_filter_pruning_mask
            assert sum(mask) == len(mask) * 0.5
        else:
            # Check that no mask was added to the layer
            assert len(bn_module.pre_ops) == 0

    # Check output mask of concat layers
    ref_concat_masks = [[0] * 8 + [1] * 8 + [0] * 8 + [1] * 8, [1] * 8 + [0] * 16 + [1] * 8 + [0] * 8 + [1] * 8]
    graph = pruned_model.nncf.get_original_graph()
    for i, node in enumerate(graph.get_nodes_by_types(["cat"])):
        assert np.allclose(node.attributes["output_mask"].tensor.numpy(), ref_concat_masks[i])


@pytest.mark.parametrize('model,ref_output_shapes',
    [(partial(BigPruningTestModel, dim=2),
    {'BigPruningTestModel/NNCFConv2d[conv1]/conv2d_0': (7, 7),
     'BigPruningTestModel/NNCFConv2d[conv_depthwise]/conv2d_0': (5, 5),
     'BigPruningTestModel/NNCFConv2d[conv2]/conv2d_0': (3, 3),
     'BigPruningTestModel/NNCFConvTranspose2d[up]/conv_transpose2d_0': (7, 7),
     'BigPruningTestModel/NNCFConv2d[conv3]/conv2d_0': (1, 1),
     'BigPruningTestModel/NNCFLinear[linear]/linear_0': (1, 128)}),
     (PruningTestModelBroadcastedLinear,
     {'PruningTestModelBroadcastedLinear/NNCFConv2d[first_conv]/conv2d_0': (8, 8),
      'PruningTestModelBroadcastedLinear/NNCFConv2d[conv1]/conv2d_0': (8, 8),
      'PruningTestModelBroadcastedLinear/NNCFLinear[linear1]/linear_0': (1, 16),
      'PruningTestModelBroadcastedLinear/NNCFLinear[last_linear]/linear_0': (1, 1)}),
     (PruningTestModelConcatWithLinear,
     {'PruningTestModelConcatWithLinear/NNCFConv2d[conv1]/conv2d_0': (7, 7),
      'PruningTestModelConcatWithLinear/NNCFConv2d[conv2]/conv2d_0': (6, 6),
      'PruningTestModelConcatWithLinear/NNCFConv2d[conv3]/conv2d_0': (6, 6),
      'PruningTestModelConcatWithLinear/NNCFLinear[linear]/linear_0': (1, 1)}),
     (PruningTestModelBroadcastedLinearWithConcat,
     {'PruningTestModelBroadcastedLinearWithConcat/NNCFConv2d[first_conv]/conv2d_0': (8, 8),
      'PruningTestModelBroadcastedLinearWithConcat/NNCFConv2d[conv1]/conv2d_0': (8, 8),
      'PruningTestModelBroadcastedLinearWithConcat/NNCFConv2d[conv2]/conv2d_0': (8, 8),
      'PruningTestModelBroadcastedLinearWithConcat/NNCFLinear[linear1]/linear_0': (1, 16),
      'PruningTestModelBroadcastedLinearWithConcat/NNCFLinear[last_linear]/linear_0': (1, 1)}),
     (PruningTestBatchedLinear,
     {'PruningTestBatchedLinear/NNCFConv2d[first_conv]/conv2d_0': (8, 8),
     'PruningTestBatchedLinear/NNCFLinear[linear1]/linear_0': (1, 32, 8, 16),
     'PruningTestBatchedLinear/NNCFLinear[last_linear]/linear_0': (1, 1)}),
     (PruningTestModelDiffChInPruningCluster,
     {'PruningTestModelDiffChInPruningCluster/NNCFConv2d[first_conv]/conv2d_0': (7, 7),
     'PruningTestModelDiffChInPruningCluster/NNCFConv2d[conv1]/conv2d_0': (6, 6),
     'PruningTestModelDiffChInPruningCluster/NNCFLinear[linear1]/linear_0': (1, 1152),
     'PruningTestModelDiffChInPruningCluster/NNCFLinear[last_linear]/linear_0': (1, 1)})
     ])  # fmt: skip
def test_collect_output_shapes(model, ref_output_shapes):
    config = get_basic_pruning_config(input_sample_size=[1, 1, 8, 8])
    config["compression"]["algorithm"] = "filter_pruning"
    config["compression"]["pruning_init"] = 0.0
    model = model()
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    graph = compression_ctrl._model.nncf.get_original_graph()
    output_shapes = collect_output_shapes(graph)
    assert output_shapes == ref_output_shapes


# fmt: off
BigPruningTestModelNextNodesRef = {
    0: [{'node_name': 'BigPruningTestModel/NNCFConv2d[conv2]/conv2d_0', 'sparse_multiplier': 1}],
    1: [{'node_name': 'BigPruningTestModel/NNCFConvTranspose2d[up]/conv_transpose2d_0', 'sparse_multiplier': 1}],
    2: [{'node_name': 'BigPruningTestModel/NNCFLinear[linear]/linear_0', 'sparse_multiplier': 49}],
    3: [{'node_name': 'BigPruningTestModel/NNCFConv2d[conv3]/conv2d_0', 'sparse_multiplier': 1}],
}


BigPruningTestModelRef = {
    0:  {
        "next_nodes":
            BigPruningTestModelNextNodesRef,
        "num_of_sparse_by_node": {
            'BigPruningTestModel/NNCFConv2d[conv_depthwise]/conv2d_0': 8,
            'BigPruningTestModel/NNCFConv2d[conv1]/conv2d_0': 8,
            'BigPruningTestModel/NNCFConv2d[conv2]/conv2d_0': 16,
            'BigPruningTestModel/NNCFConvTranspose2d[up]/conv_transpose2d_0': 32,
            'BigPruningTestModel/NNCFLinear[linear]/linear_0': 64
        },
        "pruned_in_channels": {
            'BigPruningTestModel/NNCFConv2d[conv1]/conv2d_0': 1,
            'BigPruningTestModel/NNCFConv2d[conv_depthwise]/conv2d_0': 8,
            'BigPruningTestModel/NNCFConv2d[conv2]/conv2d_0': 8,
            'BigPruningTestModel/NNCFConvTranspose2d[up]/conv_transpose2d_0': 16,
            'BigPruningTestModel/NNCFLinear[linear]/linear_0': 1568,
            'BigPruningTestModel/NNCFConv2d[conv3]/conv2d_0': 64,
        },
        "pruned_out_channels": {
            'BigPruningTestModel/NNCFConv2d[conv1]/conv2d_0': 8,
            'BigPruningTestModel/NNCFConv2d[conv_depthwise]/conv2d_0': 8,
            'BigPruningTestModel/NNCFConv2d[conv2]/conv2d_0': 16,
            'BigPruningTestModel/NNCFConvTranspose2d[up]/conv_transpose2d_0': 32,
            'BigPruningTestModel/NNCFLinear[linear]/linear_0': 64,
            'BigPruningTestModel/NNCFConv2d[conv3]/conv2d_0': 1,
        }
    },
    1:{
        "next_nodes":
            BigPruningTestModelNextNodesRef,
        "num_of_sparse_by_node": {
            'BigPruningTestModel/NNCFConv2d[conv_depthwise]/conv2d_0': 2,
            'BigPruningTestModel/NNCFConv2d[conv1]/conv2d_0': 2,
            'BigPruningTestModel/NNCFConv2d[conv2]/conv2d_0': 2,
            'BigPruningTestModel/NNCFConvTranspose2d[up]/conv_transpose2d_0': 29,
            'BigPruningTestModel/NNCFLinear[linear]/linear_0': 87
        },
        "pruned_in_channels": {
            'BigPruningTestModel/NNCFConv2d[conv1]/conv2d_0': 1,
            'BigPruningTestModel/NNCFConv2d[conv_depthwise]/conv2d_0': 14,
            'BigPruningTestModel/NNCFConv2d[conv2]/conv2d_0': 14,
            'BigPruningTestModel/NNCFConvTranspose2d[up]/conv_transpose2d_0': 30,
            'BigPruningTestModel/NNCFLinear[linear]/linear_0': 1715,
            'BigPruningTestModel/NNCFConv2d[conv3]/conv2d_0': 41
        },
        "pruned_out_channels": {
            'BigPruningTestModel/NNCFConv2d[conv1]/conv2d_0': 14,
            'BigPruningTestModel/NNCFConv2d[conv_depthwise]/conv2d_0': 14,
            'BigPruningTestModel/NNCFConv2d[conv2]/conv2d_0': 30,
            'BigPruningTestModel/NNCFConvTranspose2d[up]/conv_transpose2d_0': 35,
            'BigPruningTestModel/NNCFLinear[linear]/linear_0': 41,
            'BigPruningTestModel/NNCFConv2d[conv3]/conv2d_0': 1
        }
    },
    2: {
        "next_nodes":
            BigPruningTestModelNextNodesRef,
        "num_of_sparse_by_node": {
            'BigPruningTestModel/NNCFConv2d[conv_depthwise]/conv2d_0': 0,
            'BigPruningTestModel/NNCFConv2d[conv1]/conv2d_0': 0,
            'BigPruningTestModel/NNCFConv2d[conv2]/conv2d_0': 8,
            'BigPruningTestModel/NNCFConvTranspose2d[up]/conv_transpose2d_0': 24,
            'BigPruningTestModel/NNCFLinear[linear]/linear_0': 48
        },
        "pruned_in_channels": {
            'BigPruningTestModel/NNCFConv2d[conv1]/conv2d_0': 1,
            'BigPruningTestModel/NNCFConv2d[conv_depthwise]/conv2d_0': 16,
            'BigPruningTestModel/NNCFConv2d[conv2]/conv2d_0': 16,
            'BigPruningTestModel/NNCFConvTranspose2d[up]/conv_transpose2d_0': 24,
            'BigPruningTestModel/NNCFLinear[linear]/linear_0': 1960,
            'BigPruningTestModel/NNCFConv2d[conv3]/conv2d_0': 80
        },
        "pruned_out_channels": {
            'BigPruningTestModel/NNCFConv2d[conv1]/conv2d_0': 16,
            'BigPruningTestModel/NNCFConv2d[conv_depthwise]/conv2d_0': 16,
            'BigPruningTestModel/NNCFConv2d[conv2]/conv2d_0': 24,
            'BigPruningTestModel/NNCFConvTranspose2d[up]/conv_transpose2d_0': 40,
            'BigPruningTestModel/NNCFLinear[linear]/linear_0': 80,
            'BigPruningTestModel/NNCFConv2d[conv3]/conv2d_0': 1
        }
    },
    3: {
        "next_nodes":
            BigPruningTestModelNextNodesRef,
        "num_of_sparse_by_node": {
            'BigPruningTestModel/NNCFConv2d[conv_depthwise]/conv2d_0': 2,
            'BigPruningTestModel/NNCFConv2d[conv1]/conv2d_0': 2,
            'BigPruningTestModel/NNCFConv2d[conv2]/conv2d_0': 1,
            'BigPruningTestModel/NNCFConvTranspose2d[up]/conv_transpose2d_0': 25,
            'BigPruningTestModel/NNCFLinear[linear]/linear_0': 81
        },
        "pruned_in_channels": {
            'BigPruningTestModel/NNCFConv2d[conv1]/conv2d_0': 1,
            'BigPruningTestModel/NNCFConv2d[conv_depthwise]/conv2d_0': 14,
            'BigPruningTestModel/NNCFConv2d[conv2]/conv2d_0': 14,
            'BigPruningTestModel/NNCFConvTranspose2d[up]/conv_transpose2d_0': 31,
            'BigPruningTestModel/NNCFLinear[linear]/linear_0': 1911,
            'BigPruningTestModel/NNCFConv2d[conv3]/conv2d_0': 47
        },
        "pruned_out_channels": {
            'BigPruningTestModel/NNCFConv2d[conv1]/conv2d_0': 14,
            'BigPruningTestModel/NNCFConv2d[conv_depthwise]/conv2d_0': 14,
            'BigPruningTestModel/NNCFConv2d[conv2]/conv2d_0': 31,
            'BigPruningTestModel/NNCFConvTranspose2d[up]/conv_transpose2d_0': 39,
            'BigPruningTestModel/NNCFLinear[linear]/linear_0': 47,
            'BigPruningTestModel/NNCFConv2d[conv3]/conv2d_0': 1
        }
    },
}

PruningTestModelBroadcastedLinearRefs = {
    "next_nodes": {
        0: [{'node_name':  'PruningTestModelBroadcastedLinear/NNCFLinear[last_linear]/linear_0',
             'sparse_multiplier': 64}],
        1: [{'node_name':  'PruningTestModelBroadcastedLinear/NNCFLinear[linear1]/linear_0', 'sparse_multiplier': 64},
            {'node_name':  'PruningTestModelBroadcastedLinear/NNCFConv2d[conv1]/conv2d_0', 'sparse_multiplier': 1}],
    },
    "num_of_sparse_by_node": {
        'PruningTestModelBroadcastedLinear/NNCFConv2d[first_conv]/conv2d_0': 16,
        'PruningTestModelBroadcastedLinear/NNCFConv2d[conv1]/conv2d_0': 8,
        'PruningTestModelBroadcastedLinear/NNCFLinear[linear1]/linear_0': 8,
    },
    "pruned_in_channels": {
        'PruningTestModelBroadcastedLinear/NNCFConv2d[first_conv]/conv2d_0': 1,
        'PruningTestModelBroadcastedLinear/NNCFConv2d[conv1]/conv2d_0': 16,
        'PruningTestModelBroadcastedLinear/NNCFLinear[linear1]/linear_0': 1024,
        'PruningTestModelBroadcastedLinear/NNCFLinear[last_linear]/linear_0': 512,
    },
    "pruned_out_channels": {
        'PruningTestModelBroadcastedLinear/NNCFConv2d[first_conv]/conv2d_0': 16,
        'PruningTestModelBroadcastedLinear/NNCFConv2d[conv1]/conv2d_0': 8,
        'PruningTestModelBroadcastedLinear/NNCFLinear[linear1]/linear_0': 8,
        'PruningTestModelBroadcastedLinear/NNCFLinear[last_linear]/linear_0': 1
    }
}

PruningTestModelBroadcastedLinearWithConcatRefs = {
    "next_nodes": {
        0: [{'node_name': 'PruningTestModelBroadcastedLinearWithConcat/NNCFLinear[last_linear]/linear_0',
             'sparse_multiplier': 64}],
        1: [{'node_name': 'PruningTestModelBroadcastedLinearWithConcat/NNCFLinear[linear1]/linear_0',
             'sparse_multiplier': 64},
            {'node_name': 'PruningTestModelBroadcastedLinearWithConcat/NNCFConv2d[conv1]/conv2d_0',
             'sparse_multiplier': 1},
            {'node_name': 'PruningTestModelBroadcastedLinearWithConcat/NNCFConv2d[conv2]/conv2d_0',
             'sparse_multiplier': 1}],
        2: [{'node_name': 'PruningTestModelBroadcastedLinearWithConcat/NNCFLinear[last_linear]/linear_0',
             'sparse_multiplier': 64}],
    },
    "num_of_sparse_by_node": {
        'PruningTestModelBroadcastedLinearWithConcat/NNCFConv2d[first_conv]/conv2d_0': 16,
        'PruningTestModelBroadcastedLinearWithConcat/NNCFConv2d[conv1]/conv2d_0': 8,
        'PruningTestModelBroadcastedLinearWithConcat/NNCFConv2d[conv2]/conv2d_0': 8,
        'PruningTestModelBroadcastedLinearWithConcat/NNCFLinear[linear1]/linear_0': 8,
    },
    "pruned_in_channels": {
        'PruningTestModelBroadcastedLinearWithConcat/NNCFConv2d[first_conv]/conv2d_0': 1,
        'PruningTestModelBroadcastedLinearWithConcat/NNCFConv2d[conv1]/conv2d_0': 16,
        'PruningTestModelBroadcastedLinearWithConcat/NNCFConv2d[conv2]/conv2d_0': 16,
        'PruningTestModelBroadcastedLinearWithConcat/NNCFLinear[linear1]/linear_0': 1024,
        'PruningTestModelBroadcastedLinearWithConcat/NNCFLinear[last_linear]/linear_0': 1024,
    },
    "pruned_out_channels": {
        'PruningTestModelBroadcastedLinearWithConcat/NNCFConv2d[first_conv]/conv2d_0': 16,
        'PruningTestModelBroadcastedLinearWithConcat/NNCFConv2d[conv1]/conv2d_0': 8,
        'PruningTestModelBroadcastedLinearWithConcat/NNCFConv2d[conv2]/conv2d_0': 8,
        'PruningTestModelBroadcastedLinearWithConcat/NNCFLinear[linear1]/linear_0': 8,
        'PruningTestModelBroadcastedLinearWithConcat/NNCFLinear[last_linear]/linear_0': 1
    }
}
PruningTestModelConcatWithLinearRefs = {
    "next_nodes": {
        0: [{'node_name': 'PruningTestModelConcatWithLinear/NNCFConv2d[conv2]/conv2d_0', 'sparse_multiplier': 1},
            {'node_name': 'PruningTestModelConcatWithLinear/NNCFConv2d[conv3]/conv2d_0', 'sparse_multiplier': 1}],
        1: [{'node_name': 'PruningTestModelConcatWithLinear/NNCFLinear[linear]/linear_0', 'sparse_multiplier': 36}],
        2: [{'node_name': 'PruningTestModelConcatWithLinear/NNCFLinear[linear]/linear_0', 'sparse_multiplier': 36}],
    },
    "num_of_sparse_by_node": {
        'PruningTestModelConcatWithLinear/NNCFConv2d[conv1]/conv2d_0': 8,
        'PruningTestModelConcatWithLinear/NNCFConv2d[conv2]/conv2d_0': 16,
        'PruningTestModelConcatWithLinear/NNCFConv2d[conv3]/conv2d_0': 16,
    },
    "pruned_in_channels": {
        'PruningTestModelConcatWithLinear/NNCFConv2d[conv1]/conv2d_0': 1,
        'PruningTestModelConcatWithLinear/NNCFConv2d[conv2]/conv2d_0': 8,
        'PruningTestModelConcatWithLinear/NNCFConv2d[conv3]/conv2d_0': 8,
        'PruningTestModelConcatWithLinear/NNCFLinear[linear]/linear_0': 1152,
    },
    "pruned_out_channels": {
        'PruningTestModelConcatWithLinear/NNCFConv2d[conv1]/conv2d_0': 8,
        'PruningTestModelConcatWithLinear/NNCFConv2d[conv2]/conv2d_0': 16,
        'PruningTestModelConcatWithLinear/NNCFConv2d[conv3]/conv2d_0': 16,
        'PruningTestModelConcatWithLinear/NNCFLinear[linear]/linear_0': 1,
    }
}

PruningTestBatchedLinearRef = { "next_nodes": {},
    "num_of_sparse_by_node": {},
    "pruned_in_channels": {'PruningTestBatchedLinear/NNCFConv2d[first_conv]/conv2d_0': 1,
                           'PruningTestBatchedLinear/NNCFLinear[linear1]/linear_0': 8,
                           'PruningTestBatchedLinear/NNCFLinear[last_linear]/linear_0': 4096},
    "pruned_out_channels": {'PruningTestBatchedLinear/NNCFConv2d[first_conv]/conv2d_0': 32,
                            'PruningTestBatchedLinear/NNCFLinear[linear1]/linear_0': 16,
                            'PruningTestBatchedLinear/NNCFLinear[last_linear]/linear_0': 1}
}

PruningTestModelDiffChInPruningClusterRef = {
    "next_nodes": {
        0: [{'node_name':  'PruningTestModelDiffChInPruningCluster/NNCFConv2d[conv1]/conv2d_0',
             'sparse_multiplier': 1},
            {'node_name':  'PruningTestModelDiffChInPruningCluster/NNCFLinear[linear1]/linear_0',
             'sparse_multiplier': 49}],
    },
    "num_of_sparse_by_node": {'PruningTestModelDiffChInPruningCluster/NNCFConv2d[first_conv]/conv2d_0': 8},

    "pruned_in_channels": {'PruningTestModelDiffChInPruningCluster/NNCFConv2d[first_conv]/conv2d_0': 1,
                           'PruningTestModelDiffChInPruningCluster/NNCFConv2d[conv1]/conv2d_0': 8,
                           'PruningTestModelDiffChInPruningCluster/NNCFLinear[linear1]/linear_0': 392,
                           'PruningTestModelDiffChInPruningCluster/NNCFLinear[last_linear]/linear_0': 1152},
    "pruned_out_channels": {'PruningTestModelDiffChInPruningCluster/NNCFConv2d[first_conv]/conv2d_0': 8,
                           'PruningTestModelDiffChInPruningCluster/NNCFConv2d[conv1]/conv2d_0': 32,
                           'PruningTestModelDiffChInPruningCluster/NNCFLinear[linear1]/linear_0': 1152,
                           'PruningTestModelDiffChInPruningCluster/NNCFLinear[last_linear]/linear_0': 1}
}
# fmt: on


@pytest.mark.parametrize(
    ('model_module', 'all_weights', 'pruning_flops_target', 'ref_flops',
     'ref_params_num', 'refs'),
     [
        (partial(BigPruningTestModel, dim=2), False, None, 679888, 106280, BigPruningTestModelRef[0]),
        (partial(BigPruningTestModel, dim=2), True, None, 1146640, 83768, BigPruningTestModelRef[1]),
        (partial(BigPruningTestModel, dim=2), False, 0.5, 1236160, 169184, BigPruningTestModelRef[2]),
        (partial(BigPruningTestModel, dim=2), True, 0.5, 1328162, 104833, BigPruningTestModelRef[3]),
        (PruningTestModelBroadcastedLinear, False, 0.3, 35840, 8848, PruningTestModelBroadcastedLinearRefs),
        (PruningTestModelConcatWithLinear, False, 0.3, 79168, 2208, PruningTestModelConcatWithLinearRefs),
        (PruningTestModelBroadcastedLinearWithConcat, False, 0.3, 53248, 9488,
         PruningTestModelBroadcastedLinearWithConcatRefs),
        (PruningTestBatchedLinear, False, 0.0, 77824, 4256, PruningTestBatchedLinearRef),
        (PruningTestModelDiffChInPruningCluster, False, 0.3, 982336, 453792,
         PruningTestModelDiffChInPruningClusterRef),

     ])  # fmt: skip
def test_flops_calculator(model_module, all_weights, pruning_flops_target, ref_flops, ref_params_num, refs):
    config = get_basic_pruning_config(input_sample_size=[1, 1, 8, 8])
    config["compression"]["algorithm"] = "filter_pruning"
    config["compression"]["params"]["all_weights"] = all_weights
    config["compression"]["params"]["prune_first_conv"] = True
    config["compression"]["pruning_init"] = pruning_flops_target if pruning_flops_target is not None else 0.5
    if pruning_flops_target is not None:
        config["compression"]["params"]["pruning_flops_target"] = pruning_flops_target

    model = model_module()
    pruned_model, pruning_algo = create_compressed_model_and_algo_for_test(model_module(), config)

    # Check that all modules was correctly replaced by NNCF modules and return this NNCF modules
    check_correct_nncf_modules_replacement(model, pruned_model)

    assert pruning_algo.current_flops == ref_flops
    assert pruning_algo.current_params_num == ref_params_num

    # Check num of sparse by node

    num_of_sparse_by_node = pruning_algo._calculate_num_of_sparse_elements_by_node()

    assert len(num_of_sparse_by_node) == len(refs["num_of_sparse_by_node"])
    for node_name in num_of_sparse_by_node:
        assert num_of_sparse_by_node[node_name] == refs["num_of_sparse_by_node"][node_name]

    graph = pruning_algo._model.nncf.get_original_graph()
    pruning_groups = pruning_algo.pruned_module_groups_info

    shape_pruning_processor = ShapePruningProcessor(
        pruning_operations_metatype=PT_PRUNING_OPERATOR_METATYPES,
        prunable_types=[v.op_func_name for v in NNCF_PRUNING_MODULES_DICT],
    )

    pruning_groups_next_nodes = shape_pruning_processor.get_next_nodes(graph, pruning_groups)
    # Check output_shapes are empty in graph
    for node in graph.get_all_nodes():
        assert node.attributes["output_shape"] is None

    # Next nodes cluster check
    assert len(pruning_groups_next_nodes) == len(refs["next_nodes"])
    for idx, next_nodes in pruning_groups_next_nodes.items():
        next_nodes_ref = refs["next_nodes"][idx]
        next_nodes_ref_names = [node["node_name"] for node in next_nodes_ref]
        for next_node in next_nodes:
            idx = next_nodes_ref_names.index(next_node["node_name"])
            next_node_ref = next_nodes_ref[idx]
            assert next_node["sparse_multiplier"] == next_node_ref["sparse_multiplier"]

    tmp_in_channels, tmp_out_channels = shape_pruning_processor.calculate_in_out_channels_by_masks(
        graph=graph,
        pruning_groups=pruning_groups,
        pruning_groups_next_nodes=pruning_groups_next_nodes,
        num_of_sparse_elements_by_node=refs["num_of_sparse_by_node"],
    )

    assert len(tmp_in_channels) == len(tmp_out_channels) == len(refs["pruned_in_channels"])
    for node_name in tmp_in_channels:
        assert tmp_in_channels[node_name] == refs["pruned_in_channels"][node_name]
        assert tmp_out_channels[node_name] == refs["pruned_out_channels"][node_name]

    output_shapes = collect_output_shapes(graph)
    weights_flops_calc = WeightsFlopsCalculator(
        conv_op_metatypes=GENERAL_CONV_LAYER_METATYPES, linear_op_metatypes=LINEAR_LAYER_METATYPES
    )

    cur_flops, cur_params_num = weights_flops_calc.count_flops_and_weights(
        graph=graph, output_shapes=output_shapes, input_channels=tmp_in_channels, output_channels=tmp_out_channels
    )
    assert (cur_flops, cur_params_num) == (ref_flops, ref_params_num)


@pytest.mark.parametrize(
    "repeat_seq_of_shared_convs,ref_second_cluster", [(True, [4, 5, 6, 7, 8, 9]), (False, [4, 5, 6])]
)
@pytest.mark.parametrize("additional_last_shared_layers", [True, False])
def test_clusters_for_multiple_forward(repeat_seq_of_shared_convs, ref_second_cluster, additional_last_shared_layers):
    config = get_basic_pruning_config(input_sample_size=[1, 2, 8, 8])
    config["compression"]["algorithm"] = "filter_pruning"
    config["compression"]["params"]["all_weights"] = False
    config["compression"]["params"]["prune_first_conv"] = True
    config["compression"]["pruning_init"] = 0.5
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
        PruningTestModelConcatBN,
        PruningTestBatchedLinear,
    ),
)
def test_func_calculation_flops_for_conv(model):
    # Check _calculate_output_shape that used for disconnected graph
    config = get_basic_pruning_config([1, 1, 8, 8])
    config["compression"]["algorithm"] = "filter_pruning"
    config["compression"]["pruning_init"] = 0.0
    config["compression"]["params"]["pruning_flops_target"] = 0.0
    model = model()
    pruned_model, compression_controller = create_compressed_model_and_algo_for_test(model, config)

    graph = pruned_model.nncf.get_original_graph()

    for node_name, ref_shape in compression_controller._output_shapes.items():
        # ref_shape get from tracing graph
        node = graph.get_node_by_name(node_name)
        shape = _calculate_output_shape(graph, node)
        assert ref_shape == shape, f"Incorrect calculation output name for {node_name}"


def test_disconnected_graph():
    config = get_basic_pruning_config([1, 1, 8, 8])
    config["compression"]["algorithm"] = "filter_pruning"
    config["compression"]["pruning_init"] = 0.5
    config["compression"]["params"]["pruning_target"] = 0.5
    config["compression"]["params"]["prune_first_conv"] = True
    model = DisconectedGraphModel()
    pruned_model, compression_controller = create_compressed_model_and_algo_for_test(model, config)
    graph = pruned_model.nncf.get_original_graph()

    nodes_output_mask_map = {
        'DisconectedGraphModel/NNCFConv2d[conv1]/conv2d_0': ((8, 8), None),
        'DisconectedGraphModel/NNCFConv2d[conv2]/conv2d_0': ((8, 8), 8),
        'DisconectedGraphModel/NNCFConv2d[conv3]/conv2d_0': ((8, 8), 1),
        'DisconectedGraphModel/NNCFLinear[fc]/linear_0': ((1, 3), None),
    }  # fmt: skip

    collected_shapes = compression_controller._output_shapes
    for name, (shape, mask_sum) in nodes_output_mask_map.items():
        node = graph.get_node_by_name(name)
        if mask_sum is None:
            assert node.attributes["output_mask"] is None
        else:
            assert sum(node.attributes["output_mask"].tensor) == mask_sum
        assert collected_shapes[name] == shape


@pytest.mark.parametrize("prune_by_flops", [True, False])
def test_compression_stage(prune_by_flops):
    config = get_basic_pruning_config(input_sample_size=[1, 1, 8, 8])
    pruning_init = 0.0
    pruning_target = 0.3
    config["compression"]["algorithm"] = "filter_pruning"
    config["compression"]["pruning_init"] = pruning_init
    config["compression"]["params"]["num_init_steps"] = 0
    config["compression"]["params"]["pruning_steps"] = 3

    if prune_by_flops:
        config["compression"]["params"]["pruning_flops_target"] = pruning_target
    else:
        config["compression"]["params"]["pruning_target"] = pruning_target
    _, pruning_algo, _ = create_pruning_algo_with_config(config)

    assert pruning_algo.compression_stage() == (
        CompressionStage.PARTIALLY_COMPRESSED if prune_by_flops else CompressionStage.UNCOMPRESSED
    )
    pruning_algo.scheduler.epoch_step()
    assert pruning_algo.compression_stage() == (
        CompressionStage.PARTIALLY_COMPRESSED if prune_by_flops else CompressionStage.UNCOMPRESSED
    )
    pruning_algo.scheduler.epoch_step()
    assert pruning_algo.compression_stage() == CompressionStage.PARTIALLY_COMPRESSED
    pruning_algo.scheduler.epoch_step()
    assert pruning_algo.compression_stage() == CompressionStage.FULLY_COMPRESSED
