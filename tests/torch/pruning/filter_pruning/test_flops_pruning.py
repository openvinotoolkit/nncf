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

import numpy as np
import pytest

import nncf
from nncf.common.pruning.utils import get_prunable_layers_in_out_channels
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.pruning.helpers import BigPruningTestModel
from tests.torch.pruning.helpers import GroupedConvolutionModel
from tests.torch.pruning.helpers import MobilenetV3BlockSEReshape
from tests.torch.pruning.helpers import PruningTestBatchedLinear
from tests.torch.pruning.helpers import PruningTestModel
from tests.torch.pruning.helpers import PruningTestModelBroadcastedLinear
from tests.torch.pruning.helpers import PruningTestModelConcatWithLinear
from tests.torch.pruning.helpers import PruningTestModelDiffChInPruningCluster
from tests.torch.pruning.helpers import PruningTestModelSharedConvs
from tests.torch.pruning.helpers import PruningTestWideModelConcat
from tests.torch.pruning.helpers import PruningTestWideModelEltwise
from tests.torch.pruning.helpers import get_basic_pruning_config


@pytest.mark.parametrize(
    ("pruning_target", "pruning_flops_target", "prune_flops_ref", "pruning_target_ref"),
    [
        (0.3, None, False, 0.3),
        (None, 0.3, True, 0.3),
        (None, None, False, 0.5),
    ],
)
def test_prune_flops_param(pruning_target, pruning_flops_target, prune_flops_ref, pruning_target_ref):
    config = get_basic_pruning_config()
    config["compression"]["algorithm"] = "filter_pruning"
    if pruning_target:
        config["compression"]["params"]["pruning_target"] = pruning_target
    if pruning_flops_target:
        config["compression"]["params"]["pruning_flops_target"] = pruning_flops_target
    config["compression"]["params"]["prune_first_conv"] = True

    model = PruningTestModel()
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    assert compression_ctrl.prune_flops is prune_flops_ref
    assert compression_ctrl.scheduler.target_level == pruning_target_ref


def test_both_targets_assert():
    config = get_basic_pruning_config()
    config["compression"]["algorithm"] = "filter_pruning"
    config["compression"]["params"]["pruning_target"] = 0.3
    config["compression"]["params"]["pruning_flops_target"] = 0.5

    model = PruningTestModel()
    with pytest.raises(ValueError):
        create_compressed_model_and_algo_for_test(model, config)


@pytest.mark.parametrize(
    ("model", "ref_params"),
    (
        (
            PruningTestModel,
            {
                "in_channels": {
                    PruningTestModel.CONV_1_NODE_NAME: 1,
                    PruningTestModel.CONV_2_NODE_NAME: 3,
                    PruningTestModel.CONV_3_NODE_NAME: 1,
                },
                "out_channels": {
                    PruningTestModel.CONV_1_NODE_NAME: 3,
                    PruningTestModel.CONV_2_NODE_NAME: 1,
                    PruningTestModel.CONV_3_NODE_NAME: 1,
                },
                "nodes_flops": {
                    PruningTestModel.CONV_1_NODE_NAME: 216,
                    PruningTestModel.CONV_2_NODE_NAME: 54,
                    PruningTestModel.CONV_3_NODE_NAME: 2,
                },
            },
        ),
    ),
)
def test_init_params_for_flops_calculation(model, ref_params):
    config = get_basic_pruning_config()
    config["compression"]["algorithm"] = "filter_pruning"
    config["compression"]["params"]["pruning_flops_target"] = 0.3
    config["compression"]["params"]["prune_first_conv"] = True

    model = model()
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    assert compression_ctrl.nodes_flops == ref_params["nodes_flops"]

    inp_channels, out_channels = get_prunable_layers_in_out_channels(compression_ctrl._graph)
    assert inp_channels == ref_params["in_channels"]
    assert out_channels == ref_params["out_channels"]


@pytest.mark.parametrize(
    ("model", "all_weights", "pruning_flops_target", "ref_full_flops", "ref_current_flops", "ref_sizes"),
    (
        (PruningTestWideModelConcat, False, 0.4, 671154176, 399057920, [328, 656, 656]),
        (PruningTestWideModelEltwise, False, 0.4, 268500992, 160773120, [720, 360]),
        (PruningTestWideModelConcat, True, 0.4, 671154176, 402513920, [380, 647, 648]),
        (PruningTestWideModelEltwise, True, 0.4, 268500992, 161043328, [755, 321]),
        (PruningTestModelSharedConvs, True, 0.4, 461438976, 276594768, [361, 809]),
        (PruningTestModelSharedConvs, False, 0.4, 461438976, 275300352, [384, 768]),
        (GroupedConvolutionModel, False, 0.0, 11243520, 11243520, []),
        (PruningTestModelConcatWithLinear, False, 0.1, 305792, 230912, [16, 24, 24]),
        (partial(MobilenetV3BlockSEReshape, mode="linear"), False, 0.1, 21360, 3532, [1, 1]),
        (PruningTestBatchedLinear, False, 0.0, 77824, 77824, []),
        (PruningTestModelBroadcastedLinear, False, 0.1, 137216, 103424, [16, 24]),
        (PruningTestModelDiffChInPruningCluster, False, 0.1, 1962368, 982336, [8]),
    ),
)
def test_flops_calulation_for_spec_layers(
    model, all_weights, pruning_flops_target, ref_full_flops, ref_current_flops, ref_sizes
):
    # Need check models with large size of layers because in other case
    # different value of pruning level give the same final size of model
    config = get_basic_pruning_config([1, 1, 8, 8])
    config["compression"]["algorithm"] = "filter_pruning"
    config["compression"]["pruning_init"] = pruning_flops_target
    config["compression"]["params"]["pruning_flops_target"] = pruning_flops_target
    config["compression"]["params"]["prune_first_conv"] = True
    config["compression"]["params"]["all_weights"] = all_weights
    model = model()
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    assert compression_ctrl.full_flops == ref_full_flops
    assert compression_ctrl.current_flops == ref_current_flops

    all_clusters = compression_ctrl.pruned_module_groups_info.get_all_clusters()
    assert len(all_clusters) == len(ref_sizes)
    for cluster, ref_size in zip(all_clusters, ref_sizes):
        for node in cluster.elements:
            op = list(node.module.pre_ops.values())[0]
            mask = op.operand.binary_filter_pruning_mask
            assert int(sum(mask)) == ref_size


def test_maximal_compression_rate():
    """
    Test that we can set flops pruning target less or equal to maximal_compression_rate
    Test that we can't set flops pruning target higher than maximal_compression_rate
    """
    config = get_basic_pruning_config(input_sample_size=[1, 1, 8, 8])
    config["compression"]["algorithm"] = "filter_pruning"
    config["compression"]["params"]["pruning_flops_target"] = 0.2
    config["compression"]["ignored_scopes"] = [
        "BigPruningTestModel/NNCFLinear[linear]/linear_0",
        "BigPruningTestModel/NNCFConvTranspose2d[up]/conv_transpose2d_0",
    ]

    _, pruning_algo = create_compressed_model_and_algo_for_test(BigPruningTestModel(), config)

    maximal_compression_rate = pruning_algo.maximal_compression_rate
    for comp_rate in np.linspace(0, maximal_compression_rate, 10):
        pruning_algo.compression_rate = comp_rate
    for comp_rate in np.linspace(maximal_compression_rate + 1e-5, 1, 10):
        with pytest.raises(nncf.InternalError):
            pruning_algo.compression_rate = comp_rate
