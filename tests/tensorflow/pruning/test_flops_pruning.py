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

import numpy as np
import pytest
import tensorflow as tf

import nncf
from nncf.common.pruning.shape_pruning_processor import ShapePruningProcessor
from nncf.common.pruning.weights_flops_calculator import WeightsFlopsCalculator
from nncf.tensorflow.graph.metatypes.common import GENERAL_CONV_LAYER_METATYPES
from nncf.tensorflow.graph.metatypes.common import LINEAR_LAYER_METATYPES
from nncf.tensorflow.pruning.operations import TF_PRUNING_OPERATOR_METATYPES
from nncf.tensorflow.pruning.utils import collect_output_shapes
from tests.tensorflow.helpers import create_compressed_model_and_algo_for_test
from tests.tensorflow.pruning.helpers import get_basic_pruning_config
from tests.tensorflow.pruning.helpers import get_batched_linear_model
from tests.tensorflow.pruning.helpers import get_broadcasted_linear_model
from tests.tensorflow.pruning.helpers import get_diff_cluster_channels_model
from tests.tensorflow.pruning.helpers import get_model_depthwise_conv
from tests.tensorflow.pruning.helpers import get_model_grouped_convs
from tests.tensorflow.pruning.helpers import get_test_model_shared_convs


@pytest.mark.parametrize(
    (
        "model_fn",
        "all_weights",
        "pruning_flops_target",
        "ref_full_flops",
        "ref_current_flops",
        "ref_full_params",
        "ref_current_params",
        "ref_sizes",
        "ref_num_of_sparse",
    ),
    (
        (
            get_test_model_shared_convs,
            True,
            0.4,
            461438976,
            276858560,
            11534848,
            6920562,
            [{"conv1": 410}, {"conv2": 3382272}],
            {"conv1": 102, "conv2^1": 290, "conv2^0": 290},
        ),
        (
            get_test_model_shared_convs,
            False,
            0.4,
            461438976,
            275300352,
            11534848,
            6881664,
            [{"conv1": 384}, {"conv2": 3538944}],
            {"conv1": 128, "conv2^1": 256, "conv2^0": 256},
        ),
        (get_model_grouped_convs, False, 0.0, 10859520, 10859520, 215808, 215808, [], {}),
        (
            get_model_depthwise_conv,
            True,
            0.8,
            783360,
            154368,
            23688,
            7776,
            [{"conv1": 8}, {"conv2": 1728, "conv4": 216}, {"conv3": 9216}],
            {"conv1": 0, "conv4": 104, "conv2": 104, "conv3": 0},
        ),
        (
            get_broadcasted_linear_model,
            True,
            0.9,
            167936,
            15872,
            33568,
            1384,
            [{"conv1": 160, "linear1": 10240}, {"first_conv": 4}],
            {"conv1": 11, "linear1": 11, "first_conv": 28},
        ),
        # (get_batched_linear_model, True, 0.0, 71680, 71680, 1568, 1568, [], {}), # ticket: 90141
        (get_batched_linear_model, False, 0.0, 71680, 71680, 1568, 1568, [], {}),
        (
            get_diff_cluster_channels_model,
            False,
            0.0,
            8523776,
            8523776,
            4227616,
            4227616,
            [{"first_conv": 32}],
            {"first_conv": 0},
        ),
    ),
)
def test_flops_calulation_for_spec_layers(
    model_fn,
    all_weights,
    pruning_flops_target,
    ref_full_flops,
    ref_current_flops,
    ref_full_params,
    ref_current_params,
    ref_sizes,
    ref_num_of_sparse,
):
    config = get_basic_pruning_config(8)
    config["compression"]["algorithm"] = "filter_pruning"
    config["compression"]["pruning_init"] = pruning_flops_target
    config["compression"]["params"]["pruning_flops_target"] = pruning_flops_target
    config["compression"]["params"]["prune_first_conv"] = True
    config["compression"]["params"]["all_weights"] = all_weights
    input_shape = [1, 8, 8, 1]
    model = model_fn(input_shape)
    model.compile()
    compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    assert compression_ctrl.full_flops == ref_full_flops
    assert compression_ctrl.full_params_num == ref_full_params
    assert compression_ctrl.current_flops == ref_current_flops
    assert compression_ctrl.current_params_num == ref_current_params

    original_graph = compression_ctrl._original_graph
    pruning_groups = compression_ctrl._pruned_layer_groups_info
    shape_pruner = ShapePruningProcessor(
        prunable_types=compression_ctrl._prunable_types, pruning_operations_metatype=TF_PRUNING_OPERATOR_METATYPES
    )

    next_nodes = shape_pruner.get_next_nodes(original_graph, pruning_groups)
    # Check output_shapes are empty in graph
    for node in original_graph.get_all_nodes():
        assert node.attributes["output_shape"] is None

    assert compression_ctrl._calculate_num_of_sparse_elements_by_node() == ref_num_of_sparse

    tmp_in_channels, tmp_out_channels = shape_pruner.calculate_in_out_channels_by_masks(
        graph=original_graph,
        pruning_groups=pruning_groups,
        pruning_groups_next_nodes=next_nodes,
        num_of_sparse_elements_by_node=ref_num_of_sparse,
    )

    output_shapes = collect_output_shapes(compression_ctrl.model, original_graph)
    flops_weights_calculator = WeightsFlopsCalculator(
        conv_op_metatypes=GENERAL_CONV_LAYER_METATYPES, linear_op_metatypes=LINEAR_LAYER_METATYPES
    )
    cur_flops, cur_params_num = flops_weights_calculator.count_flops_and_weights(
        graph=original_graph,
        output_shapes=output_shapes,
        input_channels=tmp_in_channels,
        output_channels=tmp_out_channels,
    )
    assert (cur_flops, cur_params_num) == (ref_current_flops, ref_current_params)

    all_clusters = compression_ctrl._pruned_layer_groups_info.get_all_clusters()
    assert len(all_clusters) == len(ref_sizes)
    for cluster, ref_size in zip(all_clusters, ref_sizes):
        for node in cluster.elements:
            layer = compressed_model.get_layer(node.layer_name)
            key = node.layer_name
            if node.is_depthwise:
                key += "_depthwise"
            key += "_kernel_pruning_binary_mask"
            mask = layer.ops_weights[key]["mask"]
            val = int(tf.reduce_sum(mask))
            assert val == ref_size[node.layer_name]


def test_maximal_compression_rate():
    """
    Test that we can set flops pruning target less or equal to maximal_compression_rate
    Test that we can't set flops pruning target higher than maximal_compression_rate
    """
    config = get_basic_pruning_config(8)
    config["compression"]["algorithm"] = "filter_pruning"
    config["compression"]["params"]["pruning_flops_target"] = 0.2
    config["compression"]["ignored_scopes"] = ["conv2^0"]

    model = get_test_model_shared_convs([1, 8, 8, 1])
    model.compile()
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    maximal_compression_rate = compression_ctrl.maximal_compression_rate
    for comp_rate in np.linspace(0, maximal_compression_rate, 10):
        compression_ctrl.compression_rate = comp_rate
    for comp_rate in np.linspace(maximal_compression_rate + 1e-5, 1, 10):
        with pytest.raises(nncf.ParameterNotSupportedError):
            compression_ctrl.compression_rate = comp_rate
