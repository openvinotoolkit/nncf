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

from nncf.tensorflow.graph.utils import collect_wrapped_layers
from nncf.tensorflow.pruning.utils import collect_output_shapes
from tests.tensorflow.helpers import create_compressed_model_and_algo_for_test
from tests.tensorflow.pruning.helpers import get_basic_pruning_config
from tests.tensorflow.pruning.helpers import get_batched_linear_model
from tests.tensorflow.pruning.helpers import get_broadcasted_linear_model
from tests.tensorflow.pruning.helpers import get_concat_test_model
from tests.tensorflow.pruning.helpers import get_diff_cluster_channels_model
from tests.tensorflow.pruning.helpers import get_model_depthwise_conv
from tests.tensorflow.pruning.helpers import get_model_grouped_convs
from tests.tensorflow.pruning.helpers import get_split_test_model
from tests.tensorflow.pruning.helpers import get_test_model_shared_convs


def check_pruning_mask(mask, pruning_level, layer_name):
    assert np.sum(mask) == mask.size * pruning_level, f"Incorrect masks for {layer_name}"


@pytest.mark.parametrize(
    ("all_weights", "prune_batch_norms", "ref_num_wrapped_layer"),
    [
        [True, True, 5],
        [False, True, 5],
        [True, False, 3],
        [False, False, 3],
    ],
)
def test_masks_in_concat_model(all_weights, prune_batch_norms, ref_num_wrapped_layer):
    config = get_basic_pruning_config(8)
    config["compression"]["params"]["all_weights"] = all_weights
    config["compression"]["params"]["prune_batch_norms"] = prune_batch_norms
    sample_size = [1, 8, 8, 3]
    model = get_concat_test_model(sample_size)

    model, _ = create_compressed_model_and_algo_for_test(model, config)
    wrapped_layers = collect_wrapped_layers(model)

    # Check number of wrapped layers
    assert len(wrapped_layers) == ref_num_wrapped_layer

    for layer in wrapped_layers:
        # Check existed weights of masks
        assert layer.ops_weights

        # Check masks correctness
        if not all_weights:
            target_pruning_level = 0.625 if layer.name == "bn_concat" else 0.5
            assert len(layer.ops_weights) == 2
            for op in layer.ops_weights.values():
                check_pruning_mask(op["mask"].numpy(), target_pruning_level, layer.name)


@pytest.mark.parametrize(
    ("all_weights", "ref_num_wrapped_layer"),
    [
        [True, 3],
        [False, 3],
    ],
)
def test_masks_in_split_model(all_weights, ref_num_wrapped_layer):
    config = get_basic_pruning_config(8)
    config["compression"]["params"]["all_weights"] = all_weights
    sample_size = [1, 8, 8, 3]
    model = get_split_test_model(sample_size)

    model, _ = create_compressed_model_and_algo_for_test(model, config)
    wrapped_layers = collect_wrapped_layers(model)

    # Check number of wrapped layers
    assert len(wrapped_layers) == ref_num_wrapped_layer

    for layer in wrapped_layers:
        # Check existed weights of masks
        assert layer.ops_weights

        # Check masks correctness
        if not all_weights:
            assert len(layer.ops_weights) == 2
            for op in layer.ops_weights.values():
                check_pruning_mask(op["mask"].numpy(), pruning_level=0.5, layer_name=layer.name)


@pytest.mark.parametrize(
    "model_fn,ref_output_shapes",
    [
        (
            get_test_model_shared_convs,
            {"conv1": (8, 8), "conv2^0": (6, 6), "conv2^1": (2, 2), "conv3^0": (6, 6), "conv3^1": (2, 2)},
        ),
        (
            get_model_grouped_convs,
            {"conv1": (8, 8), "conv2": (6, 6), "conv3": (4, 4), "conv4": (2, 2), "dense": (128,)},
        ),
        (
            get_model_depthwise_conv,
            {"conv1": (8, 8), "conv2": (6, 6), "conv3": (2, 2), "conv4": (4, 4), "dense": (128,)},
        ),
        (
            get_broadcasted_linear_model,
            {"first_conv": (8, 8), "conv1": (8, 8), "last_linear": (8, 8), "linear1": (16,)},
        ),
        (get_batched_linear_model, {"first_conv": (8, 8), "linear1": (8, 8, 16), "last_linear": (1,)}),
        (
            get_diff_cluster_channels_model,
            {"first_conv": (8, 8), "conv1": (8, 8), "linear1": (2048,), "last_linear": (16,)},
        ),
    ],
)
def test_collect_output_shapes(model_fn, ref_output_shapes):
    config = get_basic_pruning_config(8)
    input_shape = [1, 8, 8, 1]
    model = model_fn(input_shape)
    model.compile()
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    output_shapes = collect_output_shapes(compression_ctrl.model, compression_ctrl._original_graph)
    assert output_shapes == ref_output_shapes
