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

import numpy as np
import pytest

from tests.tensorflow.helpers import create_compressed_model_and_algo_for_test
from nncf.tensorflow.graph.utils import collect_wrapped_layers
from tests.tensorflow.pruning.helpers import get_basic_pruning_config
from tests.tensorflow.pruning.helpers import get_concat_test_model


def check_pruning_mask(mask, pruning_rate, layer_name):
    assert np.sum(mask) == mask.size * pruning_rate, f"Incorrect masks for {layer_name}"


@pytest.mark.parametrize(('all_weights', 'prune_batch_norms', 'ref_num_wrapped_layer'),
                         [
                             [True, True, 5],
                             [False, True, 5],
                             [True, False, 3],
                             [False, False, 3],
                         ])
def test_masks_in_concat_model(all_weights, prune_batch_norms, ref_num_wrapped_layer):
    config = get_basic_pruning_config(8)
    config['compression']['params']['all_weights'] = all_weights
    config['compression']['params']['prune_batch_norms'] = prune_batch_norms
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
            target_pruning_rate = 0.625 if layer.name == 'bn_concat' else 0.5
            assert len(layer.ops_weights) == 2
            for op in layer.ops_weights.values():
                check_pruning_mask(op['mask'].numpy(), target_pruning_rate, layer.name)
