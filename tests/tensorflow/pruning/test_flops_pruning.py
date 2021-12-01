"""
 Copyright (c) 2021 Intel Corporation
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

from tests.tensorflow.helpers import create_compressed_model_and_algo_for_test
from tests.tensorflow.pruning.helpers import get_basic_pruning_config
from tests.tensorflow.pruning.helpers import get_model_grouped_convs
from tests.tensorflow.pruning.helpers import get_test_model_shared_convs


@pytest.mark.parametrize(
    ("model_fn", "all_weights", "pruning_flops_target", "ref_full_flops", "ref_current_flops",
     "ref_full_params", "ref_current_params"),
    (
        (get_test_model_shared_convs, True, 0.4, 461438976, 276858560, 11534848, 6920562),
        (get_test_model_shared_convs, False, 0.4, 461438976, 275300352, 11534848, 6881664),
        (get_model_grouped_convs, False, 0.0, 10859520, 10859520, 215808, 215808)
    )
)
def test_flops_calulation_for_spec_layers(model_fn, all_weights, pruning_flops_target,
                                          ref_full_flops, ref_current_flops,
                                          ref_full_params, ref_current_params):
    config = get_basic_pruning_config(8)
    config['compression']['algorithm'] = 'filter_pruning'
    config['compression']['pruning_init'] = pruning_flops_target
    config['compression']['params']['pruning_flops_target'] = pruning_flops_target
    config['compression']['params']['prune_first_conv'] = True
    config['compression']['params']['all_weights'] = all_weights
    input_shape = [1, 8, 8, 1]
    model = model_fn(input_shape)
    model.compile()
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    assert compression_ctrl.full_flops == ref_full_flops
    assert compression_ctrl.full_params_num == ref_full_params
    assert compression_ctrl.current_flops == ref_current_flops
    assert compression_ctrl.current_params_num == ref_current_params
