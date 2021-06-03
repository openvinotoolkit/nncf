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

import numpy as np

from nncf.torch.pruning.filter_pruning.algo import FilterPruningController
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import check_correct_nncf_modules_replacement
from tests.torch.pruning.helpers import get_basic_pruning_config
from tests.torch.pruning.helpers import BigPruningTestModel


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


@pytest.mark.parametrize(
    ('all_weights', 'pruning_rate_to_set', 'ref_pruning_rates', 'ref_global_pruning_rate'),
    [
        (False, 0.5, [0.5, 0.5], 0.5),
        (True, 0.5, [0.28125, 0.60937], 0.5),
        (False, {0: 0.6, 1: 0.8}, [0.5, 0.75], 0.69986),
    ]
)
def test_setting_pruning_rate(all_weights, pruning_rate_to_set, ref_pruning_rates, ref_global_pruning_rate):
    """
    Test setting global and groupwise pruning rates via the set_pruning_rate method.
    """
    # Creating algorithm with empty config
    config = get_basic_pruning_config(input_sample_size=[1, 1, 8, 8])
    config['compression']['pruning_init'] = 0.2
    config['compression']['params']['all_weights'] = all_weights

    _, pruning_controller, _ = create_pruning_algo_with_config(config)
    assert isinstance(pruning_controller, FilterPruningController)

    pruning_controller.set_pruning_rate(pruning_rate_to_set)
    groupwise_pruning_rates = list(pruning_controller.current_groupwise_pruning_rate.values())
    assert np.isclose(groupwise_pruning_rates, ref_pruning_rates).all()
    assert np.isclose(pruning_controller.pruning_rate, ref_global_pruning_rate).all()
