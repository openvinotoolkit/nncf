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

from nncf.dynamic_graph.context import Scope
from tests.helpers import create_compressed_model_and_algo_for_test
from tests.pruning.helpers import PruningTestModelEltwise, get_basic_pruning_config, TestModelBranching


@pytest.mark.parametrize("model,not_pruned_modules,pruned_groups,prune_first,prune_last,prune_downsample",
                         [(PruningTestModelEltwise,
                           ['PruningTestModelEltwise/NNCFConv2d[conv1]', 'PruningTestModelEltwise/NNCFConv2d[conv4]'],
                           [['PruningTestModelEltwise/NNCFConv2d[conv2]', 'PruningTestModelEltwise/NNCFConv2d[conv3]']],
                           False, False, False),
                          (PruningTestModelEltwise, [], [['PruningTestModelEltwise/NNCFConv2d[conv1]'],
                                                         ['PruningTestModelEltwise/NNCFConv2d[conv4]'],
                                                         ['PruningTestModelEltwise/NNCFConv2d[conv2]',
                                                          'PruningTestModelEltwise/NNCFConv2d[conv3]']], True, True,
                           False),
                          (TestModelBranching, [], [
                              ['TestModelBranching/NNCFConv2d[conv1]', 'TestModelBranching/NNCFConv2d[conv2]',
                               'TestModelBranching/NNCFConv2d[conv3]'],
                              ['TestModelBranching/NNCFConv2d[conv4]', 'TestModelBranching/NNCFConv2d[conv5]']], True,
                           True, False),
                          (TestModelBranching,
                           ['TestModelBranching/NNCFConv2d[conv1]', 'TestModelBranching/NNCFConv2d[conv2]',
                            'TestModelBranching/NNCFConv2d[conv3]'],
                           [['TestModelBranching/NNCFConv2d[conv4]', 'TestModelBranching/NNCFConv2d[conv5]']], False,
                           True, False),
                          (TestModelBranching,
                           ['TestModelBranching/NNCFConv2d[conv4]', 'TestModelBranching/NNCFConv2d[conv5]'],
                           [['TestModelBranching/NNCFConv2d[conv1]', 'TestModelBranching/NNCFConv2d[conv2]',
                             'TestModelBranching/NNCFConv2d[conv3]']], True, False, False),
                          ])
def test_groups(model, not_pruned_modules, pruned_groups, prune_first, prune_last, prune_downsample):
    model = model()
    nncf_config = get_basic_pruning_config(input_sample_size=[1, 1, 8, 8])
    nncf_config['compression']['algorithm'] = 'filter_pruning'
    nncf_config['compression']['params']['prune_first_conv'] = prune_first
    nncf_config['compression']['params']['prune_last_conv'] = prune_last

    compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, nncf_config)

    # 1. Check all not pruned modules
    clusters = compression_ctrl.pruned_module_groups_info
    all_pruned_modules_info = clusters.get_all_nodes()
    all_pruned_modules = [info.module for info in all_pruned_modules_info]
    print([minfo.module_name for minfo in all_pruned_modules_info])
    for module_name in not_pruned_modules:
        module = compressed_model.get_module_by_scope(Scope.from_str(module_name))
        assert module not in all_pruned_modules

    # 2. Check that all pruned groups are valid
    for group in pruned_groups:
        first_node_scope_name = group[0]
        cluster = clusters.get_cluster_for_node(first_node_scope_name)
        cluster_modules = [n.module for n in cluster.nodes]
        group_modules = [compressed_model.get_module_by_scope(Scope.from_str(module_scope)) for module_scope in group]

        assert cluster_modules == group_modules
