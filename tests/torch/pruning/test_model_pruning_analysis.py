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
from typing import Callable

import pytest
from torch import nn

from nncf.torch.layers import NNCF_PRUNING_MODULES_DICT
from nncf.torch.dynamic_graph.graph_tracer import ModelInputInfo
from nncf.torch.dynamic_graph.context import Scope
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.pruning.export_helpers import PTIdentityMaskForwardOps
from nncf.torch.pruning.export_helpers import PT_PRUNING_OPERATOR_METATYPES
from nncf.torch.pruning.export_helpers import PTElementwise
from nncf.torch.pruning.utils import is_depthwise_conv
from nncf.common.pruning.pruning_node_selector import PruningNodeSelector
from nncf.common.pruning.model_analysis import NodesCluster
from nncf.common.pruning.model_analysis import Clusterization
from nncf.common.pruning.model_analysis import cluster_special_ops
from nncf.common.pruning.model_analysis import ModelAnalyzer
from tests.torch.helpers import create_compressed_model_and_algo_for_test, create_nncf_model_and_algo_builder, \
    module_scope_from_node_name
from tests.torch.pruning.helpers import PruningTestModelEltwise, get_basic_pruning_config, TestModelBranching, \
    TestModelResidualConnection, TestModelEltwiseCombination, TestModelDiffConvs, \
    TestModelShuffleNetUnit, TestModelShuffleNetUnitDW, PruningTestModelSharedConvs


# pylint: disable=protected-access
def create_nncf_model_and_builder(model, config_params):
    nncf_config = get_basic_pruning_config(input_sample_size=[1, 1, 8, 8])
    nncf_config['compression']['algorithm'] = 'filter_pruning'
    for key, value in config_params.items():
        nncf_config['compression']['params'][key] = value
    nncf_model, composite_builder = create_nncf_model_and_algo_builder(model, nncf_config)

    assert len(composite_builder.child_builders) == 1
    algo_builder = composite_builder.child_builders[0]
    return nncf_model, algo_builder


class GroupPruningModulesTestStruct:
    def __init__(self, model, not_pruned_nodes, pruned_groups, pruned_groups_by_node_id, prune_params):
        self.model = model
        self.not_pruned_nodes = not_pruned_nodes
        self.pruned_groups = pruned_groups
        self.pruned_groups_by_node_id = pruned_groups_by_node_id
        self.prune_params = prune_params


GROUP_PRUNING_MODULES_TEST_CASES = [
    GroupPruningModulesTestStruct(model=PruningTestModelEltwise,
                                  not_pruned_nodes=['1 PruningTestModelEltwise/NNCFConv2d[conv1]/conv2d_0',
                                                    '7 PruningTestModelEltwise/NNCFConv2d[conv4]/conv2d_0'],
                                  pruned_groups=[['3 PruningTestModelEltwise/NNCFConv2d[conv2]/conv2d_0',
                                                  '4 PruningTestModelEltwise/NNCFConv2d[conv3]/conv2d_0']],
                                  pruned_groups_by_node_id=[[3, 4]],
                                  prune_params=(False, False, False)),
    GroupPruningModulesTestStruct(model=PruningTestModelEltwise,
                                  not_pruned_nodes=[],
                                  pruned_groups=[['1 PruningTestModelEltwise/NNCFConv2d[conv1]/conv2d_0'],
                                                 ['7 PruningTestModelEltwise/NNCFConv2d[conv4]/conv2d_0'],
                                                 ['3 PruningTestModelEltwise/NNCFConv2d[conv2]/conv2d_0',
                                                  '4 PruningTestModelEltwise/NNCFConv2d[conv3]/conv2d_0']],
                                  pruned_groups_by_node_id=[[1], [7], [3, 4]],
                                  prune_params=(True, True, False)),
    GroupPruningModulesTestStruct(model=TestModelBranching,
                                  not_pruned_nodes=[],
                                  pruned_groups=[['1 TestModelBranching/NNCFConv2d[conv1]/conv2d_0',
                                                  '2 TestModelBranching/NNCFConv2d[conv2]/conv2d_0',
                                                  '4 TestModelBranching/NNCFConv2d[conv3]/conv2d_0']],
                                  pruned_groups_by_node_id=[[1, 2, 4]],
                                  prune_params=(True, True, False)),
    GroupPruningModulesTestStruct(model=TestModelBranching,
                                  not_pruned_nodes=['1 TestModelBranching/NNCFConv2d[conv1]/conv2d_0',
                                                    '2 TestModelBranching/NNCFConv2d[conv2]/conv2d_0',
                                                    '4 TestModelBranching/NNCFConv2d[conv3]/conv2d_0'],
                                  pruned_groups=[['7 TestModelBranching/NNCFConv2d[conv4]/conv2d_0',
                                                  '8 TestModelBranching/NNCFConv2d[conv5]/conv2d_0']],
                                  pruned_groups_by_node_id=[[7, 8]],
                                  prune_params=(False, True, False)),
    GroupPruningModulesTestStruct(model=TestModelBranching,
                                  not_pruned_nodes=['7 TestModelBranching/NNCFConv2d[conv4]/conv2d_0',
                                                    '8 TestModelBranching/NNCFConv2d[conv5]/conv2d_0'],
                                  pruned_groups=[['1 TestModelBranching/NNCFConv2d[conv1]/conv2d_0',
                                                  '2 TestModelBranching/NNCFConv2d[conv2]/conv2d_0',
                                                  '4 TestModelBranching/NNCFConv2d[conv3]/conv2d_0']],
                                  pruned_groups_by_node_id=[[1, 2, 4]],
                                  prune_params=(True, False, False)),
    GroupPruningModulesTestStruct(model=TestModelResidualConnection,
                                  not_pruned_nodes=['7 TestModelResidualConnection/NNCFConv2d[conv4]/conv2d_0',
                                                    '8 TestModelResidualConnection/NNCFConv2d[conv5]/conv2d_0'],
                                  pruned_groups=[['1 TestModelResidualConnection/NNCFConv2d[conv1]/conv2d_0',
                                                  '2 TestModelResidualConnection/NNCFConv2d[conv2]/conv2d_0',
                                                  '4 TestModelResidualConnection/NNCFConv2d[conv3]/conv2d_0']],
                                  pruned_groups_by_node_id=[[1, 2, 4]],
                                  prune_params=(True, True, False)),
    GroupPruningModulesTestStruct(model=TestModelEltwiseCombination,
                                  not_pruned_nodes=[],
                                  pruned_groups=[['1 TestModelEltwiseCombination/NNCFConv2d[conv1]/conv2d_0',
                                                  '2 TestModelEltwiseCombination/NNCFConv2d[conv2]/conv2d_0',
                                                  '4 TestModelEltwiseCombination/NNCFConv2d[conv3]/conv2d_0',
                                                  '6 TestModelEltwiseCombination/NNCFConv2d[conv4]/conv2d_0'],
                                                 ['8 TestModelEltwiseCombination/NNCFConv2d[conv5]/conv2d_0',
                                                  '9 TestModelEltwiseCombination/NNCFConv2d[conv6]/conv2d_0']],
                                  pruned_groups_by_node_id=[[1, 2, 4, 6], [8, 9]],
                                  prune_params=(True, True, False)),
    GroupPruningModulesTestStruct(model=PruningTestModelSharedConvs,
                                  not_pruned_nodes=['1 PruningTestModelSharedConvs/NNCFConv2d[conv1]/conv2d_0',
                                                    '5 PruningTestModelSharedConvs/NNCFConv2d[conv3]/conv2d_0',
                                                    '6 PruningTestModelSharedConvs/NNCFConv2d[conv3]/conv2d_1'],
                                  pruned_groups=[['3 PruningTestModelSharedConvs/NNCFConv2d[conv2]/conv2d_0',
                                                  '4 PruningTestModelSharedConvs/NNCFConv2d[conv2]/conv2d_1']],
                                  pruned_groups_by_node_id=[[3, 4]],
                                  prune_params=(False, False, False))
]


@pytest.fixture(params=GROUP_PRUNING_MODULES_TEST_CASES, name='test_input_info_struct_')
def test_input_info_struct(request):
    return request.param


def test_groups(test_input_info_struct_: GroupPruningModulesTestStruct):
    model = test_input_info_struct_.model
    not_pruned_nodes = test_input_info_struct_.not_pruned_nodes
    pruned_groups = test_input_info_struct_.pruned_groups
    prune_first, prune_last, prune_downsample = test_input_info_struct_.prune_params

    model = model()
    nncf_config = get_basic_pruning_config(input_sample_size=[1, 1, 8, 8])
    nncf_config['compression']['algorithm'] = 'filter_pruning'
    nncf_config['compression']['params']['prune_first_conv'] = prune_first
    nncf_config['compression']['params']['prune_last_conv'] = prune_last
    nncf_config['compression']['params']['prune_downsample_convs'] = prune_downsample

    compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, nncf_config)

    # 1. Check all not pruned modules
    clusters = compression_ctrl.pruned_module_groups_info
    all_pruned_modules_info = clusters.get_all_nodes()
    all_pruned_modules = [info.module for info in all_pruned_modules_info]
    print([minfo.module_scope for minfo in all_pruned_modules_info])
    for node_name in not_pruned_nodes:
        module = compressed_model.get_module_by_scope(module_scope_from_node_name(node_name))
        assert module is not None and module not in all_pruned_modules

    # 2. Check that all pruned groups are valid
    for group in pruned_groups:
        first_node_scope = group[0]
        cluster = clusters.get_cluster_by_node_id(first_node_scope)
        cluster_modules = [n.module for n in cluster.nodes]
        group_modules = [compressed_model.get_module_by_scope(module_scope_from_node_name(node_name))
                         for node_name in group]

        assert cluster_modules == group_modules


def test_pruning_node_selector(test_input_info_struct_: GroupPruningModulesTestStruct):
    model = test_input_info_struct_.model
    not_pruned_nodes = test_input_info_struct_.not_pruned_nodes
    pruned_groups_by_node_id = test_input_info_struct_.pruned_groups_by_node_id
    prune_first, prune_last, prune_downsample = test_input_info_struct_.prune_params

    pruning_operations = [v.op_func_name for v in NNCF_PRUNING_MODULES_DICT]
    grouping_operations = PTElementwise.get_all_op_aliases()
    pruning_node_selector = PruningNodeSelector(PT_PRUNING_OPERATOR_METATYPES,
                                                pruning_operations,
                                                grouping_operations,
                                                None,
                                                None,
                                                prune_first,
                                                prune_last,
                                                prune_downsample)
    model = model()
    model.eval()
    nncf_network = NNCFNetwork(model, input_infos=[ModelInputInfo([1, 1, 8, 8])])
    graph = nncf_network.get_original_graph()
    pruning_groups = pruning_node_selector.create_pruning_groups(graph)

    # 1. Check all not pruned modules
    all_pruned_nodes = pruning_groups.get_all_nodes()
    all_pruned_modules = [nncf_network.get_module_by_scope(node.ia_op_exec_context.scope_in_model)
                          for node in all_pruned_nodes]
    for node_name in not_pruned_nodes:
        module = nncf_network.get_module_by_scope(module_scope_from_node_name(node_name))
        assert module is not None and module not in all_pruned_modules

    # 2. Check that all pruned groups are valid
    for group_by_id in pruned_groups_by_node_id:
        first_node_id = group_by_id[0]
        cluster = pruning_groups.get_cluster_by_node_id(first_node_id)
        cluster_node_ids = [n.node_id for n in cluster.nodes]
        cluster_node_ids.sort()

        assert cluster_node_ids == group_by_id

class GroupSpecialModulesTestStruct:
    def __init__(self, model: Callable, eltwise_clusters):
        self.model = model
        self.eltwise_clusters = eltwise_clusters


GROUP_SPECIAL_MODULES_TEST_CASES = [
    GroupSpecialModulesTestStruct(
        model=TestModelBranching,
        eltwise_clusters=[[3, 5], [9]],
    ),
    GroupSpecialModulesTestStruct(
        model=TestModelResidualConnection,
        eltwise_clusters=[[3, 5], [9]],
    ),
    GroupSpecialModulesTestStruct(
        model=TestModelEltwiseCombination,
        eltwise_clusters=[[3, 5, 7], [10]]
    )
]


@pytest.fixture(params=GROUP_SPECIAL_MODULES_TEST_CASES, name='test_special_ops_struct')
def test_special_ops_struct_(request):
    return request.param


def test_group_special_nodes(test_special_ops_struct: GroupSpecialModulesTestStruct):
    model = test_special_ops_struct.model()
    nncf_model, algo_builder = create_nncf_model_and_builder(model, {'prune_first_conv': True, 'prune_last_conv': True})

    special_ops_clusterization = cluster_special_ops(nncf_model.get_original_graph(),
                                                     algo_builder.get_types_of_grouping_ops(),
                                                     PTIdentityMaskForwardOps.get_all_op_aliases())

    for ref_cluster in test_special_ops_struct.eltwise_clusters:
        cluster = special_ops_clusterization.get_cluster_by_node_id(ref_cluster[0])
        assert sorted([node.node_id for node in cluster.nodes]) == sorted(ref_cluster)


class ModelAnalyserTestStruct:
    def __init__(self, model: nn.Module, ref_can_prune: dict):
        self.model = model
        self.ref_can_prune = ref_can_prune


MODEL_ANALYSER_TEST_CASES = [
    ModelAnalyserTestStruct(
        model=TestModelResidualConnection,
        ref_can_prune={0: True, 1: True, 2: True, 3: True, 4: True, 5: True, 6: True, 7: False, 8: False, 9: False,
                       10: False, 11: False, 12: False}
    )
]


@pytest.fixture(params=MODEL_ANALYSER_TEST_CASES, name='test_struct')
def test_struct_(request):
    return request.param


def test_model_analyzer(test_struct: GroupSpecialModulesTestStruct):
    model = test_struct.model()
    nncf_model, _ = create_nncf_model_and_builder(model, {'prune_first_conv': True, 'prune_last_conv': True})

    model_analyser = ModelAnalyzer(nncf_model.get_original_graph(), PT_PRUNING_OPERATOR_METATYPES, is_depthwise_conv)
    can_prune_analysis = model_analyser.analyse_model_before_pruning()
    for node_id in can_prune_analysis.keys():
        assert can_prune_analysis[node_id] == test_struct.ref_can_prune[node_id]


class ModulePrunableTestStruct:
    def __init__(self, model: nn.Module, config_params: dict, is_module_prunable: dict):
        self.model = model
        self.config_params = config_params
        self.is_module_prunable = is_module_prunable


IS_MODULE_PRUNABLE_TEST_CASES = [
    ModulePrunableTestStruct(
        model=TestModelDiffConvs,
        config_params={},
        is_module_prunable={'TestModelDiffConvs/NNCFConv2d[conv1]': False,
                            'TestModelDiffConvs/NNCFConv2d[conv2]': True,
                            'TestModelDiffConvs/NNCFConv2d[conv3]': False,
                            'TestModelDiffConvs/NNCFConv2d[conv4]': False},
    ),
    ModulePrunableTestStruct(
        model=TestModelDiffConvs,
        config_params={'prune_first_conv': True, 'prune_last_conv': True},
        is_module_prunable={'TestModelDiffConvs/NNCFConv2d[conv1]': True,
                            'TestModelDiffConvs/NNCFConv2d[conv2]': True,
                            'TestModelDiffConvs/NNCFConv2d[conv3]': False,
                            'TestModelDiffConvs/NNCFConv2d[conv4]': False},
    ),
    ModulePrunableTestStruct(
        model=TestModelDiffConvs,
        config_params={'prune_first_conv': True, 'prune_last_conv': True, 'prune_downsample_convs': True},
        is_module_prunable={'TestModelDiffConvs/NNCFConv2d[conv1]': True,
                            'TestModelDiffConvs/NNCFConv2d[conv2]': True,
                            'TestModelDiffConvs/NNCFConv2d[conv3]': True,
                            'TestModelDiffConvs/NNCFConv2d[conv4]': False},
    ),
    ModulePrunableTestStruct(
        model=TestModelBranching,
        config_params={},
        is_module_prunable={'TestModelBranching/NNCFConv2d[conv1]': False,
                            'TestModelBranching/NNCFConv2d[conv2]': False,
                            'TestModelBranching/NNCFConv2d[conv3]': False,
                            'TestModelBranching/NNCFConv2d[conv4]': False,
                            'TestModelBranching/NNCFConv2d[conv5]': False},
    ),
    ModulePrunableTestStruct(
        model=TestModelBranching,
        config_params={'prune_first_conv': True, 'prune_last_conv': True, },
        is_module_prunable={'TestModelBranching/NNCFConv2d[conv1]': True,
                            'TestModelBranching/NNCFConv2d[conv2]': True,
                            'TestModelBranching/NNCFConv2d[conv3]': True,
                            'TestModelBranching/NNCFConv2d[conv4]': True,
                            'TestModelBranching/NNCFConv2d[conv5]': True},
    ),
    ModulePrunableTestStruct(
        model=TestModelShuffleNetUnitDW,
        config_params={'prune_first_conv': True, 'prune_last_conv': True, },
        is_module_prunable={'TestModelShuffleNetUnitDW/NNCFConv2d[conv]': True,
                            'TestModelShuffleNetUnitDW/TestShuffleUnit[unit1]/NNCFConv2d[dw_conv4]': False,
                            'TestModelShuffleNetUnitDW/TestShuffleUnit[unit1]/NNCFConv2d[expand_conv5]': True,
                            'TestModelShuffleNetUnitDW/TestShuffleUnit[unit1]/NNCFConv2d[compress_conv1]': True,
                            'TestModelShuffleNetUnitDW/TestShuffleUnit[unit1]/NNCFConv2d[dw_conv2]': False,
                            'TestModelShuffleNetUnitDW/TestShuffleUnit[unit1]/NNCFConv2d[expand_conv3]': True},
    ),
    ModulePrunableTestStruct(
        model=TestModelShuffleNetUnit,
        config_params={'prune_first_conv': True, 'prune_last_conv': True, },
        is_module_prunable={'TestModelShuffleNetUnit/NNCFConv2d[conv]': True,
                            'TestModelShuffleNetUnit/TestShuffleUnit[unit1]/NNCFConv2d[compress_conv1]': True,
                            'TestModelShuffleNetUnit/TestShuffleUnit[unit1]/NNCFConv2d[dw_conv2]': True,
                            'TestModelShuffleNetUnit/TestShuffleUnit[unit1]/NNCFConv2d[expand_conv3]': True},
    )
]


@pytest.fixture(params=IS_MODULE_PRUNABLE_TEST_CASES, name='test_prunable_struct')
def test_prunable_struct_(request):
    return request.param


def test_is_module_prunable(test_prunable_struct: ModulePrunableTestStruct):
    model = test_prunable_struct.model()
    nncf_model, algo_builder = create_nncf_model_and_builder(model, test_prunable_struct.config_params)
    graph = nncf_model.get_original_graph()
    for module_scope_str in test_prunable_struct.is_module_prunable:
        scope = Scope.from_str(module_scope_str)
        nncf_node = graph.find_node_in_nx_graph_by_scope(scope)
        is_prunable, _ = algo_builder.pruning_node_selector._is_module_prunable(graph, nncf_node)
        assert is_prunable == test_prunable_struct.is_module_prunable[module_scope_str]


class SimpleNode:
    def __init__(self, id_):
        self.id = id_


def test_nodes_cluster():
    # test creating
    cluster_id = 0
    nodes = [SimpleNode(0)]
    nodes_orders = [0]
    cluster = NodesCluster(cluster_id, nodes, nodes_orders)
    assert cluster.id == cluster_id
    assert cluster.nodes == nodes
    assert cluster.importance == max(nodes_orders)

    # test add nodes
    new_nodes = [SimpleNode(1), SimpleNode(2)]
    new_importance = 4
    cluster.add_nodes(new_nodes, new_importance)
    assert cluster.importance == new_importance
    assert cluster.nodes == nodes + new_nodes

    # test clean
    cluster.clean_cluster()
    assert cluster.nodes == []
    assert cluster.importance == 0


def test_clusterization():
    nodes_1 = [SimpleNode(0), SimpleNode(1)]
    nodes_2 = [SimpleNode(2), SimpleNode(3)]
    cluster_1 = NodesCluster(1, nodes_1, [node.id for node in nodes_1])
    cluster_2 = NodesCluster(2, nodes_2, [node.id for node in nodes_2])

    clusterization = Clusterization()

    # test adding of clusters
    clusterization.add_cluster(cluster_1)
    assert 1 in clusterization.clusters
    assert 0 in clusterization._node_to_cluster and 1 in clusterization._node_to_cluster

    # test get_cluster_by_id
    assert clusterization.get_cluster_by_id(1) == cluster_1
    with pytest.raises(IndexError) as err:
        clusterization.get_cluster_by_id(5)
    assert 'No cluster with id' in str(err.value)

    # test get_cluster_by_node_id
    assert clusterization.get_cluster_by_node_id(1) == cluster_1
    with pytest.raises(IndexError) as err:
        clusterization.get_cluster_by_node_id(10)
    assert 'No cluster for node' in str(err.value)

    # test deleting
    clusterization.delete_cluster(1)
    with pytest.raises(IndexError) as err:
        clusterization.get_cluster_by_id(1)

    clusterization.add_cluster(cluster_1)
    clusterization.add_cluster(cluster_2)

    # test get_all_clusters
    assert clusterization.get_all_clusters() == [cluster_1, cluster_2]

    # test get_all_nodes
    assert clusterization.get_all_nodes() == nodes_1 + nodes_2

    # test merge clusters
    clusterization.merge_clusters(1, 2)
    assert 2 in clusterization.clusters
    with pytest.raises(IndexError) as err:
        clusterization.get_cluster_by_id(1)

    assert set(clusterization.get_all_nodes()) == set(nodes_1 + nodes_2)
