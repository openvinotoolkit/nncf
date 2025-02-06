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


from collections import Counter
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import pytest
import torch
from torch import nn

from nncf.common.graph import NNCFNodeName
from nncf.common.pruning.clusterization import Cluster
from nncf.common.pruning.clusterization import Clusterization
from nncf.common.pruning.mask_propagation import MaskPropagationAlgorithm
from nncf.common.pruning.model_analysis import ModelAnalyzer
from nncf.common.pruning.model_analysis import cluster_special_ops
from nncf.common.pruning.symbolic_mask import SymbolicMaskProcessor
from nncf.common.pruning.utils import PruningAnalysisDecision
from nncf.common.pruning.utils import PruningAnalysisReason
from nncf.torch.dynamic_graph.io_handling import FillerInputElement
from nncf.torch.dynamic_graph.io_handling import FillerInputInfo
from nncf.torch.layers import NNCF_PRUNING_MODULES_DICT
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.pruning.filter_pruning.algo import FilterPruningBuilder
from nncf.torch.pruning.operations import PT_PRUNING_OPERATOR_METATYPES
from nncf.torch.pruning.operations import PTConvolutionPruningOp
from nncf.torch.pruning.operations import PTElementwisePruningOp
from nncf.torch.pruning.operations import PTIdentityMaskForwardPruningOp
from nncf.torch.pruning.operations import PTLinearPruningOp
from nncf.torch.pruning.operations import PTTransposeConvolutionPruningOp
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import create_nncf_model_and_single_algo_builder
from tests.torch.pruning.helpers import BranchingModel
from tests.torch.pruning.helpers import DepthwiseConvolutionModel
from tests.torch.pruning.helpers import DiffConvsModel
from tests.torch.pruning.helpers import EltwiseCombinationModel
from tests.torch.pruning.helpers import GroupNormModel
from tests.torch.pruning.helpers import HRNetBlock
from tests.torch.pruning.helpers import MobilenetV3BlockSEReshape
from tests.torch.pruning.helpers import MultipleDepthwiseConvolutionModel
from tests.torch.pruning.helpers import MultipleSplitConcatModel
from tests.torch.pruning.helpers import NASnetBlock
from tests.torch.pruning.helpers import PruningTestBatchedLinear
from tests.torch.pruning.helpers import PruningTestMeanMetatype
from tests.torch.pruning.helpers import PruningTestModelBroadcastedLinearWithConcat
from tests.torch.pruning.helpers import PruningTestModelDiffChInPruningCluster
from tests.torch.pruning.helpers import PruningTestModelEltwise
from tests.torch.pruning.helpers import PruningTestModelPad
from tests.torch.pruning.helpers import PruningTestModelSharedConvs
from tests.torch.pruning.helpers import PruningTestModelSimplePrunableLinear
from tests.torch.pruning.helpers import PruningTestModelWrongDims
from tests.torch.pruning.helpers import PruningTestModelWrongDimsElementwise
from tests.torch.pruning.helpers import ResidualConnectionModel
from tests.torch.pruning.helpers import ShuffleNetUnitModel
from tests.torch.pruning.helpers import ShuffleNetUnitModelDW
from tests.torch.pruning.helpers import SplitConcatModel
from tests.torch.pruning.helpers import SplitIdentityModel
from tests.torch.pruning.helpers import SplitMaskPropFailModel
from tests.torch.pruning.helpers import SplitPruningInvalidModel
from tests.torch.pruning.helpers import SplitReshapeModel
from tests.torch.pruning.helpers import get_basic_pruning_config


def create_nncf_model_and_pruning_builder(
    model: torch.nn.Module, config_params: Dict
) -> Tuple[NNCFNetwork, FilterPruningBuilder]:
    nncf_config = get_basic_pruning_config(input_sample_size=[1, 1, 8, 8])
    nncf_config["compression"]["algorithm"] = "filter_pruning"
    for key, value in config_params.items():
        nncf_config["compression"]["params"][key] = value
    nncf_model, pruning_builder = create_nncf_model_and_single_algo_builder(model, nncf_config)
    return nncf_model, pruning_builder


class GroupPruningModulesTestStruct:
    def __init__(
        self,
        model: Union[Type[torch.nn.Module], Callable[[], torch.nn.Module]],
        non_pruned_module_nodes: List[NNCFNodeName],
        pruned_groups: List[List[NNCFNodeName]],
        pruned_groups_by_node_id: List[List[int]],
        can_prune_after_analysis: Dict[int, bool],
        final_can_prune: Dict[int, PruningAnalysisDecision],
        prune_params: Tuple[bool, bool],
        name: Optional[str] = None,
    ):
        self.model = model
        self.non_pruned_module_nodes = non_pruned_module_nodes
        self.pruned_groups = pruned_groups
        self.pruned_groups_by_node_id = pruned_groups_by_node_id
        self.can_prune_after_analysis = can_prune_after_analysis
        self.final_can_prune = final_can_prune
        self.prune_params = prune_params  # Prune first, Prune downsample
        self.name = name

    def __str__(self):
        if hasattr(self.model, "__name__"):
            return self.model.__name__
        assert self.name, "Can't define name from the model (usually due to partial), please specify it explicitly"
        return self.name


GROUP_PRUNING_MODULES_TEST_CASES = [
    GroupPruningModulesTestStruct(model=PruningTestModelEltwise,
                                  non_pruned_module_nodes=['PruningTestModelEltwise/NNCFConv2d[conv1]/conv2d_0',
                                                           'PruningTestModelEltwise/NNCFConv2d[conv4]/conv2d_0'],
                                  pruned_groups=[['PruningTestModelEltwise/NNCFConv2d[conv2]/conv2d_0',
                                                  'PruningTestModelEltwise/NNCFConv2d[conv3]/conv2d_0']],
                                  pruned_groups_by_node_id=[[3, 4]],
                                  can_prune_after_analysis={0: True, 1: False, 2: True, 3: True,
                                                            4: True, 5: True, 6: True, 7: True, 8: True},
                                  final_can_prune={1: PruningAnalysisDecision(
                                                          False, [PruningAnalysisReason.CLOSING_CONV_MISSING]),
                                                   3: PruningAnalysisDecision(True), 4: PruningAnalysisDecision(True),
                                                   7: PruningAnalysisDecision(
                                                          False, [PruningAnalysisReason.LAST_CONV])},
                                  prune_params=(False, False)),
    GroupPruningModulesTestStruct(model=PruningTestModelEltwise,
                                  non_pruned_module_nodes=['PruningTestModelEltwise/NNCFConv2d[conv4]/conv2d_0'],
                                  pruned_groups=[['PruningTestModelEltwise/NNCFConv2d[conv1]/conv2d_0'],
                                                 ['PruningTestModelEltwise/NNCFConv2d[conv2]/conv2d_0',
                                                  'PruningTestModelEltwise/NNCFConv2d[conv3]/conv2d_0']],
                                  pruned_groups_by_node_id=[[1], [3, 4]],
                                  can_prune_after_analysis={0: True, 1: True, 2: True, 3: True, 4: True,
                                                            5: True, 6: True, 7: True, 8: True},
                                  final_can_prune={1: PruningAnalysisDecision(True),
                                                   3: PruningAnalysisDecision(True),
                                                   4: PruningAnalysisDecision(True),
                                                   7: PruningAnalysisDecision(
                                                          False, [PruningAnalysisReason.LAST_CONV])},
                                  prune_params=(True, False)),
    GroupPruningModulesTestStruct(model=BranchingModel,
                                  non_pruned_module_nodes=[],
                                  pruned_groups=[['BranchingModel/NNCFConv2d[conv1]/conv2d_0',
                                                  'BranchingModel/NNCFConv2d[conv2]/conv2d_0',
                                                  'BranchingModel/NNCFConv2d[conv3]/conv2d_0']],
                                  pruned_groups_by_node_id=[[1, 2, 4]],
                                  can_prune_after_analysis={0: True, 1: True, 2: True, 3: True, 4: True,
                                                            5: True, 6: True, 7: True, 8: True, 9: True, 10: True},
                                  final_can_prune={1: PruningAnalysisDecision(True),
                                                   2: PruningAnalysisDecision(True),
                                                   4: PruningAnalysisDecision(True),
                                                   7: PruningAnalysisDecision(
                                                          False, [PruningAnalysisReason.LAST_CONV]),
                                                   8: PruningAnalysisDecision(
                                                          False, [PruningAnalysisReason.LAST_CONV])},
                                  prune_params=(True, False)),
    GroupPruningModulesTestStruct(model=BranchingModel,
                                  non_pruned_module_nodes=[
                                                      'BranchingModel/NNCFConv2d[conv1]/conv2d_0',
                                                      'BranchingModel/NNCFConv2d[conv2]/conv2d_0',
                                                      'BranchingModel/NNCFConv2d[conv3]/conv2d_0',
                                                      'BranchingModel/NNCFConv2d[conv4]/conv2d_0',
                                                      'BranchingModel/NNCFConv2d[conv5]/conv2d_0'],
                                  pruned_groups=[],
                                  pruned_groups_by_node_id=[],
                                  can_prune_after_analysis={0: True, 1: False, 2: False, 3: True, 4: False,
                                                            5: True, 6: True, 7: True, 8: True, 9: True, 10: True},
                                  final_can_prune={1: PruningAnalysisDecision(
                                                          False, [PruningAnalysisReason.CLOSING_CONV_MISSING]),
                                                   2: PruningAnalysisDecision(
                                                           False, [PruningAnalysisReason.CLOSING_CONV_MISSING]),
                                                   4: PruningAnalysisDecision(
                                                           False, [PruningAnalysisReason.CLOSING_CONV_MISSING]),
                                                   7: PruningAnalysisDecision(
                                                           False, [PruningAnalysisReason.LAST_CONV]),
                                                   8: PruningAnalysisDecision(
                                                           False, [PruningAnalysisReason.LAST_CONV])},
                                  prune_params=(False, False)),
    GroupPruningModulesTestStruct(model=BranchingModel,
                                  non_pruned_module_nodes=['BranchingModel/NNCFConv2d[conv4]/conv2d_0',
                                                      'BranchingModel/NNCFConv2d[conv5]/conv2d_0'],
                                  pruned_groups=[['BranchingModel/NNCFConv2d[conv1]/conv2d_0',
                                                  'BranchingModel/NNCFConv2d[conv2]/conv2d_0',
                                                  'BranchingModel/NNCFConv2d[conv3]/conv2d_0']],
                                  pruned_groups_by_node_id=[[1, 2, 4]],
                                  can_prune_after_analysis={0: True, 1: True, 2: True, 3: True, 4: True,
                                                            5: True, 6: True, 7: True, 8: True, 9: True, 10: True},
                                  final_can_prune={1: PruningAnalysisDecision(True),
                                                   2: PruningAnalysisDecision(True),
                                                   4: PruningAnalysisDecision(True),
                                                   7: PruningAnalysisDecision(
                                                          False, [PruningAnalysisReason.LAST_CONV]),
                                                   8: PruningAnalysisDecision(
                                                          False, [PruningAnalysisReason.LAST_CONV])},
                                  prune_params=(True, False)),
    GroupPruningModulesTestStruct(model=partial(ResidualConnectionModel, last_layer_accept_pruning=False),
                                  name='ResidualConnectionModel with not pruned last layer',
                                  non_pruned_module_nodes=['ResidualConnectionModel/NNCFLinear[linear]/linear_0',
                                                           'ResidualConnectionModel/NNCFConv2d[conv4]/conv2d_0',
                                                           'ResidualConnectionModel/NNCFConv2d[conv5]/conv2d_0'],
                                  pruned_groups=[['ResidualConnectionModel/NNCFConv2d[conv1]/conv2d_0',
                                                  'ResidualConnectionModel/NNCFConv2d[conv2]/conv2d_0',
                                                  'ResidualConnectionModel/NNCFConv2d[conv3]/conv2d_0']],
                                  pruned_groups_by_node_id=[[1, 2, 4]],
                                  can_prune_after_analysis={0: True, 1: True, 2: True, 3: True, 4: True,
                                                            5: True, 6: True, 7: True, 8: True, 9: True,
                                                            10: True, 11: True, 12: True},
                                  final_can_prune={1: PruningAnalysisDecision(True),
                                                   2: PruningAnalysisDecision(True),
                                                   4: PruningAnalysisDecision(True),
                                                   7: PruningAnalysisDecision(
                                                          False, [PruningAnalysisReason.DIMENSION_MISMATCH]),
                                                   8: PruningAnalysisDecision(
                                                          False, [PruningAnalysisReason.DIMENSION_MISMATCH]),
                                                   11: PruningAnalysisDecision(
                                                          False, [PruningAnalysisReason.LAST_CONV])},
                                  prune_params=(True, False)),
    GroupPruningModulesTestStruct(model=ResidualConnectionModel,
                                  non_pruned_module_nodes=['ResidualConnectionModel/NNCFLinear[linear]/linear_0'],
                                  pruned_groups=[['ResidualConnectionModel/NNCFConv2d[conv1]/conv2d_0',
                                                  'ResidualConnectionModel/NNCFConv2d[conv2]/conv2d_0',
                                                  'ResidualConnectionModel/NNCFConv2d[conv3]/conv2d_0'],
                                                 ['ResidualConnectionModel/NNCFConv2d[conv4]/conv2d_0',
                                                  'ResidualConnectionModel/NNCFConv2d[conv5]/conv2d_0']],
                                  pruned_groups_by_node_id=[[1, 2, 4]],
                                  can_prune_after_analysis={0: True, 1: True, 2: True, 3: True, 4: True,
                                                            5: True, 6: True, 7: True, 8: True, 9: True,
                                                            10: True, 11: True, 12: True},
                                  final_can_prune={1: PruningAnalysisDecision(True),
                                                   2: PruningAnalysisDecision(True),
                                                   4: PruningAnalysisDecision(True),
                                                   7: PruningAnalysisDecision(True),
                                                   8: PruningAnalysisDecision(True),
                                                   11: PruningAnalysisDecision(
                                                       False, [PruningAnalysisReason.LAST_CONV])},
                                  prune_params=(True, False)),
    GroupPruningModulesTestStruct(model=EltwiseCombinationModel,
                                  non_pruned_module_nodes=[
                                                  'EltwiseCombinationModel/NNCFConv2d[conv5]/conv2d_0',
                                                  'EltwiseCombinationModel/NNCFConv2d[conv6]/conv2d_0'],
                                  pruned_groups=[['EltwiseCombinationModel/NNCFConv2d[conv1]/conv2d_0',
                                                  'EltwiseCombinationModel/NNCFConv2d[conv2]/conv2d_0',
                                                  'EltwiseCombinationModel/NNCFConv2d[conv4]/conv2d_0',
                                                  'EltwiseCombinationModel/NNCFConv2d[conv3]/conv2d_0',],
                                                 ],
                                  pruned_groups_by_node_id=[[1, 2, 4, 6]],
                                  can_prune_after_analysis={0: True, 1: True, 2: True, 3: True, 4: True,
                                                            5: True, 6: True, 7: True, 8: True, 9: True,
                                                            10: True, 11: True},
                                  final_can_prune={1: PruningAnalysisDecision(True),
                                                   2: PruningAnalysisDecision(True),
                                                   4: PruningAnalysisDecision(True),
                                                   6: PruningAnalysisDecision(True),
                                                   8: PruningAnalysisDecision(
                                                          False, [PruningAnalysisReason.LAST_CONV]),
                                                   9: PruningAnalysisDecision(
                                                          False, [PruningAnalysisReason.LAST_CONV])},
                                  prune_params=(True, False)),
    GroupPruningModulesTestStruct(model=PruningTestModelSharedConvs,
                                  non_pruned_module_nodes=['PruningTestModelSharedConvs/NNCFConv2d[conv1]/conv2d_0',
                                                    'PruningTestModelSharedConvs/NNCFConv2d[conv3]/conv2d_0',
                                                    'PruningTestModelSharedConvs/NNCFConv2d[conv3]/conv2d_1'],
                                  pruned_groups=[['PruningTestModelSharedConvs/NNCFConv2d[conv2]/conv2d_0',
                                                  'PruningTestModelSharedConvs/NNCFConv2d[conv2]/conv2d_1']],
                                  pruned_groups_by_node_id=[[3, 4]],
                                  can_prune_after_analysis={0: True, 1: False, 2: True, 3: True, 4: True,
                                                            5: True, 6: True, 7: True, 8: True},
                                  final_can_prune={1: PruningAnalysisDecision(
                                                          False, [PruningAnalysisReason.CLOSING_CONV_MISSING]),
                                                   3: PruningAnalysisDecision(True),
                                                   4: PruningAnalysisDecision(True),
                                                   5: PruningAnalysisDecision(
                                                          False, [PruningAnalysisReason.LAST_CONV]),
                                                   6: PruningAnalysisDecision(
                                                          False, [PruningAnalysisReason.LAST_CONV])},
                                  prune_params=(False, False)),
    GroupPruningModulesTestStruct(model=DepthwiseConvolutionModel,
                                  non_pruned_module_nodes=['DepthwiseConvolutionModel/NNCFConv2d[conv1]/conv2d_0',
                                                           'DepthwiseConvolutionModel/NNCFConv2d[conv4]/conv2d_0'],
                                  pruned_groups=[['DepthwiseConvolutionModel/NNCFConv2d[conv2]/conv2d_0',
                                                  'DepthwiseConvolutionModel/NNCFConv2d[conv3]/conv2d_0',
                                                  'DepthwiseConvolutionModel/NNCFConv2d[depthwise_conv]/conv2d_0']],
                                  pruned_groups_by_node_id=[[2, 3, 5]],
                                  can_prune_after_analysis={0: True, 1: False, 2: True, 3: True, 4: True,
                                                            5: True, 6: True, 7: True},
                                  final_can_prune={1: PruningAnalysisDecision(
                                                          False, [PruningAnalysisReason.CLOSING_CONV_MISSING]),
                                                   2: PruningAnalysisDecision(True),
                                                   3: PruningAnalysisDecision(True),
                                                   6: PruningAnalysisDecision(
                                                          False, [PruningAnalysisReason.LAST_CONV])},
                                  prune_params=(False, False)),
    GroupPruningModulesTestStruct(
        model=MultipleDepthwiseConvolutionModel,
        non_pruned_module_nodes=['MultipleDepthwiseConvolutionModel/NNCFConv2d[conv1]/conv2d_0',
                                 'MultipleDepthwiseConvolutionModel/NNCFConv2d[conv4]/conv2d_0',
                                 'MultipleDepthwiseConvolutionModel/NNCFConv2d[conv2]/conv2d_0',
                                 'MultipleDepthwiseConvolutionModel/NNCFConv2d[conv3]/conv2d_0',
                                 'MultipleDepthwiseConvolutionModel/NNCFConv2d[depthwise_conv]/conv2d_0'],
        pruned_groups=[],
        pruned_groups_by_node_id=[],
        can_prune_after_analysis={0: True, 1: False, 2: False, 3: False, 4: False, 5: False, 6: True, 7: True},
        final_can_prune={1: PruningAnalysisDecision(
                                False, [PruningAnalysisReason.CLOSING_CONV_MISSING]),
                         2: PruningAnalysisDecision(
                                False, [PruningAnalysisReason.CLOSING_CONV_MISSING]),
                         3: PruningAnalysisDecision(
                                False, [PruningAnalysisReason.CLOSING_CONV_MISSING]),
                         6: PruningAnalysisDecision(
                                False, [PruningAnalysisReason.LAST_CONV])},
        prune_params=(False, False)),
    GroupPruningModulesTestStruct(
        model=NASnetBlock,
        non_pruned_module_nodes=['NASnetBlock/NNCFConv2d[first_conv]/conv2d_0',
                                 'NASnetBlock/CellB[cell]/SepConv[sep_conv1]/NNCFConv2d[conv1]/conv2d_0',
                                 'NASnetBlock/CellB[cell]/SepConv[sep_conv2]/NNCFConv2d[conv1]/conv2d_0',
                                 'NASnetBlock/CellB[cell]/NNCFConv2d[conv1]/conv2d_0',
                                 'NASnetBlock/CellB[cell]/SepConv[sep_conv3]/NNCFConv2d[conv1]/conv2d_0',
                                 'NASnetBlock/CellB[cell]/NNCFConv2d[conv2]/conv2d_0'],
        pruned_groups=[],
        pruned_groups_by_node_id=[],
        can_prune_after_analysis={0: True, 1: False, 2: False, 3: True, 4: False, 5: True, 6: False, 7: False,
                                  8: True, 9: False, 10: True, 11: True, 12: True, 13: True, 14: True, 15: True,
                                  16: True, 17: True, 18: True, 19: True},
        final_can_prune={16: PruningAnalysisDecision(
                                 False, [PruningAnalysisReason.LAST_CONV]),
                         1: PruningAnalysisDecision(
                                 False, [PruningAnalysisReason.CLOSING_CONV_MISSING]),
                         7: PruningAnalysisDecision(
                                 False, [PruningAnalysisReason.CLOSING_CONV_MISSING])},
        prune_params=(True, True)),
    GroupPruningModulesTestStruct(
        model=MobilenetV3BlockSEReshape,
        non_pruned_module_nodes=['MobilenetV3BlockSEReshape/NNCFConv2d[last_conv]/conv2d_0'],
        pruned_groups=[
            ['MobilenetV3BlockSEReshape/NNCFConv2d[first_conv]/conv2d_0',
             'MobilenetV3BlockSEReshape/InvertedResidual[inverted_residual]/Sequential[conv]/NNCFConv2d[4]/conv2d_0',
             'MobilenetV3BlockSEReshape/InvertedResidual[inverted_residual]/Sequential[conv]/NNCFConv2d[0]/conv2d_0',
             'MobilenetV3BlockSEReshape/InvertedResidual[inverted_residual]/Sequential[conv]/SELayerWithReshape[3]/'
             'Sequential[fc]/NNCFConv2d[3]/conv2d_0'],
            ['MobilenetV3BlockSEReshape/InvertedResidual[inverted_residual]/Sequential[conv]/'
             'SELayerWithReshape[3]/Sequential[fc]/NNCFConv2d[0]/conv2d_0'],
            ],
        pruned_groups_by_node_id=[[8], [1, 2, 11, 15]],
        can_prune_after_analysis={0: True, 1: True, 2: True, 3: True, 4: True, 5: True, 6: True, 7: True,
                                  8: True, 9: True, 10: True, 11: True, 12: True, 13: True, 14: True, 15: True,
                                  16: True, 17: True, 18: True, 19: True},
        final_can_prune={1: PruningAnalysisDecision(True),
                         8: PruningAnalysisDecision(True),
                         11: PruningAnalysisDecision(True),
                         15: PruningAnalysisDecision(True),
                         18: PruningAnalysisDecision(False, [PruningAnalysisReason.LAST_CONV])},

        prune_params=(True, True)),

    GroupPruningModulesTestStruct(
        model=partial(MobilenetV3BlockSEReshape, mode='linear'),
        name='MobilenetV3BlockSEReshape with linear mode',
        non_pruned_module_nodes=['MobilenetV3BlockSEReshape/NNCFConv2d[last_conv]/conv2d_0'],
        pruned_groups=[
            ['MobilenetV3BlockSEReshape/NNCFConv2d[first_conv]/conv2d_0',
             'MobilenetV3BlockSEReshape/InvertedResidual[inverted_residual]/Sequential[conv]/'
             'NNCFConv2d[4]/conv2d_0',
             'MobilenetV3BlockSEReshape/InvertedResidual[inverted_residual]/Sequential[conv]/'
             'NNCFConv2d[0]/conv2d_0',
             'MobilenetV3BlockSEReshape/InvertedResidual[inverted_residual]/Sequential[conv]/'
             'SELayerWithReshapeAndLinear[3]/Sequential[fc]/NNCFLinear[3]/linear_0'],
            ['MobilenetV3BlockSEReshape/InvertedResidual[inverted_residual]/Sequential[conv]/'
             'SELayerWithReshapeAndLinear[3]/Sequential[fc]/NNCFLinear[0]/linear_0'],
        ],
        pruned_groups_by_node_id=[[7], [1, 2, 10, 15]],
        can_prune_after_analysis={0: True, 1: True, 2: True, 3: True, 4: True, 5: True, 6: True, 7: True,
                                  8: True, 9: True, 10: True, 11: True, 12: True, 13: True, 14: True, 15: True,
                                  16: True, 17: True, 18: True, 19: True},
        final_can_prune={1: PruningAnalysisDecision(True),
                         7: PruningAnalysisDecision(True),
                         10: PruningAnalysisDecision(True),
                         15: PruningAnalysisDecision(True),
                         18: PruningAnalysisDecision(False, [PruningAnalysisReason.LAST_CONV])},

        prune_params=(True, True)),
    GroupPruningModulesTestStruct(
        model=partial(MobilenetV3BlockSEReshape, mode='linear_mean'),
        name='MobilenetV3BlockSEReshape with linear mean',
        non_pruned_module_nodes=
            ['MobilenetV3BlockSEReshape/NNCFConv2d[last_conv]/conv2d_0'],
        pruned_groups=[
            ['MobilenetV3BlockSEReshape/NNCFConv2d[first_conv]/conv2d_0',
             'MobilenetV3BlockSEReshape/InvertedResidual[inverted_residual]/Sequential[conv]/'
             'NNCFConv2d[0]/conv2d_0',
             'MobilenetV3BlockSEReshape/InvertedResidual[inverted_residual]/Sequential[conv]/'
             'SELayerWithReshapeAndLinearAndMean[3]/Sequential[fc]/NNCFLinear[2]/linear_0',
             'MobilenetV3BlockSEReshape/InvertedResidual[inverted_residual]/Sequential[conv]/'
             'NNCFConv2d[4]/conv2d_0'],
            ['MobilenetV3BlockSEReshape/InvertedResidual[inverted_residual]/Sequential[conv]/'
             'SELayerWithReshapeAndLinearAndMean[3]/Sequential[fc]/NNCFLinear[0]/linear_0']
            ],
        pruned_groups_by_node_id=[[1, 2, 8, 12], [6]],
        can_prune_after_analysis={0: True, 1: True, 2: True, 3: True, 4: True, 5: True, 6: True, 7: True,
                                  8: True, 9: True, 10: True, 11: True, 12: True, 13: True, 14: True, 15: True,
                                  16: True},
        final_can_prune={1: PruningAnalysisDecision(True),
                         6: PruningAnalysisDecision(True),
                         8: PruningAnalysisDecision(True),
                         12: PruningAnalysisDecision(True),
                         15: PruningAnalysisDecision(False, [PruningAnalysisReason.LAST_CONV])},

        prune_params=(True, True)),
    GroupPruningModulesTestStruct(
        model=partial(PruningTestModelWrongDimsElementwise, use_last_conv=False),
        name='PruningTestModelWrongDimsElementwise without last conv',
        non_pruned_module_nodes=['PruningTestModelWrongDimsElementwise/NNCFConv2d[first_conv]/conv2d_0',
                                 'PruningTestModelWrongDimsElementwise/NNCFConv2d[branch_conv]/conv2d_0'],
        pruned_groups=[],
        pruned_groups_by_node_id=[],
        can_prune_after_analysis={0: True, 1: True, 2: True, 3: True, 4: True, 5: True},
        final_can_prune={1: PruningAnalysisDecision(False, PruningAnalysisReason.LAST_CONV),
                         2: PruningAnalysisDecision(False, PruningAnalysisReason.LAST_CONV)},
        prune_params=(True, True)),
    GroupPruningModulesTestStruct(
        model=partial(PruningTestModelWrongDimsElementwise, use_last_conv=True),
        name='PruningTestModelWrongDimsElementwise with last conv',
        non_pruned_module_nodes=['PruningTestModelWrongDimsElementwise/NNCFConv2d[first_conv]/conv2d_0',
                                 'PruningTestModelWrongDimsElementwise/NNCFConv2d[branch_conv]/conv2d_0',
                                 'PruningTestModelWrongDimsElementwise/NNCFConv2d[last_conv]/conv2d_0'],
        pruned_groups=[],
        pruned_groups_by_node_id=[],
        can_prune_after_analysis={0: True, 1: True, 2: True, 3: True, 4: True, 5: True, 6: True},
        final_can_prune={1: PruningAnalysisDecision(False, PruningAnalysisReason.DIMENSION_MISMATCH),
                         2: PruningAnalysisDecision(False, PruningAnalysisReason.DIMENSION_MISMATCH),
                         5: PruningAnalysisDecision(False, PruningAnalysisReason.LAST_CONV)},
        prune_params=(True, True)),
    GroupPruningModulesTestStruct(
        model=PruningTestModelWrongDims,
        non_pruned_module_nodes=['PruningTestModelWrongDims/NNCFConv2d[first_conv]/conv2d_0',
                                 'PruningTestModelWrongDims/NNCFConv2d[last_conv]/conv2d_0'],
        pruned_groups=[],
        pruned_groups_by_node_id=[],
        can_prune_after_analysis={0: True, 1: True, 2: True, 3: True, 4: True},
        final_can_prune={1: PruningAnalysisDecision(False, PruningAnalysisReason.DIMENSION_MISMATCH),
                         3: PruningAnalysisDecision(False, PruningAnalysisReason.LAST_CONV)},
        prune_params=(True, True)),
    GroupPruningModulesTestStruct(
        model=PruningTestModelSimplePrunableLinear,
        non_pruned_module_nodes=['PruningTestModelSimplePrunableLinear/NNCFLinear[last_linear]/linear_0'],
        pruned_groups=[['PruningTestModelSimplePrunableLinear/NNCFConv2d[conv]/conv2d_0'],
                       ['PruningTestModelSimplePrunableLinear/NNCFLinear[linear]/linear_0']],
        pruned_groups_by_node_id=[[1], [3]],
        can_prune_after_analysis={0: True, 1: True, 2: True, 3: True, 4: True, 5: True},
        final_can_prune={1: PruningAnalysisDecision(True),
                         3: PruningAnalysisDecision(True),
                         4: PruningAnalysisDecision(False, PruningAnalysisReason.LAST_CONV)},
        prune_params=(True, True)),
    GroupPruningModulesTestStruct(
        model=GroupNormModel,
        non_pruned_module_nodes=['GroupNormModel/NNCFConv2d[conv2]/conv2d_0'],
        pruned_groups=[['GroupNormModel/NNCFConv2d[conv1]/conv2d_0']],
        pruned_groups_by_node_id=[[1]],
        can_prune_after_analysis={0: True, 1: True, 2: True, 3: False, 4: False, 5: False},
        final_can_prune={1: PruningAnalysisDecision(True),
                         3: PruningAnalysisDecision(False, PruningAnalysisReason.CLOSING_CONV_MISSING)},
        prune_params=(True, True)),
    GroupPruningModulesTestStruct(
        model=SplitIdentityModel,
        non_pruned_module_nodes=['SplitIdentityModel/NNCFConv2d[conv2]/conv2d_0'],
        pruned_groups=[['SplitIdentityModel/NNCFConv2d[conv1]/conv2d_0']],
        pruned_groups_by_node_id=[[1]],
        can_prune_after_analysis={0: True, 1: True, 2: True, 3: True},
        final_can_prune={1: PruningAnalysisDecision(True),
                         3: PruningAnalysisDecision(False, [PruningAnalysisReason.LAST_CONV])},
        prune_params=(True, True)),
    GroupPruningModulesTestStruct(
        model=SplitMaskPropFailModel,
        non_pruned_module_nodes=['SplitMaskPropFailModel/NNCFConv2d[conv1]/conv2d_0',
                                 'SplitMaskPropFailModel/NNCFConv2d[conv2]/conv2d_0',
                                 'SplitMaskPropFailModel/NNCFConv2d[conv3]/conv2d_0',],
        pruned_groups=[],
        pruned_groups_by_node_id=[],
        can_prune_after_analysis={0: True, 1: False, 2: True, 3: False, 4: False},
        final_can_prune={1: PruningAnalysisDecision(False, [PruningAnalysisReason.CLOSING_CONV_MISSING]),
                         3: PruningAnalysisDecision(False, [PruningAnalysisReason.CLOSING_CONV_MISSING]),
                         4: PruningAnalysisDecision(False, [PruningAnalysisReason.CLOSING_CONV_MISSING])},
        prune_params=(True, True)),
    GroupPruningModulesTestStruct(
        model=SplitPruningInvalidModel,
        non_pruned_module_nodes=['SplitPruningInvalidModel/NNCFConv2d[conv1]/conv2d_0',
                                 'SplitPruningInvalidModel/NNCFConv2d[conv2]/conv2d_0',
                                 'SplitPruningInvalidModel/NNCFConv2d[conv3]/conv2d_0',],
        pruned_groups=[],
        pruned_groups_by_node_id=[],
        can_prune_after_analysis={0: True, 1: False, 2: True, 3: False, 4: False},
        final_can_prune={1: PruningAnalysisDecision(False, [PruningAnalysisReason.CLOSING_CONV_MISSING]),
                         3: PruningAnalysisDecision(False, [PruningAnalysisReason.CLOSING_CONV_MISSING]),
                         4: PruningAnalysisDecision(False, [PruningAnalysisReason.CLOSING_CONV_MISSING])},
        prune_params=(True, True)),
    GroupPruningModulesTestStruct(
        model=SplitConcatModel,
        non_pruned_module_nodes=[],
        pruned_groups=[['SplitConcatModel/NNCFConv2d[conv1]/conv2d_0'],
                       ['SplitConcatModel/NNCFConv2d[conv2]/conv2d_0'],
                       ['SplitConcatModel/NNCFConv2d[conv3]/conv2d_0']],
        pruned_groups_by_node_id=[[1], [3], [4]],
        can_prune_after_analysis={0: True, 1: True, 2: True, 3: True, 4: True, 5: True, 6: True, 7: True},
        final_can_prune={1: PruningAnalysisDecision(True),
                         3: PruningAnalysisDecision(True),
                         4: PruningAnalysisDecision(True),
                         6: PruningAnalysisDecision(False, [PruningAnalysisReason.LAST_CONV])},
        prune_params=(True, True)),
    GroupPruningModulesTestStruct(
        model=MultipleSplitConcatModel,
        non_pruned_module_nodes=[],
        pruned_groups=[['MultipleSplitConcatModel/NNCFConv2d[conv1]/conv2d_0']],
        pruned_groups_by_node_id=[[1]],
        can_prune_after_analysis={0: True, 1: True, 2: True, 3: True, 4: True, 5: True, 6: True, 7: True,
                                  8: True, 9: True},
        final_can_prune={1: PruningAnalysisDecision(True),
                         3: PruningAnalysisDecision(False, [PruningAnalysisReason.CLOSING_CONV_MISSING]),
                         6: PruningAnalysisDecision(False, [PruningAnalysisReason.CLOSING_CONV_MISSING]),
                         7: PruningAnalysisDecision(False, [PruningAnalysisReason.CLOSING_CONV_MISSING]),
                         8: PruningAnalysisDecision(False, [PruningAnalysisReason.LAST_CONV])},
        prune_params=(True, True)),
    GroupPruningModulesTestStruct(
        model=SplitReshapeModel,
        non_pruned_module_nodes=['SplitReshapeModel/NNCFConv2d[conv1]/conv2d_0',
                                 'SplitReshapeModel/NNCFConv2d[conv2]/conv2d_0',
                                 'SplitReshapeModel/NNCFConv2d[conv3]/conv2d_0',],
        pruned_groups=[],
        pruned_groups_by_node_id=[],
        can_prune_after_analysis={0: True, 1: True, 2: True, 3: True, 4: True, 5: True, 6: True, 7: True, 8: True},
        final_can_prune={1: PruningAnalysisDecision(False, [PruningAnalysisReason.CLOSING_CONV_MISSING]),
                         5: PruningAnalysisDecision(False, [PruningAnalysisReason.LAST_CONV]),
                         6: PruningAnalysisDecision(False, [PruningAnalysisReason.LAST_CONV])},
        prune_params=(True, True)),
    GroupPruningModulesTestStruct(
        model=HRNetBlock,
        non_pruned_module_nodes=[],
        pruned_groups=[['HRNetBlock/NNCFConv2d[conv1]/conv2d_0'],
                       ['HRNetBlock/NNCFConv2d[conv2]/conv2d_0'],
                       ['HRNetBlock/NNCFConv2d[conv3]/conv2d_0'],
                       ['HRNetBlock/NNCFConv2d[conv4]/conv2d_0'],
                       ['HRNetBlock/NNCFConv2d[conv5]/conv2d_0']],
        pruned_groups_by_node_id=[[1], [2], [6], [7], [9]],
        can_prune_after_analysis={0: True, 1: True, 2: True, 3: True, 4: True, 5: True, 6: True, 7: True,
                                  8: True, 9: True, 10: True, 11: True, 12: True, 13: True},
        final_can_prune={1: PruningAnalysisDecision(True),
                         2: PruningAnalysisDecision(True),
                         6: PruningAnalysisDecision(True),
                         7: PruningAnalysisDecision(True),
                         9: PruningAnalysisDecision(True),
                         12: PruningAnalysisDecision(False, [PruningAnalysisReason.LAST_CONV]),
                         13: PruningAnalysisDecision(False, [PruningAnalysisReason.LAST_CONV])},
        prune_params=(True, True)),
    GroupPruningModulesTestStruct(
        model=PruningTestModelPad,
        non_pruned_module_nodes=['PruningTestModelPad/NNCFConv2d[conv7]/conv2d_0',
                                 'PruningTestModelPad/NNCFConv2d[conv8]/conv2d_0',
                                 'PruningTestModelPad/NNCFConv2d[conv9]/conv2d_0'],
        pruned_groups=[['PruningTestModelPad/NNCFConv2d[conv1]/conv2d_0'],
                       ['PruningTestModelPad/NNCFConv2d[conv2]/conv2d_0'],
                       ['PruningTestModelPad/NNCFConv2d[conv3]/conv2d_0'],
                       ['PruningTestModelPad/NNCFConv2d[conv4]/conv2d_0'],
                       ['PruningTestModelPad/NNCFConv2d[conv5]/conv2d_0'],
                       ['PruningTestModelPad/NNCFConv2d[conv6]/conv2d_0']],
        pruned_groups_by_node_id=[[1], [2], [4], [5], [7], [9]],
        can_prune_after_analysis={0: True, 1: True, 2: True, 3: True, 4: True, 5: True, 6: True, 7: True, 8: True,
                                  9: True, 10: True, 11: False, 12: False, 13: False, 14: False, 15: True, 16: True},
        final_can_prune={1: PruningAnalysisDecision(True),
                         2: PruningAnalysisDecision(True),
                         4: PruningAnalysisDecision(True),
                         5: PruningAnalysisDecision(True),
                         7: PruningAnalysisDecision(True),
                         9: PruningAnalysisDecision(True),
                         11: PruningAnalysisDecision(False, [PruningAnalysisReason.CLOSING_CONV_MISSING]),
                         13: PruningAnalysisDecision(False, [PruningAnalysisReason.CLOSING_CONV_MISSING]),
                         15: PruningAnalysisDecision(False, [PruningAnalysisReason.LAST_CONV])},
        prune_params=(True, True)),
    GroupPruningModulesTestStruct(model=PruningTestBatchedLinear,
        non_pruned_module_nodes=['PruningTestBatchedLinear/NNCFConv2d[first_conv]/conv2d_0',
                                 'PruningTestBatchedLinear/NNCFLinear[linear1]/linear_0',
                                 'PruningTestBatchedLinear/NNCFLinear[last_linear]/linear_0'],
        pruned_groups=[],
        pruned_groups_by_node_id=[],
        can_prune_after_analysis={0: True, 1: True, 2: False, 3: True, 4: True, 5: True},
        final_can_prune={1: PruningAnalysisDecision(False, PruningAnalysisReason.CLOSING_CONV_MISSING),
                         4: PruningAnalysisDecision(False, PruningAnalysisReason.LAST_CONV)},
        prune_params=(True, True)),
    GroupPruningModulesTestStruct(
        model=PruningTestModelBroadcastedLinearWithConcat,
        non_pruned_module_nodes=['PruningTestModelBroadcastedLinearWithConcat/NNCFLinear[last_linear]/linear_0'],
        pruned_groups=[['PruningTestModelBroadcastedLinearWithConcat/NNCFConv2d[first_conv]/conv2d_0'],
                       ['PruningTestModelBroadcastedLinearWithConcat/NNCFConv2d[conv1]/conv2d_0',
                        'PruningTestModelBroadcastedLinearWithConcat/NNCFLinear[linear1]/linear_0'],
                       ['PruningTestModelBroadcastedLinearWithConcat/NNCFConv2d[conv2]/conv2d_0']],
        pruned_groups_by_node_id=[[1], [2, 4], [7]],
        can_prune_after_analysis={0: True, 1: True, 2: True, 3: True, 4: True, 5: True, 6: True,
                                  7: True, 8: True, 9: True, 10: True, 11: True},
        final_can_prune={1: PruningAnalysisDecision(True),
                         2: PruningAnalysisDecision(True),
                         4: PruningAnalysisDecision(True),
                         7: PruningAnalysisDecision(True),
                         10: PruningAnalysisDecision(False, PruningAnalysisReason.LAST_CONV)},
        prune_params=(True, True)),
    GroupPruningModulesTestStruct(
        model=PruningTestModelDiffChInPruningCluster,
        non_pruned_module_nodes=[
            'PruningTestModelDiffChInPruningCluster/NNCFConv2d[conv1]/conv2d_0',
            'PruningTestModelDiffChInPruningCluster/NNCFLinear[linear1]/linear_0',
            'PruningTestModelDiffChInPruningCluster/NNCFLinear[last_linear]/linear_0'],
        pruned_groups=[['PruningTestModelDiffChInPruningCluster/NNCFConv2d[first_conv]/conv2d_0']],
        pruned_groups_by_node_id=[[1]],
        can_prune_after_analysis={0: True, 1: True, 2: False, 3: True, 4: True, 5: False, 6: True,
                                  7: True, 8: True},
        final_can_prune={1: PruningAnalysisDecision(True),
                         2: PruningAnalysisDecision(False, PruningAnalysisReason.CLOSING_CONV_MISSING),
                         5: PruningAnalysisDecision(False, PruningAnalysisReason.CLOSING_CONV_MISSING),
                         7: PruningAnalysisDecision(False, PruningAnalysisReason.LAST_CONV)},
        prune_params=(True, True)),
    GroupPruningModulesTestStruct(
        model=partial(PruningTestMeanMetatype, mean_dim=1),
        name='PruningTestMeanMetatype with mean dimension 1',
        non_pruned_module_nodes=['PruningTestMeanMetatype/NNCFConv2d[last_conv]/conv2d_0',
                                 'PruningTestMeanMetatype/NNCFConv2d[conv1]/conv2d_0'],
        pruned_groups=[],
        pruned_groups_by_node_id=[],
        can_prune_after_analysis={0: True, 1: False, 2: False, 3: True},
        final_can_prune={1: PruningAnalysisDecision(False, [PruningAnalysisReason.CLOSING_CONV_MISSING]),
                         3: PruningAnalysisDecision(False, [PruningAnalysisReason.LAST_CONV])},
        prune_params=(True, True)
    ),
    GroupPruningModulesTestStruct(
        model=partial(PruningTestMeanMetatype, mean_dim=2),
        name='PruningTestMeanMetatype with mean dimension 2',
        non_pruned_module_nodes=['PruningTestMeanMetatype/NNCFConv2d[last_conv]/conv2d_0'],
        pruned_groups=[['PruningTestMeanMetatype/NNCFConv2d[conv1]/conv2d_0']],
        pruned_groups_by_node_id=[[1]],
        can_prune_after_analysis={0: True, 1: True, 2: True, 3: True},
        final_can_prune={1: PruningAnalysisDecision(True),
                         3: PruningAnalysisDecision(False, [PruningAnalysisReason.LAST_CONV])},
        prune_params=(True, True)
    ),
]  # fmt: skip


@pytest.fixture(
    params=GROUP_PRUNING_MODULES_TEST_CASES,
    name="test_input_info_struct_",
    ids=list(map(str, GROUP_PRUNING_MODULES_TEST_CASES)),
)
def test_input_info_struct(request):
    return request.param


def test_groups(test_input_info_struct_: GroupPruningModulesTestStruct):
    model = test_input_info_struct_.model
    non_pruned_module_nodes = test_input_info_struct_.non_pruned_module_nodes
    pruned_groups = test_input_info_struct_.pruned_groups
    prune_first, prune_downsample = test_input_info_struct_.prune_params

    model = model()
    nncf_config = get_basic_pruning_config(input_sample_size=[1, 1, 8, 8])
    nncf_config["compression"]["algorithm"] = "filter_pruning"
    nncf_config["compression"]["params"]["prune_first_conv"] = prune_first
    nncf_config["compression"]["params"]["prune_downsample_convs"] = prune_downsample

    compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, nncf_config)

    # 1. Check all not pruned modules
    clusters = compression_ctrl.pruned_module_groups_info
    all_pruned_modules_info = clusters.get_all_nodes()
    all_pruned_modules = [info.module for info in all_pruned_modules_info]

    for node_name in non_pruned_module_nodes:
        module = compressed_model.nncf.get_containing_module(node_name)
        assert module is not None and module not in all_pruned_modules

    # 2. Check that all pruned groups are valid
    for group in pruned_groups:
        first_node_name = group[0]
        cluster = clusters.get_cluster_containing_element(first_node_name)
        cluster_modules = [n.module for n in cluster.elements]
        group_modules = [compressed_model.nncf.get_containing_module(node_name) for node_name in group]

        assert Counter(cluster_modules) == Counter(group_modules)
    assert len(pruned_groups) == len(clusters.get_all_clusters())


def test_pruning_node_selector(test_input_info_struct_: GroupPruningModulesTestStruct):
    model = test_input_info_struct_.model
    non_pruned_module_nodes = test_input_info_struct_.non_pruned_module_nodes
    pruned_groups_by_node_id = test_input_info_struct_.pruned_groups_by_node_id
    prune_first, prune_downsample = test_input_info_struct_.prune_params

    pruning_operations = [v.op_func_name for v in NNCF_PRUNING_MODULES_DICT]
    grouping_operations = PTElementwisePruningOp.get_all_op_aliases()
    from nncf.common.pruning.node_selector import PruningNodeSelector

    pruning_node_selector = PruningNodeSelector(
        PT_PRUNING_OPERATOR_METATYPES,
        pruning_operations,
        grouping_operations,
        None,
        None,
        prune_first,
        prune_downsample,
    )
    model = model()
    model.eval()
    nncf_network = NNCFNetwork(model, input_info=FillerInputInfo([FillerInputElement([1, 1, 8, 8])]))
    graph = nncf_network.nncf.get_original_graph()
    pruning_groups = pruning_node_selector.create_pruning_groups(graph)

    # 1. Check all not pruned modules
    all_pruned_nodes = pruning_groups.get_all_nodes()
    all_pruned_modules = [nncf_network.nncf.get_containing_module(node.node_name) for node in all_pruned_nodes]
    for node_name in non_pruned_module_nodes:
        module = nncf_network.nncf.get_containing_module(node_name)
        assert module is not None and module not in all_pruned_modules

    # 2. Check that all pruned groups are valid
    for group_by_id in pruned_groups_by_node_id:
        first_node_id = group_by_id[0]
        cluster = pruning_groups.get_cluster_containing_element(first_node_id)
        cluster_node_ids = [n.node_id for n in cluster.elements]
        cluster_node_ids.sort()

        assert Counter(cluster_node_ids) == Counter(group_by_id)


def test_symbolic_mask_propagation(test_input_info_struct_):
    model = test_input_info_struct_.model()
    prune_first, *_ = test_input_info_struct_.prune_params
    nncf_model, _ = create_nncf_model_and_pruning_builder(model, {"prune_first_conv": prune_first})
    pruning_types = [v.op_func_name for v in NNCF_PRUNING_MODULES_DICT]
    nncf_model.eval()
    graph = nncf_model.nncf.get_graph()
    algo = MaskPropagationAlgorithm(graph, PT_PRUNING_OPERATOR_METATYPES, SymbolicMaskProcessor)
    final_can_prune = algo.symbolic_mask_propagation(pruning_types, test_input_info_struct_.can_prune_after_analysis)
    # Check all output masks are deleted
    for node in graph.get_all_nodes():
        assert node.attributes["output_mask"] is None

    # Check ref decisions
    ref_final_can_prune = test_input_info_struct_.final_can_prune
    assert len(final_can_prune) == len(ref_final_can_prune)
    for idx in final_can_prune:
        assert final_can_prune[idx] == ref_final_can_prune[idx]


class GroupSpecialModulesTestStruct:
    def __init__(self, model: Callable, eltwise_clusters):
        self.model = model
        self.eltwise_clusters = eltwise_clusters


GROUP_SPECIAL_MODULES_TEST_CASES = [
    GroupSpecialModulesTestStruct(
        model=BranchingModel,
        eltwise_clusters=[[3, 5], [9]],
    ),
    GroupSpecialModulesTestStruct(
        model=ResidualConnectionModel,
        eltwise_clusters=[[3, 5], [9]],
    ),
    GroupSpecialModulesTestStruct(model=EltwiseCombinationModel, eltwise_clusters=[[3, 5, 7], [10]]),
]


@pytest.fixture(params=GROUP_SPECIAL_MODULES_TEST_CASES, name="test_special_ops_struct")
def test_special_ops_struct_(request):
    return request.param


def test_group_special_nodes(test_special_ops_struct: GroupSpecialModulesTestStruct):
    model = test_special_ops_struct.model()
    nncf_model, algo_builder = create_nncf_model_and_pruning_builder(model, {"prune_first_conv": True})

    special_ops_clusterization = cluster_special_ops(
        nncf_model.nncf.get_original_graph(),
        algo_builder.get_types_of_grouping_ops(),
        PTIdentityMaskForwardPruningOp.get_all_op_aliases(),
    )

    for ref_cluster in test_special_ops_struct.eltwise_clusters:
        cluster = special_ops_clusterization.get_cluster_containing_element(ref_cluster[0])
        assert sorted([node.node_id for node in cluster.elements]) == sorted(ref_cluster)


class ModelAnalyserTestStruct:
    def __init__(self, model: nn.Module, ref_can_prune: dict):
        self.model = model
        self.ref_can_prune = ref_can_prune


MODEL_ANALYSER_TEST_CASES = [
    ModelAnalyserTestStruct(
        model=ResidualConnectionModel,
        ref_can_prune={0: True, 1: True, 2: True, 3: True, 4: True, 5: True, 6: True, 7: True, 8: True, 9: True,
                       10: True, 11: True, 12: True}
    ),
    ModelAnalyserTestStruct(
        model=MultipleDepthwiseConvolutionModel,
        ref_can_prune={0: True, 1: True, 2: False, 3: False, 4: False, 5: True, 6: True, 7: True}
    ),
    ModelAnalyserTestStruct(
        model=NASnetBlock,
        ref_can_prune={0: True, 1: False, 2: True, 3: True, 4: True, 5: True, 6: False, 7: True, 8: True,
                       9: True, 10: True, 11: True, 12: True, 13: True, 14: True, 15: True, 16: True, 17: True,
                       18: True, 19: True}
    ),
    ModelAnalyserTestStruct(
        model=partial(MobilenetV3BlockSEReshape, mode='linear_mean'),
        ref_can_prune={0: True, 1: True, 2: True, 3: True, 4: True, 5: True, 6: True, 7: True,
                       8: True, 9: True, 10: True, 11: True, 12: True, 13: True, 14: True, 15: True,
                       16: True}
    )
]  # fmt: skip


@pytest.fixture(params=MODEL_ANALYSER_TEST_CASES, name="test_struct")
def test_struct_(request):
    return request.param


def test_model_analyzer(test_struct: GroupSpecialModulesTestStruct):
    model = test_struct.model()
    nncf_model, _ = create_nncf_model_and_pruning_builder(model, {"prune_first_conv": True})
    prune_operations_types = [
        op_name
        for meta_op in [PTConvolutionPruningOp, PTTransposeConvolutionPruningOp, PTLinearPruningOp]
        for op_name in meta_op.get_all_op_aliases()
    ]
    model_analyser = ModelAnalyzer(
        nncf_model.nncf.get_original_graph(), PT_PRUNING_OPERATOR_METATYPES, prune_operations_types
    )
    can_prune_analysis = model_analyser.analyse_model_before_pruning()
    for node_id in can_prune_analysis:
        assert can_prune_analysis[node_id].decision == test_struct.ref_can_prune[node_id], f"node id={node_id}"


class ModulePrunableTestStruct:
    def __init__(self, model: NNCFNetwork, config_params: dict, is_module_prunable: Dict[NNCFNodeName, bool]):
        self.model = model
        self.config_params = config_params
        self.is_module_prunable = is_module_prunable


IS_MODULE_PRUNABLE_TEST_CASES = [
    ModulePrunableTestStruct(
        model=DiffConvsModel,
        config_params={},
        is_module_prunable={'DiffConvsModel/NNCFConv2d[conv1]/conv2d_0': False,
                            'DiffConvsModel/NNCFConv2d[conv2]/conv2d_0': True,
                            'DiffConvsModel/NNCFConv2d[conv3]/conv2d_0': False,
                            'DiffConvsModel/NNCFConv2d[conv4]/conv2d_0': False},
    ),
    ModulePrunableTestStruct(
        model=DiffConvsModel,
        config_params={'prune_first_conv': True},
        is_module_prunable={'DiffConvsModel/NNCFConv2d[conv1]/conv2d_0': True,
                            'DiffConvsModel/NNCFConv2d[conv2]/conv2d_0': True,
                            'DiffConvsModel/NNCFConv2d[conv3]/conv2d_0': False,
                            'DiffConvsModel/NNCFConv2d[conv4]/conv2d_0': False},
    ),
    ModulePrunableTestStruct(
        model=DiffConvsModel,
        config_params={'prune_first_conv': True, 'prune_downsample_convs': True},
        is_module_prunable={'DiffConvsModel/NNCFConv2d[conv1]/conv2d_0': True,
                            'DiffConvsModel/NNCFConv2d[conv2]/conv2d_0': True,
                            'DiffConvsModel/NNCFConv2d[conv3]/conv2d_0': True,
                            'DiffConvsModel/NNCFConv2d[conv4]/conv2d_0': False},
    ),
    ModulePrunableTestStruct(
        model=BranchingModel,
        config_params={},
        is_module_prunable={'BranchingModel/NNCFConv2d[conv1]/conv2d_0': False,
                            'BranchingModel/NNCFConv2d[conv2]/conv2d_0': False,
                            'BranchingModel/NNCFConv2d[conv3]/conv2d_0': False,
                            'BranchingModel/NNCFConv2d[conv4]/conv2d_0': True,
                            'BranchingModel/NNCFConv2d[conv5]/conv2d_0': True},
    ),
    ModulePrunableTestStruct(
        model=BranchingModel,
        config_params={'prune_first_conv': True},
        is_module_prunable={'BranchingModel/NNCFConv2d[conv1]/conv2d_0': True,
                            'BranchingModel/NNCFConv2d[conv2]/conv2d_0': True,
                            'BranchingModel/NNCFConv2d[conv3]/conv2d_0': True,
                            'BranchingModel/NNCFConv2d[conv4]/conv2d_0': True,
                            'BranchingModel/NNCFConv2d[conv5]/conv2d_0': True},
    ),
    ModulePrunableTestStruct(
        model=ShuffleNetUnitModelDW,
        config_params={'prune_first_conv': True},
        is_module_prunable={
            'ShuffleNetUnitModelDW/NNCFConv2d[conv]/conv2d_0': True,
            'ShuffleNetUnitModelDW/TestShuffleUnit[unit1]/NNCFConv2d[dw_conv4]/conv2d_0': False,
            'ShuffleNetUnitModelDW/TestShuffleUnit[unit1]/NNCFConv2d[expand_conv5]/conv2d_0': True,
            'ShuffleNetUnitModelDW/TestShuffleUnit[unit1]/NNCFConv2d[compress_conv1]/conv2d_0': True,
            'ShuffleNetUnitModelDW/TestShuffleUnit[unit1]/NNCFConv2d[dw_conv2]/conv2d_0': False,
            'ShuffleNetUnitModelDW/TestShuffleUnit[unit1]/NNCFConv2d[expand_conv3]/conv2d_0': True},
    ),
    ModulePrunableTestStruct(
        model=ShuffleNetUnitModel,
        config_params={'prune_first_conv': True},
        is_module_prunable={'ShuffleNetUnitModel/NNCFConv2d[conv]/conv2d_0': True,
                            'ShuffleNetUnitModel/TestShuffleUnit[unit1]/NNCFConv2d[compress_conv1]/conv2d_0': True,
                            'ShuffleNetUnitModel/TestShuffleUnit[unit1]/NNCFConv2d[dw_conv2]/conv2d_0': True,
                            'ShuffleNetUnitModel/TestShuffleUnit[unit1]/NNCFConv2d[expand_conv3]/conv2d_0': True},
    ),
    ModulePrunableTestStruct(
        model=MultipleDepthwiseConvolutionModel,
        config_params={'prune_first_conv': False},
        is_module_prunable={'MultipleDepthwiseConvolutionModel/NNCFConv2d[conv1]/conv2d_0': False,
                            'MultipleDepthwiseConvolutionModel/NNCFConv2d[conv4]/conv2d_0': True,
                            'MultipleDepthwiseConvolutionModel/NNCFConv2d[conv2]/conv2d_0': True,
                            'MultipleDepthwiseConvolutionModel/NNCFConv2d[conv3]/conv2d_0': True,
                            'MultipleDepthwiseConvolutionModel/NNCFConv2d[depthwise_conv]/conv2d_0': False})
]  # fmt: skip


@pytest.fixture(params=IS_MODULE_PRUNABLE_TEST_CASES, name="test_prunable_struct")
def test_prunable_struct_(request):
    return request.param


def test_is_module_prunable(test_prunable_struct: ModulePrunableTestStruct):
    model = test_prunable_struct.model()
    nncf_model, algo_builder = create_nncf_model_and_pruning_builder(model, test_prunable_struct.config_params)
    graph = nncf_model.nncf.get_original_graph()
    for module_node_name in test_prunable_struct.is_module_prunable:
        nncf_node = graph.get_node_by_name(module_node_name)
        decision = algo_builder.pruning_node_selector._is_module_prunable(graph, nncf_node)
        assert decision.decision == test_prunable_struct.is_module_prunable[module_node_name]


class SimpleNode:
    def __init__(self, id_):
        self.id = id_


def test_nodes_cluster():
    # test creating
    cluster_id = 0
    nodes = [SimpleNode(0)]
    nodes_orders = [0]
    cluster = Cluster[SimpleNode](cluster_id, nodes, nodes_orders)
    assert cluster.id == cluster_id
    assert cluster.elements == nodes
    assert cluster.importance == max(nodes_orders)

    # test add nodes
    new_nodes = [SimpleNode(1), SimpleNode(2)]
    new_importance = 4
    cluster.add_elements(new_nodes, new_importance)
    assert cluster.importance == new_importance
    assert cluster.elements == nodes + new_nodes

    # test clean
    cluster.clean_cluster()
    assert cluster.elements == []
    assert cluster.importance == 0


def test_clusterization():
    nodes_1 = [SimpleNode(0), SimpleNode(1)]
    nodes_2 = [SimpleNode(2), SimpleNode(3)]
    cluster_1 = Cluster[SimpleNode](1, nodes_1, [node.id for node in nodes_1])
    cluster_2 = Cluster[SimpleNode](2, nodes_2, [node.id for node in nodes_2])

    clusterization = Clusterization()

    # test adding of clusters
    clusterization.add_cluster(cluster_1)
    assert 1 in clusterization.clusters
    assert 0 in clusterization._element_to_cluster and 1 in clusterization._element_to_cluster

    # test get_cluster_by_id
    assert clusterization.get_cluster_by_id(1) == cluster_1
    with pytest.raises(IndexError) as err:
        clusterization.get_cluster_by_id(5)
    assert "No cluster with id" in str(err.value)

    # test get_cluster_containing_element
    assert clusterization.get_cluster_containing_element(1) == cluster_1
    with pytest.raises(IndexError) as err:
        clusterization.get_cluster_containing_element(10)
    assert "No cluster for node" in str(err.value)

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
