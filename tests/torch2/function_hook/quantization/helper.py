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

from typing import List, Optional

import torch

from nncf.common.graph.layer_attributes import ConstantLayerAttributes
from nncf.common.graph.layer_attributes import ConvolutionLayerAttributes
from nncf.torch import wrap_model
from nncf.torch.graph.graph import PTNNCFGraph
from nncf.torch.graph.operator_metatypes import PTConstNoopMetatype
from nncf.torch.graph.operator_metatypes import PTModuleConv2dMetatype
from nncf.torch.graph.operator_metatypes import PTModuleDepthwiseConv2dSubtype
from nncf.torch.graph.operator_metatypes import PTModuleLinearMetatype
from nncf.torch.graph.operator_metatypes import PTSumMetatype
from tests.cross_fw.test_templates.models import NNCFGraphToTest
from tests.cross_fw.test_templates.models import NNCFGraphToTestDepthwiseConv
from tests.cross_fw.test_templates.models import NNCFGraphToTestSumAggregation


def get_single_conv_nncf_graph() -> NNCFGraphToTest:
    conv_layer_attrs = ConvolutionLayerAttributes(
        weight_requires_grad=True,
        in_channels=4,
        out_channels=4,
        kernel_size=(4, 4),
        stride=1,
        dilations=1,
        groups=1,
        transpose=False,
        padding_values=[],
    )
    return NNCFGraphToTest(
        PTModuleConv2dMetatype,
        conv_layer_attrs,
        PTNNCFGraph,
        const_metatype=PTConstNoopMetatype,
        const_layer_attrs=ConstantLayerAttributes("w", shape=[4, 4, 4, 4]),
    )


def get_depthwise_conv_nncf_graph() -> NNCFGraphToTestDepthwiseConv:
    conv_layer_attrs = ConvolutionLayerAttributes(
        weight_requires_grad=False,
        in_channels=3,
        out_channels=3,
        dilations=1,
        kernel_size=(1, 1),
        stride=(1, 1),
        groups=3,
        transpose=False,
        padding_values=(1, 1),
    )
    return NNCFGraphToTestDepthwiseConv(
        PTModuleDepthwiseConv2dSubtype,
        conv_layer_attrs,
        nncf_graph_cls=PTNNCFGraph,
        const_metatype=PTConstNoopMetatype,
        const_layer_attrs=ConstantLayerAttributes("w", shape=[4, 4, 4, 4]),
    )


def get_single_no_weight_matmul_nncf_graph() -> NNCFGraphToTest:
    return NNCFGraphToTest(PTModuleLinearMetatype, None, PTNNCFGraph)


def get_sum_aggregation_nncf_graph() -> NNCFGraphToTestSumAggregation:
    conv_layer_attrs = ConvolutionLayerAttributes(
        weight_requires_grad=True,
        in_channels=4,
        out_channels=4,
        kernel_size=(4, 4),
        stride=1,
        dilations=1,
        groups=1,
        transpose=False,
        padding_values=[],
    )
    return NNCFGraphToTestSumAggregation(
        PTModuleConv2dMetatype,
        PTSumMetatype,
        conv_layer_attrs,
        PTNNCFGraph,
        const_metatype=PTConstNoopMetatype,
        const_layer_attrs=ConstantLayerAttributes("w", shape=[4, 4, 4, 4]),
    )


def get_nncf_network(model: torch.nn.Module, input_shape: Optional[List[int]] = None):
    if input_shape is None:
        input_shape = [1, 3, 32, 32]
    model = model.eval()
    device = next(model.named_parameters())[1].device
    return wrap_model(model, torch.ones(input_shape).to(device=device), trace_parameters=True)
