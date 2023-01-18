"""
 Copyright (c) 2023 Intel Corporation
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
from typing import List
from typing import Optional
from typing import Type

import pytest
from torch import Size
from torch import nn

from nncf.common.graph import BaseLayerAttributes
from nncf.common.graph import OperatorMetatype
from nncf.common.graph.layer_attributes import ConvolutionLayerAttributes
from nncf.common.graph.layer_attributes import GenericWeightedLayerAttributes
from nncf.common.graph.layer_attributes import GroupNormLayerAttributes
from nncf.common.graph.layer_attributes import LinearLayerAttributes
from nncf.torch.dynamic_graph.graph_tracer import ModelInputInfo
from nncf.torch.graph.operator_metatypes import PTBatchNormMetatype
from nncf.torch.graph.operator_metatypes import PTConv1dMetatype
from nncf.torch.graph.operator_metatypes import PTConv2dMetatype
from nncf.torch.graph.operator_metatypes import PTConv3dMetatype
from nncf.torch.graph.operator_metatypes import PTConvTranspose1dMetatype
from nncf.torch.graph.operator_metatypes import PTConvTranspose2dMetatype
from nncf.torch.graph.operator_metatypes import PTConvTranspose3dMetatype
from nncf.torch.graph.operator_metatypes import PTEmbeddingBagMetatype
from nncf.torch.graph.operator_metatypes import PTEmbeddingMetatype
from nncf.torch.graph.operator_metatypes import PTGroupNormMetatype
from nncf.torch.graph.operator_metatypes import PTInputNoopMetatype
from nncf.torch.graph.operator_metatypes import PTLinearMetatype
from nncf.torch.graph.operator_metatypes import PTOutputNoopMetatype
from nncf.torch.nncf_network import NNCFNetwork


class RefNodeDesc:
    def __init__(self,
                 metatype_cls: Type[OperatorMetatype],
                 layer_attributes: Optional[BaseLayerAttributes] = None,
                 layer_attributes_comparator: Optional[
                     Callable[[BaseLayerAttributes, BaseLayerAttributes], bool]] = None):
        self.metatype_cls = metatype_cls
        self.layer_attributes = layer_attributes
        self.layer_attributes_comparator = layer_attributes_comparator

    def __eq__(self, other: 'RefNodeDesc'):
        eq_metatype = self.metatype_cls == other.metatype_cls
        if not eq_metatype:
            print('metatype classes are different: {} vs {}'.format(self.metatype_cls, other.metatype_cls))
        eq_layer_attributes = self.layer_attributes == other.layer_attributes
        if self.layer_attributes_comparator is not None:
            eq_layer_attributes = self.layer_attributes_comparator(self.layer_attributes, other.layer_attributes)
        return eq_layer_attributes and eq_metatype


def default_comparator(first_attr: BaseLayerAttributes, second_attr: BaseLayerAttributes):
    if first_attr is None and second_attr is None:
        return True
    if first_attr is None or second_attr is None:
        print('attributes are different, because one of them is equal to None, another - not')
        return False
    are_equal = first_attr.__dict__ == second_attr.__dict__
    if not are_equal:
        pairs = ['  vs  '.join([f'{f[0]}:{f[1]}', f'{s[0]}:{s[1]}']) for f, s in
                 zip(first_attr.__dict__.items(), second_attr.__dict__.items()) if f[1] != s[1]]
        print('attributes are different:\n{}'.format('\n'.join(pairs)))
    return are_equal


COMPARATOR_TYPE = Callable[[BaseLayerAttributes, BaseLayerAttributes], bool]


class LayerAttributesTestDesc:
    def __init__(self,
                 module: nn.Module,
                 model_input_info_list: List[ModelInputInfo],
                 layer_attributes: BaseLayerAttributes,
                 metatype_cls: Type[OperatorMetatype],
                 layer_attributes_comparator: COMPARATOR_TYPE = default_comparator):
        self.module = module
        self.layer_attributes = layer_attributes
        self.model_input_info_list = model_input_info_list
        self.metatype_cls = metatype_cls
        self.layer_attributes_comparator = layer_attributes_comparator

    def __str__(self):
        return str(self.metatype_cls.__name__)


BATCH_NORM_REF_ATTR = GenericWeightedLayerAttributes(weight_requires_grad=True,
                                                     weight_shape=Size([1]),
                                                     filter_dimension_idx=0)
LIST_TEST_DESCS = [
    LayerAttributesTestDesc(
        module=nn.GroupNorm(1, 2),
        model_input_info_list=[ModelInputInfo([1, 2, 1, 1])],
        layer_attributes=GroupNormLayerAttributes(
            weight_requires_grad=True,
            num_channels=2,
            num_groups=1),
        metatype_cls=PTGroupNormMetatype
    ),
    LayerAttributesTestDesc(
        module=nn.BatchNorm2d(1),
        model_input_info_list=[ModelInputInfo([1, 1, 1, 1])],
        layer_attributes=BATCH_NORM_REF_ATTR,
        metatype_cls=PTBatchNormMetatype
    ),
    LayerAttributesTestDesc(
        module=nn.BatchNorm3d(1),
        model_input_info_list=[ModelInputInfo([1, 1, 1, 1, 1])],
        layer_attributes=BATCH_NORM_REF_ATTR,
        metatype_cls=PTBatchNormMetatype
    ),
    LayerAttributesTestDesc(
        module=nn.BatchNorm1d(1),
        model_input_info_list=[ModelInputInfo([1, 1, 1])],
        layer_attributes=BATCH_NORM_REF_ATTR,
        metatype_cls=PTBatchNormMetatype
    ),
    LayerAttributesTestDesc(
        module=nn.Conv2d(1, 1, 1),
        model_input_info_list=[ModelInputInfo([1, 1, 1, 1])],
        layer_attributes=ConvolutionLayerAttributes(
            weight_requires_grad=True,
            in_channels=1,
            out_channels=1,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            transpose=False,
            padding_values=(0, 0)),
        metatype_cls=PTConv2dMetatype
    ),
    LayerAttributesTestDesc(
        module=nn.Conv2d(2, 2, 1, groups=2),
        model_input_info_list=[ModelInputInfo([1, 2, 1, 1])],
        layer_attributes=ConvolutionLayerAttributes(
            weight_requires_grad=True,
            in_channels=2,
            out_channels=2,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=2,
            transpose=False,
            padding_values=(0, 0)),
        metatype_cls=PTConv2dMetatype
    ),
    LayerAttributesTestDesc(
        module=nn.Conv1d(1, 1, 1),
        model_input_info_list=[ModelInputInfo([1, 1, 1])],
        layer_attributes=ConvolutionLayerAttributes(
            weight_requires_grad=True,
            in_channels=1,
            out_channels=1,
            kernel_size=(1,),
            stride=(1,),
            groups=1,
            transpose=False,
            padding_values=(0,)),
        metatype_cls=PTConv1dMetatype
    ),
    LayerAttributesTestDesc(
        module=nn.Conv3d(1, 1, 1),
        model_input_info_list=[ModelInputInfo([1, 1, 1, 1, 1])],
        layer_attributes=ConvolutionLayerAttributes(
            weight_requires_grad=True,
            in_channels=1,
            out_channels=1,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            groups=1,
            transpose=False,
            padding_values=(0, 0, 0)),
        metatype_cls=PTConv3dMetatype
    ),
    LayerAttributesTestDesc(
        module=nn.ConvTranspose1d(1, 1, 1),
        model_input_info_list=[ModelInputInfo([1, 1, 1])],
        layer_attributes=ConvolutionLayerAttributes(
            weight_requires_grad=True,
            in_channels=1,
            out_channels=1,
            kernel_size=(1,),
            stride=(1,),
            groups=1,
            transpose=True,
            padding_values=(0,)),
        metatype_cls=PTConvTranspose1dMetatype
    ),
    LayerAttributesTestDesc(
        module=nn.ConvTranspose2d(1, 1, 1),
        model_input_info_list=[ModelInputInfo([1, 1, 1, 1])],
        layer_attributes=ConvolutionLayerAttributes(
            weight_requires_grad=True,
            in_channels=1,
            out_channels=1,
            kernel_size=(1, 1),
            stride=(1, 1),
            groups=1,
            transpose=True,
            padding_values=(0, 0)),
        metatype_cls=PTConvTranspose2dMetatype
    ),
    LayerAttributesTestDesc(
        module=nn.ConvTranspose3d(1, 1, 1),
        model_input_info_list=[ModelInputInfo([1, 1, 1, 1, 1])],
        layer_attributes=ConvolutionLayerAttributes(
            weight_requires_grad=True,
            in_channels=1,
            out_channels=1,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            groups=1,
            transpose=True,
            padding_values=(0, 0, 0)),
        metatype_cls=PTConvTranspose3dMetatype
    ),
    LayerAttributesTestDesc(
        module=nn.Linear(1, 1),
        model_input_info_list=[ModelInputInfo([1, 1, 1, 1])],
        layer_attributes=LinearLayerAttributes(
            weight_requires_grad=True,
            in_features=1,
            out_features=1),
        metatype_cls=PTLinearMetatype
    ),
    LayerAttributesTestDesc(
        module=nn.Linear(1, 1, bias=False),
        model_input_info_list=[ModelInputInfo([1, 1, 1, 1])],
        layer_attributes=LinearLayerAttributes(
            weight_requires_grad=True,
            in_features=1,
            out_features=1,
            bias=False),
        metatype_cls=PTLinearMetatype
    ),
    LayerAttributesTestDesc(
        module=nn.Embedding(2, 1),
        model_input_info_list=[ModelInputInfo([1, 1], type_str='long')],
        layer_attributes=GenericWeightedLayerAttributes(
            weight_requires_grad=True,
            weight_shape=Size([2, 1]),
            filter_dimension_idx=0),
        metatype_cls=PTEmbeddingMetatype
    ),
    LayerAttributesTestDesc(
        module=nn.EmbeddingBag(1, 1),
        model_input_info_list=[ModelInputInfo([1, 1], type_str='long', filler='zeros')],
        layer_attributes=GenericWeightedLayerAttributes(
            weight_requires_grad=True,
            weight_shape=Size([1, 1]),
            filter_dimension_idx=0),
        metatype_cls=PTEmbeddingBagMetatype
    ),
]


@pytest.mark.parametrize('desc', LIST_TEST_DESCS, ids=map(str, LIST_TEST_DESCS))
def test_can_set_valid_layer_attributes(desc: LayerAttributesTestDesc):
    single_layer_model = desc.module

    nncf_network = NNCFNetwork(single_layer_model, desc.model_input_info_list)

    nncf_network.eval()
    graph = nncf_network.get_graph()
    ref_values = [
        RefNodeDesc(PTInputNoopMetatype),
        RefNodeDesc(desc.metatype_cls, desc.layer_attributes, desc.layer_attributes_comparator),
        RefNodeDesc(PTOutputNoopMetatype),
    ]
    actual_values = [RefNodeDesc(node.metatype, node.layer_attributes) for node in graph.get_all_nodes()]
    assert ref_values == actual_values
