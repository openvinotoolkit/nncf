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
from typing import Callable, Optional, Type

import pytest
from torch import Size
from torch import nn

from nncf.common.graph.layer_attributes import BaseLayerAttributes
from nncf.common.graph.layer_attributes import ConvolutionLayerAttributes
from nncf.common.graph.layer_attributes import GenericWeightedLayerAttributes
from nncf.common.graph.layer_attributes import GetItemLayerAttributes
from nncf.common.graph.layer_attributes import GroupNormLayerAttributes
from nncf.common.graph.layer_attributes import LinearLayerAttributes
from nncf.common.graph.layer_attributes import MultipleInputLayerAttributes
from nncf.common.graph.layer_attributes import MultipleOutputLayerAttributes
from nncf.common.graph.layer_attributes import PermuteLayerAttributes
from nncf.common.graph.layer_attributes import ReshapeLayerAttributes
from nncf.common.graph.layer_attributes import TransposeLayerAttributes
from nncf.common.graph.layer_attributes import WeightedLayerAttributes
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.torch import wrap_model
from nncf.torch.dynamic_graph.graph_tracer import create_dummy_forward_fn
from nncf.torch.dynamic_graph.io_handling import FillerInputElement
from nncf.torch.dynamic_graph.io_handling import FillerInputInfo
from nncf.torch.dynamic_graph.io_handling import ModelInputInfo
from nncf.torch.dynamic_graph.layer_attributes_handlers import apply_args_defaults
from nncf.torch.graph.graph_builder import GraphBuilder
from nncf.torch.graph.operator_metatypes import PTBatchNormMetatype
from nncf.torch.graph.operator_metatypes import PTCatMetatype
from nncf.torch.graph.operator_metatypes import PTConv1dMetatype
from nncf.torch.graph.operator_metatypes import PTConv2dMetatype
from nncf.torch.graph.operator_metatypes import PTConv3dMetatype
from nncf.torch.graph.operator_metatypes import PTConvTranspose1dMetatype
from nncf.torch.graph.operator_metatypes import PTConvTranspose2dMetatype
from nncf.torch.graph.operator_metatypes import PTConvTranspose3dMetatype
from nncf.torch.graph.operator_metatypes import PTDepthwiseConv2dSubtype
from nncf.torch.graph.operator_metatypes import PTEmbeddingBagMetatype
from nncf.torch.graph.operator_metatypes import PTEmbeddingMetatype
from nncf.torch.graph.operator_metatypes import PTGatherMetatype
from nncf.torch.graph.operator_metatypes import PTGroupNormMetatype
from nncf.torch.graph.operator_metatypes import PTInputNoopMetatype
from nncf.torch.graph.operator_metatypes import PTLayerNormMetatype
from nncf.torch.graph.operator_metatypes import PTLinearMetatype
from nncf.torch.graph.operator_metatypes import PTOutputNoopMetatype
from nncf.torch.graph.operator_metatypes import PTReshapeMetatype
from nncf.torch.graph.operator_metatypes import PTSplitMetatype
from nncf.torch.graph.operator_metatypes import PTSqueezeMetatype
from nncf.torch.graph.operator_metatypes import PTTransposeMetatype
from nncf.torch.nncf_network import NNCFNetwork
from tests.torch.test_models.synthetic import ModelForGraphBuildingTest
from tests.torch.test_models.synthetic import ModelForGraphBuildingTestWithConcat
from tests.torch.test_models.synthetic import ModelForGraphBuildingTestWithReshapeFlattenAndConcat
from tests.torch.test_models.synthetic import ModelForGraphBuildingTestWithSplit
from tests.torch.test_models.synthetic import ModelWithDummyParameter
from tests.torch.test_models.synthetic import ModelWithPermute


class RefNodeDesc:
    def __init__(
        self,
        metatype_cls: Type[OperatorMetatype],
        layer_attributes: Optional[BaseLayerAttributes] = None,
        layer_attributes_comparator: Optional[Callable[[BaseLayerAttributes, BaseLayerAttributes], bool]] = None,
    ):
        self.metatype_cls = metatype_cls
        self.layer_attributes = layer_attributes
        self.layer_attributes_comparator = layer_attributes_comparator

    def __eq__(self, other: "RefNodeDesc"):
        eq_metatype = self.metatype_cls == other.metatype_cls
        if not eq_metatype:
            print("metatype classes are different: {} vs {}".format(self.metatype_cls, other.metatype_cls))
        eq_layer_attributes = self.layer_attributes == other.layer_attributes
        if self.layer_attributes_comparator is not None:
            eq_layer_attributes = self.layer_attributes_comparator(self.layer_attributes, other.layer_attributes)
        return eq_layer_attributes and eq_metatype


def default_comparator(first_attr: BaseLayerAttributes, second_attr: BaseLayerAttributes):
    if first_attr is None and second_attr is None:
        return True
    if first_attr is None or second_attr is None:
        print("attributes are different, because one of them is equal to None, another - not")
        return False
    are_equal = first_attr.__dict__ == second_attr.__dict__
    if not are_equal:
        pairs = [
            "  vs  ".join([f"{f[0]}:{f[1]}", f"{s[0]}:{s[1]}"])
            for f, s in zip(first_attr.__dict__.items(), second_attr.__dict__.items())
            if f[1] != s[1]
        ]
        print("attributes are different:\n{}".format("\n".join(pairs)))
    return are_equal


COMPARATOR_TYPE = Callable[[BaseLayerAttributes, BaseLayerAttributes], bool]


class LayerAttributesTestDesc:
    def __init__(
        self,
        module_fn: nn.Module,
        model_input_info: ModelInputInfo,
        layer_attributes: BaseLayerAttributes,
        metatype_cls: Type[OperatorMetatype],
        layer_attributes_comparator: COMPARATOR_TYPE = default_comparator,
    ):
        self.module_fn = module_fn
        self.layer_attributes = layer_attributes
        self.model_input_info = model_input_info
        self.metatype_cls = metatype_cls
        self.layer_attributes_comparator = layer_attributes_comparator

    def __str__(self):
        return str(self.metatype_cls.__name__)


BATCH_NORM_REF_ATTR = GenericWeightedLayerAttributes(
    weight_requires_grad=True, weight_shape=Size([1]), filter_dimension_idx=0, with_bias=True
)
LIST_TEST_DESCS = [
    LayerAttributesTestDesc(
        module_fn=lambda: nn.GroupNorm(1, 2),
        model_input_info=FillerInputInfo([FillerInputElement([1, 2, 1, 1])]),
        layer_attributes=GroupNormLayerAttributes(weight_requires_grad=True, num_channels=2, num_groups=1),
        metatype_cls=PTGroupNormMetatype,
    ),
    LayerAttributesTestDesc(
        module_fn=lambda: nn.BatchNorm2d(1),
        model_input_info=FillerInputInfo([FillerInputElement([1, 1, 1, 1])]),
        layer_attributes=BATCH_NORM_REF_ATTR,
        metatype_cls=PTBatchNormMetatype,
    ),
    LayerAttributesTestDesc(
        module_fn=lambda: nn.BatchNorm3d(1),
        model_input_info=FillerInputInfo([FillerInputElement([1, 1, 1, 1, 1])]),
        layer_attributes=BATCH_NORM_REF_ATTR,
        metatype_cls=PTBatchNormMetatype,
    ),
    LayerAttributesTestDesc(
        module_fn=lambda: nn.BatchNorm1d(1),
        model_input_info=FillerInputInfo([FillerInputElement([1, 1, 1])]),
        layer_attributes=BATCH_NORM_REF_ATTR,
        metatype_cls=PTBatchNormMetatype,
    ),
    LayerAttributesTestDesc(
        module_fn=lambda: nn.Conv2d(2, 1, 1),
        model_input_info=FillerInputInfo([FillerInputElement([1, 2, 1, 1])]),
        layer_attributes=ConvolutionLayerAttributes(
            weight_requires_grad=True,
            in_channels=2,
            out_channels=1,
            kernel_size=(1, 1),
            stride=(1, 1),
            dilations=(1, 1),
            groups=1,
            transpose=False,
            padding_values=(0, 0),
            with_bias=True,
        ),
        metatype_cls=PTConv2dMetatype,
    ),
    LayerAttributesTestDesc(
        module_fn=lambda: nn.Conv2d(2, 4, 1, groups=2),
        model_input_info=FillerInputInfo([FillerInputElement([1, 2, 1, 1])]),
        layer_attributes=ConvolutionLayerAttributes(
            weight_requires_grad=True,
            in_channels=2,
            out_channels=4,
            kernel_size=(1, 1),
            stride=(1, 1),
            dilations=(1, 1),
            groups=2,
            transpose=False,
            padding_values=(0, 0),
            with_bias=True,
        ),
        metatype_cls=PTDepthwiseConv2dSubtype,
    ),
    LayerAttributesTestDesc(
        module_fn=lambda: nn.Conv1d(1, 2, 1),
        model_input_info=FillerInputInfo([FillerInputElement([1, 1, 1])]),
        layer_attributes=ConvolutionLayerAttributes(
            weight_requires_grad=True,
            in_channels=1,
            out_channels=2,
            kernel_size=(1,),
            stride=(1,),
            dilations=(1,),
            groups=1,
            transpose=False,
            padding_values=(0,),
            with_bias=True,
        ),
        metatype_cls=PTConv1dMetatype,
    ),
    LayerAttributesTestDesc(
        module_fn=lambda: nn.Conv3d(1, 2, 1),
        model_input_info=FillerInputInfo([FillerInputElement([1, 1, 1, 1, 1])]),
        layer_attributes=ConvolutionLayerAttributes(
            weight_requires_grad=True,
            in_channels=1,
            out_channels=2,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            dilations=(1, 1, 1),
            groups=1,
            transpose=False,
            padding_values=(0, 0, 0),
            with_bias=True,
        ),
        metatype_cls=PTConv3dMetatype,
    ),
    LayerAttributesTestDesc(
        module_fn=lambda: nn.ConvTranspose1d(1, 2, 1),
        model_input_info=FillerInputInfo([FillerInputElement([1, 1, 1])]),
        layer_attributes=ConvolutionLayerAttributes(
            weight_requires_grad=True,
            in_channels=1,
            out_channels=2,
            kernel_size=(1,),
            stride=(1,),
            dilations=(1,),
            groups=1,
            transpose=True,
            padding_values=(0,),
            with_bias=True,
            output_padding_values=(0,),
        ),
        metatype_cls=PTConvTranspose1dMetatype,
    ),
    LayerAttributesTestDesc(
        module_fn=lambda: nn.ConvTranspose2d(1, 2, 1),
        model_input_info=FillerInputInfo([FillerInputElement([1, 1, 1, 1])]),
        layer_attributes=ConvolutionLayerAttributes(
            weight_requires_grad=True,
            in_channels=1,
            out_channels=2,
            kernel_size=(1, 1),
            stride=(1, 1),
            dilations=(1, 1),
            groups=1,
            transpose=True,
            padding_values=(0, 0),
            with_bias=True,
            output_padding_values=(0, 0),
        ),
        metatype_cls=PTConvTranspose2dMetatype,
    ),
    LayerAttributesTestDesc(
        module_fn=lambda: nn.ConvTranspose2d(2, 4, 1, groups=2),
        model_input_info=FillerInputInfo([FillerInputElement([1, 2, 1, 1])]),
        layer_attributes=ConvolutionLayerAttributes(
            weight_requires_grad=True,
            in_channels=2,
            out_channels=4,
            kernel_size=(1, 1),
            stride=(1, 1),
            dilations=(1, 1),
            groups=2,
            transpose=True,
            padding_values=(0, 0),
            with_bias=True,
            output_padding_values=(0, 0),
        ),
        metatype_cls=PTConvTranspose2dMetatype,
    ),
    LayerAttributesTestDesc(
        module_fn=lambda: nn.ConvTranspose3d(1, 2, 1),
        model_input_info=FillerInputInfo([FillerInputElement([1, 1, 1, 1, 1])]),
        layer_attributes=ConvolutionLayerAttributes(
            weight_requires_grad=True,
            in_channels=1,
            out_channels=2,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            dilations=(1, 1, 1),
            groups=1,
            transpose=True,
            padding_values=(0, 0, 0),
            with_bias=True,
            output_padding_values=(0, 0, 0),
        ),
        metatype_cls=PTConvTranspose3dMetatype,
    ),
    LayerAttributesTestDesc(
        module_fn=lambda: nn.Linear(1, 1),
        model_input_info=FillerInputInfo([FillerInputElement([1, 1, 1, 1])]),
        layer_attributes=LinearLayerAttributes(weight_requires_grad=True, in_features=1, out_features=1),
        metatype_cls=PTLinearMetatype,
    ),
    LayerAttributesTestDesc(
        module_fn=lambda: nn.Linear(1, 1, bias=False),
        model_input_info=FillerInputInfo([FillerInputElement([1, 1, 1, 1])]),
        layer_attributes=LinearLayerAttributes(
            weight_requires_grad=True, in_features=1, out_features=1, with_bias=False
        ),
        metatype_cls=PTLinearMetatype,
    ),
    LayerAttributesTestDesc(
        module_fn=lambda: nn.Embedding(2, 1),
        model_input_info=FillerInputInfo([FillerInputElement([1, 1], type_str="long")]),
        layer_attributes=GenericWeightedLayerAttributes(
            weight_requires_grad=True, weight_shape=Size([2, 1]), filter_dimension_idx=0
        ),
        metatype_cls=PTEmbeddingMetatype,
    ),
    LayerAttributesTestDesc(
        module_fn=lambda: nn.EmbeddingBag(1, 1),
        model_input_info=FillerInputInfo([FillerInputElement([1, 1], type_str="long", filler="zeros")]),
        layer_attributes=GenericWeightedLayerAttributes(
            weight_requires_grad=True, weight_shape=Size([1, 1]), filter_dimension_idx=0
        ),
        metatype_cls=PTEmbeddingBagMetatype,
    ),
    LayerAttributesTestDesc(
        module_fn=lambda: nn.LayerNorm(1, 1),
        model_input_info=FillerInputInfo([FillerInputElement([1, 1])]),
        layer_attributes=GenericWeightedLayerAttributes(
            weight_requires_grad=True, weight_shape=Size([1]), filter_dimension_idx=0, with_bias=True
        ),
        metatype_cls=PTLayerNormMetatype,
    ),
]


@pytest.mark.parametrize("desc", LIST_TEST_DESCS, ids=map(str, LIST_TEST_DESCS))
def test_can_set_valid_layer_attributes(desc: LayerAttributesTestDesc):
    single_layer_model = desc.module_fn()

    nncf_network = NNCFNetwork(single_layer_model, desc.model_input_info)

    nncf_network.eval()
    graph = nncf_network.nncf.get_graph()
    ref_values = [
        RefNodeDesc(PTInputNoopMetatype),
        RefNodeDesc(desc.metatype_cls, desc.layer_attributes, desc.layer_attributes_comparator),
        RefNodeDesc(PTOutputNoopMetatype),
    ]
    actual_values = [RefNodeDesc(node.metatype, node.layer_attributes) for node in graph.get_all_nodes()]
    assert ref_values == actual_values


@pytest.mark.parametrize("input_shape", (ModelForGraphBuildingTestWithConcat.INPUT_SHAPE,))
def test_concat_attributes_saved_during_graph_building(input_shape):
    model = ModelForGraphBuildingTestWithConcat()
    graph_builder = GraphBuilder(
        create_dummy_forward_fn(
            FillerInputInfo([FillerInputElement(input_shape)]),
            with_input_tracing=True,
            with_output_tracing=True,
        )
    )
    graph = graph_builder.build_graph(model)
    cat_nodes_with_attributes = {
        "ModelForGraphBuildingTestWithConcat/stack_0": MultipleInputLayerAttributes(axis=0, num_inputs=2),
        "ModelForGraphBuildingTestWithConcat/stack_1": MultipleInputLayerAttributes(axis=0, num_inputs=3),
        "ModelForGraphBuildingTestWithConcat/stack_2": MultipleInputLayerAttributes(axis=3, num_inputs=4),
        "ModelForGraphBuildingTestWithConcat/stack_3": MultipleInputLayerAttributes(axis=2, num_inputs=5),
        "ModelForGraphBuildingTestWithConcat/cat_0": MultipleInputLayerAttributes(axis=0, num_inputs=2),
        "ModelForGraphBuildingTestWithConcat/cat_1": MultipleInputLayerAttributes(axis=0, num_inputs=3),
        "ModelForGraphBuildingTestWithConcat/cat_2": MultipleInputLayerAttributes(axis=3, num_inputs=4),
        "ModelForGraphBuildingTestWithConcat/cat_3": MultipleInputLayerAttributes(axis=2, num_inputs=5),
    }

    for node in graph.get_all_nodes():
        if node.metatype is PTCatMetatype:
            assert node.node_name in cat_nodes_with_attributes
            if isinstance(node.layer_attributes, MultipleInputLayerAttributes):
                assert node.layer_attributes == cat_nodes_with_attributes[node.node_name]
            else:
                assert node.layer_attributes is None
                assert cat_nodes_with_attributes[node.node_name] is None


@pytest.mark.parametrize("input_shape", ModelForGraphBuildingTest.INPUT_SHAPES)
def test_reshape_attributes_saved_during_graph_building(input_shape):
    model = ModelForGraphBuildingTestWithReshapeFlattenAndConcat()
    graph_builder = GraphBuilder(
        create_dummy_forward_fn(
            FillerInputInfo(
                [
                    FillerInputElement(input_shape),
                ]
            ),
            with_input_tracing=True,
            with_output_tracing=True,
        )
    )
    graph = graph_builder.build_graph(model)
    reshape_nodes_with_attributes = {
        "ModelForGraphBuildingTestWithReshapeFlattenAndConcat/view_0": {
            "input_shape": (input_shape[0], ModelForGraphBuildingTest.OUT_CHANNELS, input_shape[2], input_shape[3]),
            "output_shape": (
                input_shape[0],
                ModelForGraphBuildingTest.OUT_CHANNELS,
                input_shape[2],
                input_shape[3],
                1,
                1,
            ),
        },
        "ModelForGraphBuildingTestWithReshapeFlattenAndConcat/flatten_0": {
            "input_shape": (
                2,
                input_shape[0],
                ModelForGraphBuildingTest.OUT_CHANNELS,
                input_shape[2],
                input_shape[3],
                1,
                2,
            ),
            "output_shape": (
                input_shape[0] * ModelForGraphBuildingTest.OUT_CHANNELS * input_shape[2] * input_shape[3] * 4,
            ),
        },
        "ModelForGraphBuildingTestWithReshapeFlattenAndConcat/view_1": None,
    }

    for node in graph.get_all_nodes():
        if node.metatype in [PTReshapeMetatype, PTSqueezeMetatype]:
            assert node.node_name in reshape_nodes_with_attributes
            if isinstance(node.layer_attributes, ReshapeLayerAttributes):
                ref_attrs = reshape_nodes_with_attributes[node.node_name]
                assert node.layer_attributes.input_shape == ref_attrs["input_shape"]
                assert node.layer_attributes.output_shape == ref_attrs["output_shape"]
            else:
                assert node.layer_attributes is None
                assert reshape_nodes_with_attributes[node.node_name] is None


transpose_input_shapes = [(1, 10, 20, 10), (10, 10, 10, 10)]


@pytest.mark.parametrize("input_shape", transpose_input_shapes)
def test_permute_attributes_saved_during_graph_building(input_shape):
    model = ModelWithPermute()
    graph_builder = GraphBuilder(
        create_dummy_forward_fn(
            FillerInputInfo(
                [
                    FillerInputElement(input_shape),
                ]
            ),
            with_input_tracing=True,
            with_output_tracing=True,
        )
    )
    graph = graph_builder.build_graph(model)
    transpose_nodes_with_attributes = {
        "ModelWithPermute/transpose_0": TransposeLayerAttributes(1, 3),
        "ModelWithPermute/transpose_1": TransposeLayerAttributes(1, 3),
        "ModelWithPermute/transpose_2": TransposeLayerAttributes(1, 3),
        "ModelWithPermute/permute_0": PermuteLayerAttributes((3, 2, 1, 0)),
        "ModelWithPermute/permute_1": PermuteLayerAttributes([3, 2, 1, 0]),
    }

    for node in graph.get_all_nodes():
        if node.metatype is PTTransposeMetatype:
            assert node.node_name in transpose_nodes_with_attributes
            if isinstance(node.layer_attributes, (TransposeLayerAttributes, PermuteLayerAttributes)):
                ref_attrs = transpose_nodes_with_attributes[node.node_name]
                assert node.layer_attributes == ref_attrs
            else:
                assert node.layer_attributes is None
                assert transpose_nodes_with_attributes[node.node_name] is None


@pytest.mark.parametrize("input_shape", ModelForGraphBuildingTest.INPUT_SHAPES)
def test_split_attributes(input_shape):
    model = ModelForGraphBuildingTestWithSplit(input_shape)
    graph_builder = GraphBuilder(
        create_dummy_forward_fn(
            FillerInputInfo(
                [
                    FillerInputElement(input_shape),
                ]
            ),
            with_input_tracing=True,
            with_output_tracing=True,
        )
    )

    graph = graph_builder.build_graph(model)
    chunk_nodes_with_attributes = {
        "ModelForGraphBuildingTestWithSplit/chunk_0": MultipleOutputLayerAttributes(chunks=2, axis=1),
        "ModelForGraphBuildingTestWithSplit/unbind_0": MultipleOutputLayerAttributes(chunks=20, axis=1),
    }

    for node in graph.get_all_nodes():
        if node.metatype is PTSplitMetatype:
            assert node.node_name in chunk_nodes_with_attributes
            if isinstance(node.layer_attributes, MultipleOutputLayerAttributes):
                ref_attrs = chunk_nodes_with_attributes[node.node_name]
                assert node.layer_attributes == ref_attrs
            else:
                assert node.layer_attributes is None
                assert chunk_nodes_with_attributes[node.node_name] is None


class SplitByGetItemModel(ModelWithDummyParameter):
    def forward(self, x):
        return x[0:1], x[(0, 1)], x[2]


@pytest.mark.parametrize("input_shape", [(3, 2)])
def test_getitem_attributes(input_shape):
    model = SplitByGetItemModel()
    custom_forward_fn = create_dummy_forward_fn(
        FillerInputInfo(
            [
                FillerInputElement(input_shape),
            ]
        ),
        with_input_tracing=True,
        with_output_tracing=True,
    )
    graph_builder = GraphBuilder(custom_forward_fn)
    graph = graph_builder.build_graph(model)
    getitem_nodes_with_attributes = {
        "SplitByGetItemModel/__getitem___0": slice(0, 1, None),
        "SplitByGetItemModel/__getitem___1": (0, 1),
        "SplitByGetItemModel/__getitem___2": 2,
    }

    for node in graph.get_all_nodes():
        if node.metatype is PTGatherMetatype:
            assert node.node_name in getitem_nodes_with_attributes
            if isinstance(node.layer_attributes, GetItemLayerAttributes):
                ref_key = getitem_nodes_with_attributes[node.node_name]
                assert node.layer_attributes.key == ref_key
            else:
                assert node.layer_attributes is None
                assert getitem_nodes_with_attributes[node.node_name] is None


@pytest.mark.parametrize("desc", LIST_TEST_DESCS, ids=map(str, LIST_TEST_DESCS))
def test_can_set_valid_layer_attributes_wrap_model(desc: LayerAttributesTestDesc):
    nncf_network = wrap_model(desc.module_fn(), desc.model_input_info.get_forward_inputs()[0], trace_parameters=True)
    graph = nncf_network.nncf.get_graph()
    ref_values = [RefNodeDesc(desc.metatype_cls, desc.layer_attributes, desc.layer_attributes_comparator)]
    actual_values = [
        RefNodeDesc(node.metatype, node.layer_attributes) for node in graph.get_nodes_by_metatypes([desc.metatype_cls])
    ]
    assert ref_values == actual_values

    if isinstance(desc.layer_attributes, WeightedLayerAttributes):
        assert hasattr(desc.metatype_cls, "weight_port_ids")
        assert len(desc.metatype_cls.weight_port_ids) > 0


@pytest.mark.parametrize(
    "signature, args, kwargs",
    (
        (["a", "b"], [1, 2], {}),
        (["a", "b"], [], {"a": 1, "b": 2}),
        (["a", "b"], [1], {"b": 2}),
        (["a", ("b", 2)], [1], {"b": 2}),
        ([("a", 1), ("b", 2)], [], {"b": 2}),
        ([("a", 1), ("b", 2)], [], {}),
    ),
)
def test_apply_args_defaults(signature, args, kwargs):
    ret = apply_args_defaults(args, kwargs, signature)
    assert ret == {"a": 1, "b": 2}


@pytest.mark.parametrize(
    "signature, args, kwargs",
    (
        (["a", "b"], [1], {}),
        ([1, 2], [], {}),
    ),
)
def test_apply_args_defaults_errors(signature, args, kwargs):
    with pytest.raises(ValueError):
        apply_args_defaults(args, kwargs, signature)
