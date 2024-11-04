# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from functools import partial
from typing import List, Optional, Tuple, Union

import networkx as nx
import pytest
import torch
import torchvision.models as models

import nncf.experimental.torch2.function_hook.nncf_graph.operator_metatypes as om
from nncf.common.graph.layer_attributes import Dtype
from nncf.common.graph.operator_metatypes import UnknownMetatype
from nncf.experimental.torch2.function_hook.graph.build_graph_mode import build_graph
from nncf.experimental.torch2.function_hook.graph.graph_utils import ConstMeta
from nncf.experimental.torch2.function_hook.graph.graph_utils import FunctionMeta
from nncf.experimental.torch2.function_hook.graph.graph_utils import InOutMeta
from nncf.experimental.torch2.function_hook.graph.graph_utils import NodeType
from nncf.experimental.torch2.function_hook.graph.graph_utils import TensorMeta
from nncf.experimental.torch2.function_hook.nncf_graph.nncf_graph_builder import build_nncf_graph
from nncf.experimental.torch2.function_hook.nncf_graph.nncf_graph_builder import convert_to_nncf_graph
from nncf.experimental.torch2.function_hook.nncf_graph.nncf_graph_builder import get_dtype
from nncf.experimental.torch2.function_hook.nncf_graph.nncf_graph_builder import get_meta_type
from nncf.experimental.torch2.function_hook.nncf_graph.nncf_graph_builder import get_name_of_node
from nncf.experimental.torch2.function_hook.nncf_graph.nncf_graph_builder import get_node_type
from nncf.experimental.torch2.function_hook.wrapper import wrap_model
from tests.cross_fw.shared.paths import TEST_ROOT
from tests.torch2.function_hook import helpers
from tests.torch2.utils import compare_with_reference_file

REF_DIR = TEST_ROOT / "torch2" / "data" / "function_hook" / "nncf_graph"


@pytest.mark.parametrize(
    "node_type, meta, ref",
    [
        [NodeType.input, InOutMeta(torch.float32, (1), "input"), "nncf_model_input"],
        [NodeType.output, InOutMeta(torch.float32, (1), "output"), "nncf_model_output"],
        [NodeType.output, FunctionMeta("op", "fn_name_ref", [], {}), "fn_name_ref"],
        [NodeType.output, ConstMeta(torch.float32, (1), "model.bias"), "nncf_model_const"],
    ],
)
def test_get_node_type(node_type: NodeType, meta: Union[ConstMeta, FunctionMeta, InOutMeta], ref: str):
    assert get_node_type(node_type, meta) == ref


@pytest.mark.parametrize(
    "meta, ref",
    [
        [InOutMeta(torch.float32, (1), "input"), "input"],
        [InOutMeta(torch.float32, (1), "output"), "output"],
        [FunctionMeta("op_name", "fn_name", [], {}), "op_name"],
        [ConstMeta(torch.float32, (1), "model.bias"), "model.bias"],
    ],
)
def test_get_name_of_node(
    meta: list[InOutMeta | str] | list[FunctionMeta | str] | list[ConstMeta | str],
    ref: list[InOutMeta | str] | list[FunctionMeta | str] | list[ConstMeta | str],
):
    assert get_name_of_node(meta) == ref


@pytest.mark.parametrize(
    "dtype, ref",
    [
        [torch.float, Dtype.FLOAT],
        [torch.float32, Dtype.FLOAT],
        [torch.float64, Dtype.FLOAT],
        [torch.bfloat16, Dtype.FLOAT],
        [torch.int, Dtype.INTEGER],
        [torch.int8, Dtype.INTEGER],
        [torch.int16, Dtype.INTEGER],
        [torch.int32, Dtype.INTEGER],
        [torch.int64, Dtype.INTEGER],
    ],
)
def test_get_dtype(dtype: torch.dtype, ref: Dtype):
    assert get_dtype(dtype) == ref


@dataclass
class OpMetaTypeParam:
    node_type: NodeType
    ref_metatype: om.PTOperatorMetatype
    meta: Optional[Union[ConstMeta, FunctionMeta, InOutMeta]] = None

    def __str__(self):
        return f"{self.node_type}_{self.ref_metatype.__name__}"


def _conv_fn_meta(dim: int, groups: int, use_kwargs: bool) -> FunctionMeta:
    args = [TensorMeta(torch.float32, (10, 4, *[1] * dim), True), TensorMeta(torch.float32, (4, 1, *[1] * dim), True)]
    kwargs = {}
    if use_kwargs:
        kwargs = {"groups": groups}
    else:
        args.extend([*[None] * 5, groups])
    return FunctionMeta(
        f"op/conv{dim}d",
        f"conv{dim}d",
        args,
        kwargs,
    )


@pytest.mark.parametrize(
    "param",
    [
        OpMetaTypeParam("nncf_model_input", om.PTInputNoopMetatype, InOutMeta(torch.float32, (1), "input")),
        OpMetaTypeParam("nncf_model_output", om.PTOutputNoopMetatype, InOutMeta(torch.float32, (1), "output")),
        OpMetaTypeParam("nncf_model_const", om.PTConstNoopMetatype, ConstMeta(torch.float32, (1), "model.bias")),
        OpMetaTypeParam("unknown_fn", UnknownMetatype),
        OpMetaTypeParam("nncf_model_input", om.PTInputNoopMetatype),
        OpMetaTypeParam("nncf_model_output", om.PTOutputNoopMetatype),
        OpMetaTypeParam("nncf_model_const", om.PTConstNoopMetatype),
        OpMetaTypeParam("contiguous", om.PTNoopMetatype),
        OpMetaTypeParam("clone", om.PTNoopMetatype),
        OpMetaTypeParam("detach", om.PTNoopMetatype),
        OpMetaTypeParam("detach_", om.PTNoopMetatype),
        OpMetaTypeParam("to", om.PTNoopMetatype),
        OpMetaTypeParam(
            "conv1d",
            om.PTDepthwiseConv1dSubtype,
            _conv_fn_meta(dim=1, groups=4, use_kwargs=False),
        ),
        OpMetaTypeParam("conv1d", om.PTDepthwiseConv1dSubtype, _conv_fn_meta(dim=1, groups=4, use_kwargs=True)),
        OpMetaTypeParam("conv1d", om.PTConv1dMetatype, _conv_fn_meta(dim=1, groups=1, use_kwargs=False)),
        OpMetaTypeParam("conv1d", om.PTConv1dMetatype, _conv_fn_meta(dim=1, groups=1, use_kwargs=True)),
        OpMetaTypeParam("conv2d", om.PTDepthwiseConv2dSubtype, _conv_fn_meta(dim=2, groups=4, use_kwargs=False)),
        OpMetaTypeParam("conv2d", om.PTDepthwiseConv2dSubtype, _conv_fn_meta(dim=2, groups=4, use_kwargs=True)),
        OpMetaTypeParam("conv2d", om.PTConv2dMetatype, _conv_fn_meta(dim=2, groups=1, use_kwargs=False)),
        OpMetaTypeParam("conv2d", om.PTConv2dMetatype, _conv_fn_meta(dim=2, groups=1, use_kwargs=True)),
        OpMetaTypeParam("conv3d", om.PTDepthwiseConv3dSubtype, _conv_fn_meta(dim=3, groups=4, use_kwargs=False)),
        OpMetaTypeParam("conv3d", om.PTDepthwiseConv3dSubtype, _conv_fn_meta(dim=3, groups=4, use_kwargs=True)),
        OpMetaTypeParam("conv3d", om.PTConv3dMetatype, _conv_fn_meta(dim=3, groups=1, use_kwargs=False)),
        OpMetaTypeParam("conv3d", om.PTConv3dMetatype, _conv_fn_meta(dim=3, groups=1, use_kwargs=True)),
        OpMetaTypeParam("conv_transpose1d", om.PTConvTranspose1dMetatype),
        OpMetaTypeParam("conv_transpose2d", om.PTConvTranspose2dMetatype),
        OpMetaTypeParam("conv_transpose3d", om.PTConvTranspose3dMetatype),
        OpMetaTypeParam("deform_conv2d", om.PTDeformConv2dMetatype),
        OpMetaTypeParam("linear", om.PTLinearMetatype),
        OpMetaTypeParam("hardtanh", om.PTHardTanhMetatype),
        OpMetaTypeParam("hardswish", om.PTHardSwishMetatype),
        OpMetaTypeParam("hardswish_", om.PTHardSwishMetatype),
        OpMetaTypeParam("hardsigmoid", om.PTHardSigmoidMetatype),
        OpMetaTypeParam("tanh", om.PTTanhMetatype),
        OpMetaTypeParam("elu", om.PTELUMetatype),
        OpMetaTypeParam("elu_", om.PTELUMetatype),
        OpMetaTypeParam("prelu", om.PTPRELUMetatype),
        OpMetaTypeParam("leaky_relu", om.PTLeakyRELUMetatype),
        OpMetaTypeParam("layer_norm", om.PTLayerNormMetatype),
        OpMetaTypeParam("group_norm", om.PTGroupNormMetatype),
        OpMetaTypeParam("gelu", om.PTGELUMetatype),
        OpMetaTypeParam("silu", om.PTSILUMetatype),
        OpMetaTypeParam("silu_", om.PTSILUMetatype),
        OpMetaTypeParam("sigmoid", om.PTSigmoidMetatype),
        OpMetaTypeParam("add", om.PTAddMetatype),
        OpMetaTypeParam("add_", om.PTAddMetatype),
        OpMetaTypeParam("__add__", om.PTAddMetatype),
        OpMetaTypeParam("__iadd__", om.PTAddMetatype),
        OpMetaTypeParam("__radd__", om.PTAddMetatype),
        OpMetaTypeParam("sub", om.PTSubMetatype),
        OpMetaTypeParam("sub_", om.PTSubMetatype),
        OpMetaTypeParam("__sub__", om.PTSubMetatype),
        OpMetaTypeParam("__isub__", om.PTSubMetatype),
        OpMetaTypeParam("__rsub__", om.PTSubMetatype),
        OpMetaTypeParam("mul", om.PTMulMetatype),
        OpMetaTypeParam("mul_", om.PTMulMetatype),
        OpMetaTypeParam("__mul__", om.PTMulMetatype),
        OpMetaTypeParam("__imul__", om.PTMulMetatype),
        OpMetaTypeParam("__rmul__", om.PTMulMetatype),
        OpMetaTypeParam("div", om.PTDivMetatype),
        OpMetaTypeParam("div_", om.PTDivMetatype),
        OpMetaTypeParam("__div__", om.PTDivMetatype),
        OpMetaTypeParam("__idiv__", om.PTDivMetatype),
        OpMetaTypeParam("__rdiv__", om.PTDivMetatype),
        OpMetaTypeParam("__truediv__", om.PTDivMetatype),
        OpMetaTypeParam("__itruediv__", om.PTDivMetatype),
        OpMetaTypeParam("__rtruediv__", om.PTDivMetatype),
        OpMetaTypeParam("floor_divide", om.PTFloorDivMetatype),
        OpMetaTypeParam("__floordiv__", om.PTFloorDivMetatype),
        OpMetaTypeParam("__ifloordiv__", om.PTFloorDivMetatype),
        OpMetaTypeParam("__rfloordiv__", om.PTFloorDivMetatype),
        OpMetaTypeParam("exp", om.PTExpMetatype),
        OpMetaTypeParam("exp_", om.PTExpMetatype),
        OpMetaTypeParam("log", om.PTLogMetatype),
        OpMetaTypeParam("log_", om.PTLogMetatype),
        OpMetaTypeParam("abs", om.PTAbsMetatype),
        OpMetaTypeParam("abs_", om.PTAbsMetatype),
        OpMetaTypeParam("__abs__", om.PTAbsMetatype),
        OpMetaTypeParam("erf", om.PTErfMetatype),
        OpMetaTypeParam("erf_", om.PTErfMetatype),
        OpMetaTypeParam("matmul", om.PTMatMulMetatype),
        OpMetaTypeParam("bmm", om.PTMatMulMetatype),
        OpMetaTypeParam("mm", om.PTMatMulMetatype),
        OpMetaTypeParam("__matmul__", om.PTMatMulMetatype),
        OpMetaTypeParam("__rmatmul__", om.PTMatMulMetatype),
        OpMetaTypeParam("addmm", om.PTAddmmMetatype),
        OpMetaTypeParam("baddbmm", om.PTAddmmMetatype),
        OpMetaTypeParam("mean", om.PTMeanMetatype),
        OpMetaTypeParam("round", om.PTRoundMetatype),
        OpMetaTypeParam("round_", om.PTRoundMetatype),
        OpMetaTypeParam("dropout", om.PTDropoutMetatype),
        OpMetaTypeParam("dropout_", om.PTDropoutMetatype),
        OpMetaTypeParam("batch_norm", om.PTBatchNormMetatype),
        OpMetaTypeParam("batch_norm_", om.PTBatchNormMetatype),
        OpMetaTypeParam("avg_pool2d", om.PTAvgPool2dMetatype),
        OpMetaTypeParam("adaptive_avg_pool2d", om.PTAvgPool2dMetatype),
        OpMetaTypeParam("avg_pool3d", om.PTAvgPool3dMetatype),
        OpMetaTypeParam("adaptive_avg_pool3d", om.PTAvgPool3dMetatype),
        OpMetaTypeParam("adaptive_max_pool2d", om.PTAdaptiveMaxPool2dMetatype),
        OpMetaTypeParam("adaptive_max_pool3d", om.PTAdaptiveMaxPool3dMetatype),
        OpMetaTypeParam("max_pool2d", om.PTMaxPool2dMetatype),
        OpMetaTypeParam("max_pool3d", om.PTMaxPool3dMetatype),
        OpMetaTypeParam("max_unpool1d", om.PTMaxUnpool1dMetatype),
        OpMetaTypeParam("max_unpool2d", om.PTMaxUnpool2dMetatype),
        OpMetaTypeParam("max_unpool3d", om.PTMaxUnpool3dMetatype),
        OpMetaTypeParam("pad", om.PTPadMetatype),
        OpMetaTypeParam("cat", om.PTCatMetatype),
        OpMetaTypeParam("stack", om.PTCatMetatype),
        OpMetaTypeParam("concat", om.PTCatMetatype),
        OpMetaTypeParam("relu", om.PTRELUMetatype),
        OpMetaTypeParam("relu_", om.PTRELUMetatype),
        OpMetaTypeParam("relu6", om.PTRELU6Metatype),
        OpMetaTypeParam("max", om.PTMaxMetatype),
        OpMetaTypeParam("min", om.PTMinMetatype),
        OpMetaTypeParam("transpose", om.PTTransposeMetatype),
        OpMetaTypeParam("permute", om.PTTransposeMetatype),
        OpMetaTypeParam("transpose_", om.PTTransposeMetatype),
        OpMetaTypeParam("index_select", om.PTGatherMetatype),
        OpMetaTypeParam("__getitem__", om.PTGatherMetatype),
        OpMetaTypeParam("gather", om.PTGatherMetatype),
        OpMetaTypeParam("select", om.PTGatherMetatype),
        OpMetaTypeParam("where", om.PTGatherMetatype),
        OpMetaTypeParam("scatter", om.PTScatterMetatype),
        OpMetaTypeParam("masked_fill", om.PTScatterMetatype),
        OpMetaTypeParam("masked_fill_", om.PTScatterMetatype),
        OpMetaTypeParam("reshape", om.PTReshapeMetatype),
        OpMetaTypeParam("view", om.PTReshapeMetatype),
        OpMetaTypeParam("flatten", om.PTReshapeMetatype),
        OpMetaTypeParam("unflatten", om.PTReshapeMetatype),
        OpMetaTypeParam("unsqueeze", om.PTReshapeMetatype),
        OpMetaTypeParam("squeeze", om.PTSqueezeMetatype),
        OpMetaTypeParam("split", om.PTSplitMetatype),
        OpMetaTypeParam("chunk", om.PTSplitMetatype),
        OpMetaTypeParam("unbind", om.PTSplitMetatype),
        OpMetaTypeParam("expand", om.PTExpandMetatype),
        OpMetaTypeParam("expand_as", om.PTExpandAsMetatype),
        OpMetaTypeParam("embedding", om.PTEmbeddingMetatype),
        OpMetaTypeParam("embedding_bag", om.PTEmbeddingBagMetatype),
        OpMetaTypeParam("softmax", om.PTSoftmaxMetatype),
        OpMetaTypeParam("__lt__", om.PTLessMetatype),
        OpMetaTypeParam("__le__", om.PTLessEqualMetatype),
        OpMetaTypeParam("__gt__", om.PTGreaterMetatype),
        OpMetaTypeParam("gt", om.PTGreaterMetatype),
        OpMetaTypeParam("__ge__", om.PTGreaterEqualMetatype),
        OpMetaTypeParam("ge", om.PTGreaterEqualMetatype),
        OpMetaTypeParam("__mod__", om.PTModMetatype),
        OpMetaTypeParam("__eq__", om.PTEqualsMetatype),
        OpMetaTypeParam("eq", om.PTEqualsMetatype),
        OpMetaTypeParam("__ne__", om.PTNotEqualMetatype),
        OpMetaTypeParam("ne", om.PTNotEqualMetatype),
        OpMetaTypeParam("__or__", om.PTLogicalOrMetatype),
        OpMetaTypeParam("__ior__", om.PTLogicalOrMetatype),
        OpMetaTypeParam("__ror__", om.PTLogicalOrMetatype),
        OpMetaTypeParam("__xor__", om.PTLogicalXorMetatype),
        OpMetaTypeParam("__ixor__", om.PTLogicalXorMetatype),
        OpMetaTypeParam("__rxor__", om.PTLogicalXorMetatype),
        OpMetaTypeParam("__and__", om.PTLogicalAndMetatype),
        OpMetaTypeParam("__iand__", om.PTLogicalAndMetatype),
        OpMetaTypeParam("__rand__", om.PTLogicalAndMetatype),
        OpMetaTypeParam("logical_not_", om.PTLogicalNotMetatype),
        OpMetaTypeParam("__invert__", om.PTLogicalNotMetatype),
        OpMetaTypeParam("neg", om.PTNegativeMetatype),
        OpMetaTypeParam("__neg__", om.PTNegativeMetatype),
        OpMetaTypeParam("pow", om.PTPowerMetatype),
        OpMetaTypeParam("__pow__", om.PTPowerMetatype),
        OpMetaTypeParam("__ipow__", om.PTPowerMetatype),
        OpMetaTypeParam("__rpow__", om.PTPowerMetatype),
        OpMetaTypeParam("sqrt", om.PTSqrtMetatype),
        OpMetaTypeParam("sqrt_", om.PTSqrtMetatype),
        OpMetaTypeParam("interpolate", om.PTInterpolateMetatype),
        OpMetaTypeParam("repeat_interleave", om.PTRepeatMetatype),
        OpMetaTypeParam("pixel_shuffle", om.PTPixelShuffleMetatype),
        OpMetaTypeParam("sum", om.PTSumMetatype),
        OpMetaTypeParam("normalize", om.PTReduceL2),
        OpMetaTypeParam("scaled_dot_product_attention", om.PTScaledDotProductAttentionMetatype),
    ],
    ids=str,
)
def test_get_meta_type(param: OpMetaTypeParam):
    op = get_meta_type(param.node_type, param.meta)
    ref = param.ref_metatype
    assert op == ref


def test_convert_to_nncf_graph(regen_ref_data: bool):
    model = helpers.get_wrapped_simple_model_with_hook()
    nx_graph = build_graph(model, model.get_example_inputs())

    nncf_graph = convert_to_nncf_graph(nx_graph)
    graph = nncf_graph.get_graph_for_structure_analysis(extended=True)
    nx_nncf_graph = nx.nx_pydot.to_pydot(graph)

    ref_file = REF_DIR / "convert_to_nncf_graph.dot"

    compare_with_reference_file(str(nx_nncf_graph), ref_file, regen_ref_data)


def test_convert_to_nncf_graph_multi_edges(regen_ref_data: bool):
    model = helpers.ModelMultiEdge()
    model = wrap_model(model)
    nx_graph = build_graph(model, torch.ones(1, 1))
    nncf_graph = convert_to_nncf_graph(nx_graph)
    graph = nncf_graph.get_graph_for_structure_analysis(extended=True)
    nx_nncf_graph = nx.nx_pydot.to_pydot(graph)
    ref_file = REF_DIR / "convert_to_nncf_graph_multi_edges.dot"

    compare_with_reference_file(str(nx_nncf_graph), ref_file, regen_ref_data)


@dataclass
class ModelDesc:
    model_name: str
    model_builder: callable
    inputs_info: Union[List[List[int]], Tuple[List[int], ...]]

    def __str__(self):
        return self.model_name


TEST_MODELS_DESC = [
    ModelDesc("convnext_small", partial(models.convnext_small, weights=None), [1, 3, 64, 64]),
    ModelDesc("densenet121", partial(models.densenet121, weights=None), [1, 3, 64, 64]),
    ModelDesc("efficientnet_b0", partial(models.efficientnet_b0, weights=None), [1, 3, 64, 64]),
    ModelDesc("inception_v3", partial(models.inception_v3, weights=None), [1, 3, 300, 300]),
    ModelDesc("mobilenet_v2", partial(models.mobilenet_v2, weights=None), [1, 3, 64, 64]),
    ModelDesc("mobilenet_v3_small", partial(models.mobilenet_v3_small, weights=None), [1, 3, 64, 64]),
    ModelDesc("resnet18", partial(models.resnet18, weights=None), [1, 3, 64, 64]),
    ModelDesc("resnext50_32x4d", partial(models.resnext50_32x4d, weights=None), [1, 3, 64, 64]),
    ModelDesc("shufflenet_v2_x0_5", partial(models.shufflenet_v2_x0_5, weights=None), [1, 3, 224, 224]),
    ModelDesc("squeezenet1_0", partial(models.squeezenet1_0, weights=None), [1, 3, 64, 64]),
    ModelDesc("swin_v2_b", partial(models.swin_v2_b, weights=None), [1, 3, 64, 64]),
    ModelDesc("vgg16", partial(models.vgg16, weights=None), [1, 3, 32, 32]),
]


@pytest.mark.parametrize("desc", TEST_MODELS_DESC, ids=str)
def test_model_graph(desc: ModelDesc, regen_ref_data: bool):
    model: torch.nn.Module = desc.model_builder()
    model = model.eval()
    inputs = [torch.randn(desc.inputs_info)]
    model = wrap_model(model)
    nncf_graph = build_nncf_graph(model, *inputs)
    graph = nncf_graph.get_graph_for_structure_analysis(extended=True)
    nx_nncf_graph = nx.nx_pydot.to_pydot(graph)
    ref_file = REF_DIR / f"model_graph_{desc}.dot"
    compare_with_reference_file(str(nx_nncf_graph), ref_file, regen_ref_data)
