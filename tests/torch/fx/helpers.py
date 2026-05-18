# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch.fx
from torch.fx.passes.graph_drawer import FxGraphDrawer

from nncf.experimental.torch.fx.transformations import apply_quantization_transformations
from nncf.torch.graph.graph import PTNNCFGraph
from nncf.torch.graph.operator_metatypes import PTConstNoopMetatype
from nncf.torch.graph.operator_metatypes import PTConv2dMetatype
from nncf.torch.graph.operator_metatypes import PTDepthwiseConv2dSubtype
from nncf.torch.graph.operator_metatypes import PTSumMetatype
from tests.cross_fw.test_templates.models import NNCFGraphToTest
from tests.cross_fw.test_templates.models import NNCFGraphToTestDepthwiseConv
from tests.cross_fw.test_templates.models import NNCFGraphToTestSumAggregation


def visualize_fx_model(model: torch.fx.GraphModule, output_svg_path: str):
    g = FxGraphDrawer(model, output_svg_path)
    g.get_dot_graph().write_svg(output_svg_path)


def get_torch_fx_model(
    model: torch.nn.Module, ex_input: torch.Tensor | tuple[torch.Tensor, ...], dynamic_shapes=None
) -> torch.fx.GraphModule:
    """
    Converts given module to GraphModule.

    :param model: Given torch Module.
    :return: Exported GraphModule.
    """
    try:
        named_param = next(model.named_parameters())
    except StopIteration:
        named_param = None
    if named_param is None:
        device = torch.device("cpu")
    else:
        device = named_param[1].device

    if isinstance(ex_input, torch.Tensor):
        ex_input = (ex_input,)
    device_ex_input = []
    for inp in ex_input:
        device_ex_input.append(inp.to(device))
    device_ex_input = tuple(device_ex_input)

    # Temporary workaround for executorch tests. Can be removed once torch>=2.11.0
    export_fn = (
        torch.export.export_for_training if hasattr(torch.export, "export_for_training") else torch.export.export
    )
    model.eval()
    with torch.no_grad():
        return export_fn(model, args=device_ex_input, dynamic_shapes=dynamic_shapes, strict=True).module(
            check_guards=False
        )


def get_torch_fx_model_q_transformed(model: torch.nn.Module, ex_input: torch.Tensor) -> torch.fx.GraphModule:
    """
    Converts given module to GraphModule and applies required quantization transformations to it.

    :param model: Given torch Module.
    :return: Exported GraphModule.
    """
    fx_model = get_torch_fx_model(model, ex_input)
    apply_quantization_transformations(fx_model)
    return fx_model


def get_single_conv_nncf_graph() -> NNCFGraphToTest:
    return NNCFGraphToTest(
        conv_metatype=PTConv2dMetatype,
        nncf_graph_cls=PTNNCFGraph,
        const_metatype=PTConstNoopMetatype,
    )


def get_depthwise_conv_nncf_graph() -> NNCFGraphToTestDepthwiseConv:
    return NNCFGraphToTestDepthwiseConv(
        depthwise_conv_metatype=PTDepthwiseConv2dSubtype,
        nncf_graph_cls=PTNNCFGraph,
        const_metatype=PTConstNoopMetatype,
    )


def get_sum_aggregation_nncf_graph() -> NNCFGraphToTestSumAggregation:
    return NNCFGraphToTestSumAggregation(
        conv_metatype=PTConv2dMetatype,
        sum_metatype=PTSumMetatype,
        nncf_graph_cls=PTNNCFGraph,
        const_metatype=PTConstNoopMetatype,
    )
