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

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import pytest
import torch
from networkx.drawing.nx_pydot import to_pydot
from torchvision.models import mobilenet_v2
from torchvision.models import mobilenet_v3_small

import nncf
from nncf.parameters import ModelType
from nncf.torch.function_hook.nncf_graph.nncf_graph_builder import build_nncf_graph
from tests.cross_fw.shared.paths import TEST_ROOT
from tests.cross_fw.test_templates.helpers import EmbeddingModel
from tests.cross_fw.test_templates.helpers import RoPEModel
from tests.cross_fw.test_templates.helpers import ScaledDotProductAttentionModel
from tests.cross_fw.test_templates.helpers import UnbindScaledDotProductAttentionModel
from tests.torch import test_models
from tests.torch.function_hook.helpers import SharedLayersModel
from tests.torch.utils import compare_with_reference_file
from tests.torch.utils import to_comparable_nx_graph

REF_DIR = TEST_ROOT / "torch" / "data" / "function_hook" / "quantization" / "test_quantized_graphs"


@dataclass
class ModelDesc:
    model_name: str
    model_builder: Callable[..., Any]
    input_sample_sizes: tuple[int, ...]


TEST_MODELS_DESC = [
    (ModelDesc("embedding_model", EmbeddingModel, [1, 10]), {}),
    (
        ModelDesc("rope_model", partial(RoPEModel, with_transpose=True, with_reshape=True), [1, 10]),
        {"model_type": ModelType.TRANSFORMER},
    ),
    (
        ModelDesc(
            "scaled_dot_product_attention_model",
            ScaledDotProductAttentionModel,
            {"query": [1, 8, 16], "key": [1, 8, 16], "value": [1, 8, 16]},
        ),
        {},
    ),
    (
        ModelDesc(
            "unbind_scaled_dot_product_attention_model",
            UnbindScaledDotProductAttentionModel,
            {"x": [3, 1, 8, 16]},
        ),
        {},
    ),
    (ModelDesc("shared_model", SharedLayersModel, [1, 1, 5, 6]), {}),
    (ModelDesc("alexnet", test_models.AlexNet, [1, 3, 32, 32]), {}),
    (ModelDesc("lenet", test_models.LeNet, [1, 3, 32, 32]), {}),
    (ModelDesc("resnet18", test_models.ResNet18, [1, 3, 32, 32]), {}),
    (ModelDesc("vgg16", partial(test_models.VGG, "VGG16"), [1, 3, 32, 32]), {}),
    (ModelDesc("inception", test_models.GoogLeNet, [1, 3, 32, 32]), {}),
    (ModelDesc("densenet121", test_models.DenseNet121, [1, 3, 32, 32]), {}),
    (
        ModelDesc(
            "inception_v3", partial(test_models.Inception3, aux_logits=True, transform_input=True), [2, 3, 299, 299]
        ),
        {},
    ),
    (ModelDesc("squeezenet1_1", test_models.squeezenet1_1, [1, 3, 32, 32]), {}),
    (ModelDesc("shufflenetv2", partial(test_models.ShuffleNetV2, net_size=0.5), [1, 3, 32, 32]), {}),
    # TODO(AlexanderDokuchaev): too long without disabled tracing of ssd_head - no_nncf_trace()
    # (ModelDesc("ssd_vgg", test_models.ssd_vgg300, [2, 3, 300, 300]), {}),
    # (ModelDesc("ssd_mobilenet", test_models.ssd_mobilenet, [2, 3, 300, 300]), {}),
    (ModelDesc("mobilenet_v2", mobilenet_v2, [1, 3, 32, 32]), {}),
    (ModelDesc("mobilenet_v3_small", mobilenet_v3_small, [1, 3, 32, 32]), {}),
    (ModelDesc("unet", test_models.UNet, [1, 3, 360, 480]), {}),
]


@pytest.mark.parametrize(
    ("desc", "quantization_parameters"), TEST_MODELS_DESC, ids=[m[0].model_name for m in TEST_MODELS_DESC]
)
def test_quantized_graphs(desc: ModelDesc, quantization_parameters: dict[str, Any], regen_ref_data: bool):
    model = desc.model_builder().eval()

    if isinstance(desc.input_sample_sizes, dict):
        example_input = {}
        for name, size in desc.input_sample_sizes.items():
            example_input[name] = torch.ones(size)
    else:
        example_input = torch.ones(desc.input_sample_sizes)

    quantization_parameters["advanced_parameters"] = nncf.AdvancedQuantizationParameters(disable_bias_correction=True)
    quantization_parameters["subset_size"] = 1

    q_model = nncf.quantize(model, nncf.Dataset([example_input]), **quantization_parameters)

    nncf_graph = build_nncf_graph(q_model, example_input)
    nx_graph = to_comparable_nx_graph(nncf_graph)
    dot_nncf_graph = to_pydot(nx_graph)
    ref_file = REF_DIR / f"{desc.model_name}.dot"
    compare_with_reference_file(str(dot_nncf_graph), ref_file, regen_ref_data)
