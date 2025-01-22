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

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Tuple

import pytest
import torch
import torch.fx
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from torch.ao.quantization.quantizer.quantizer import Quantizer
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
from torch.ao.quantization.quantizer.x86_inductor_quantizer import get_default_x86_inductor_quantization_config

import nncf
from nncf.experimental.torch.fx.nncf_graph_builder import GraphConverter
from nncf.experimental.torch.fx.quantization.quantize_pt2e import quantize_pt2e
from tests.torch import test_models
from tests.torch.fx.helpers import get_torch_fx_model
from tests.torch.test_compressed_graph import check_graph
from tests.torch.test_models.synthetic import ShortTransformer
from tests.torch.test_models.synthetic import YOLO11N_SDPABlock

FX_QUANTIZED_DIR_NAME = Path("fx") / "experimental"


@dataclass
class ModelCase:
    model_builder: Callable[[], torch.nn.Module]
    model_id: str
    input_shape: Tuple[int]


def torchvision_model_case(model_id: str, input_shape: Tuple[int,]):
    model = getattr(models, model_id)
    return ModelCase(partial(model, weights=None), model_id, input_shape)


TEST_MODELS = (
    torchvision_model_case("resnet18", (1, 3, 224, 224)),
    torchvision_model_case("mobilenet_v3_small", (1, 3, 224, 224)),
    torchvision_model_case("vit_b_16", (1, 3, 224, 224)),
    torchvision_model_case("swin_v2_s", (1, 3, 224, 224)),
    ModelCase(test_models.UNet, "unet", [1, 3, 224, 224]),
    ModelCase(partial(ShortTransformer, 5, 10), "synthetic_transformer", [5]),
    ModelCase(YOLO11N_SDPABlock, "yolo11n_sdpa_block", YOLO11N_SDPABlock.INPUT_SIZE),
)


def get_dot_filename(model_name):
    return model_name + ".dot"


def get_x86_quantizer(*args, **kwarsg) -> X86InductorQuantizer:
    quantizer = X86InductorQuantizer()
    quantizer.set_global(get_default_x86_inductor_quantization_config())
    return quantizer


TEST_MODELS_QUANIZED = (
    (ModelCase(test_models.UNet, "unet", [1, 3, 224, 224]), {}, {}),
    (torchvision_model_case("resnet18", (1, 3, 224, 224)), {}, {}),
    (torchvision_model_case("mobilenet_v3_small", (1, 3, 224, 224)), {}, {}),
    (
        torchvision_model_case("vit_b_16", (1, 3, 224, 224)),
        {"model_type": nncf.ModelType.TRANSFORMER},
        {"smooth_quant": True},
    ),
    (
        torchvision_model_case("swin_v2_s", (1, 3, 224, 224)),
        {"model_type": nncf.ModelType.TRANSFORMER},
        {"smooth_quant": True},
    ),
    (
        ModelCase(partial(ShortTransformer, 5, 10), "synthetic_transformer", [5]),
        {"model_type": nncf.ModelType.TRANSFORMER},
        {"smooth_quant": True},
    ),
    (
        ModelCase(YOLO11N_SDPABlock, "yolo11n_sdpa_block", YOLO11N_SDPABlock.INPUT_SIZE),
        {"model_type": nncf.ModelType.TRANSFORMER},
        {"smooth_quant": True},
    ),
)


@pytest.mark.parametrize("quantizer_builder", [get_x86_quantizer])
class TestModelQuantization:
    @staticmethod
    def _build_torch_fx_model(model_case: ModelCase) -> Tuple[torch.fx.GraphModule, torch.Tensor]:
        model = model_case.model_builder()
        dtype = torch.int32 if model_case.model_id == "synthetic_transformer" else torch.float32
        example_input = torch.ones(model_case.input_shape, dtype=dtype)
        fx_model = get_torch_fx_model(model, example_input)
        return fx_model, example_input

    @staticmethod
    def _get_calibration_dataset(example_input: torch.Tensor) -> nncf.Dataset:
        def transform_fn(data_item):
            return data_item.to("cpu")

        return nncf.Dataset([example_input], transform_fn)

    @pytest.mark.parametrize(
        ("model_case", "quantizer_params", "pt2e_params"),
        TEST_MODELS_QUANIZED,
        ids=[m[0].model_id for m in TEST_MODELS_QUANIZED],
    )
    def test_quantized_model(
        self,
        quantizer_builder: Callable[[Tuple[Any, ...]], Quantizer],
        model_case: ModelCase,
        quantizer_params,
        pt2e_params,
    ):
        fx_model, example_input = self._build_torch_fx_model(model_case)
        calibration_dataset = self._get_calibration_dataset(example_input)

        quantizer = quantizer_builder(**quantizer_params)
        quantized_model = quantize_pt2e(
            fx_model,
            quantizer,
            calibration_dataset=calibration_dataset,
            fast_bias_correction=None,  # BC is disabled
            fold_quantize=True,
            do_copy=True,
            **pt2e_params,
        )

        # Uncomment to visualize torch fx graph
        # from tests.torch.fx.helpers import visualize_fx_model
        # visualize_fx_model(quantized_model, f"{model_case.model_id}_int8.svg")

        nncf_graph = GraphConverter.create_nncf_graph(quantized_model)
        check_graph(
            nncf_graph,
            get_dot_filename(model_case.model_id),
            FX_QUANTIZED_DIR_NAME / quantizer.__class__.__name__,
            extended=True,
        )

        # Uncomment to visualize reference graphs
        # from torch.ao.quantization.quantize_pt2e import convert_pt2e
        # from torch.ao.quantization.quantize_pt2e import prepare_pt2e
        # from tests.torch.fx.helpers import visualize_fx_model
        # prepared_model = prepare_pt2e(fx_model, quantizer)
        # prepared_model(example_input)
        # ao_quantized_model = convert_pt2e(prepared_model)
        # visualize_fx_model(ao_quantized_model, f"{model_case.model_id}ao_int8.svg")
        # ao_nncf_graph = GraphConverter.create_nncf_graph(ao_quantized_model)
        # ao_nncf_graph.visualize_graph("ao_" + get_dot_filename(model_case.model_id))
