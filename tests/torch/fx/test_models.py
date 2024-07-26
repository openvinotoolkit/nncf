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


import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Type

import openvino.torch  # noqa
import pytest
import torch
import torch.fx
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from diffusers import StableDiffusionPipeline
from torch._export import capture_pre_autograd_graph
from ultralytics.models.yolo import YOLO

from nncf.common.graph.graph import NNCFNodeName
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.utils.os import safe_open
from nncf.experimental.torch.fx.nncf_graph_builder import GraphConverter
from nncf.torch.dynamic_graph.patch_pytorch import disable_patching
from nncf.torch.graph.graph import PTNNCFGraph
from tests.shared.paths import TEST_ROOT
from tests.torch.test_compressed_graph import check_graph


@pytest.fixture(name="fx_dir")
def fx_dir_fixture(request):
    fx_dir_name = "fx"
    return fx_dir_name


@dataclass
class ModelCase:
    model: torch.nn.Module
    model_id: str
    ex_input: Any


def stable_diffusion_model_builder(model_id: str, prompt: str, input_shape: Tuple[int, ...]):
    pipe = StableDiffusionPipeline.from_pretrained(model_id).to("cpu")

    text_encoder = pipe.text_encoder
    text_encoder.eval()

    unet = pipe.unet
    unet.eval()

    vae = pipe.vae
    vae.eval()

    tokenizer = pipe.tokenizer

    ex_input = torch.ones(input_shape)
    device = "cpu"
    prompt = "A truly masterpeice of a work"
    text_inputs = tokenizer(prompt, return_tensors="pt")
    text_input_ids = text_inputs.input_ids.to("cpu")
    text_embeddings = text_encoder(text_input_ids)[0]

    batch_size = text_input_ids.shape[0]
    latents = torch.ones(
        (batch_size, unet.config.in_channels, pipe.unet.sample_size, pipe.unet.sample_size), device=device
    )
    dummy_t = torch.tensor([0.0], device=device)
    unet_input = (
        latents,
        dummy_t,
        text_embeddings,
    )

    return (
        model_id,
        ModelCase(unet, "SD_UNET", unet_input),
        ModelCase(vae, "SD_VAE", ex_input),
        ModelCase(text_encoder, "SD_Text_Encoder", text_input_ids),
    )


def torchvision_model_builder(model_id: str, input_shape: Tuple[int, ...]):
    model = getattr(models, model_id)(weights=None)
    ex_input = torch.ones(input_shape)
    return ModelCase(model, model_id, ex_input)


def ultralytics_model_builder(model_id: str, input_shape: Tuple[int, ...]):
    model_config = model_id + ".yaml"  # Initialize the model with random weights instead of downloading them.
    model = YOLO(model_config)
    model = model.model
    ex_input = torch.ones(input_shape)
    model.eval()
    _ = model(ex_input)  # inferring from model to avoid anchor mutation in YOLOv8
    return ModelCase(model, model_id, ex_input)


TEST_MODELS = (
    torchvision_model_builder("resnet18", (1, 3, 224, 224)),
    torchvision_model_builder("mobilenet_v3_small", (1, 3, 224, 224)),
    torchvision_model_builder("vit_b_16", (1, 3, 224, 224)),
    torchvision_model_builder("swin_v2_s", (1, 3, 224, 224)),
    ultralytics_model_builder("yolov8n", (1, 3, 224, 224)),
    stable_diffusion_model_builder("stabilityai/stable-diffusion-2-1-base", "Hello world", (1, 3, 224, 224)),
)


def get_dot_filename(model_name):
    return model_name + ".dot"


def get_json_filename(model_name):
    return model_name + ".json"


def get_full_path_to_json(model_json_name: str) -> str:
    path_to_dir = TEST_ROOT / "torch" / "data" / "reference_graphs" / "fx" / "reference_metatypes"
    path_to_json = path_to_dir / model_json_name
    return path_to_json


def get_ref_metatypes_from_json(
    model_name: str, model_metatypes: Dict[NNCFNodeName, Type[OperatorMetatype]], fx_dir: str
) -> Dict[NNCFNodeName, Type[OperatorMetatype]]:

    model_json_name = get_json_filename(model_name)
    complete_path = get_full_path_to_json(model_json_name)

    json_parent_dir = Path(complete_path).parent

    pipeline_model_dir = fx_dir.split("/")
    if len(pipeline_model_dir) > 1:
        json_parent_dir = json_parent_dir / "/".join(pipeline_model_dir[1:])
        complete_path = json_parent_dir / model_json_name

    if os.getenv("NNCF_TEST_REGEN_JSON") is not None:
        if not os.path.exists(json_parent_dir):
            os.makedirs(json_parent_dir)
        with safe_open(complete_path, "w") as file:
            json.dump(model_metatypes, file)

    with safe_open(complete_path, "r") as file:
        return json.load(file)


def compare_nncf_graph_model(model: PTNNCFGraph, model_name: str, path_to_dot: str):
    dot_filename = get_dot_filename(model_name)
    check_graph(model, dot_filename, path_to_dot)


def run_test(test_case: ModelCase, fx_dir):
    device = torch.device("cpu")
    with disable_patching():
        model_name = test_case.model_id
        model = test_case.model
        model.to(device)
        model.eval()

        with torch.no_grad():
            ex_input = test_case.ex_input
            path_to_dot = fx_dir
            if not isinstance(ex_input, tuple):
                ex_input = (ex_input,)
            exported_model = capture_pre_autograd_graph(model, args=ex_input)
            nncf_graph = GraphConverter.create_nncf_graph(exported_model)
            compare_nncf_graph_model(nncf_graph, model_name, path_to_dot)
            model_metatypes = {n.node_name: n.metatype.name for n in nncf_graph.get_all_nodes()}
            ref_metatypes = get_ref_metatypes_from_json(model_name, model_metatypes, fx_dir)
            assert model_metatypes == ref_metatypes


@pytest.mark.parametrize("test_case", TEST_MODELS)
def test_models(test_case: ModelCase, fx_dir):
    if isinstance(test_case, tuple):
        for model in test_case[1:]:
            run_test(model, fx_dir + "/" + test_case[0])
    else:
        run_test(test_case, fx_dir)
