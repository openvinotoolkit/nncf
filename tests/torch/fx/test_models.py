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
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Tuple, Type

import openvino.torch  # noqa
import pytest
import torch
import torch.fx
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from torch._export import capture_pre_autograd_graph
from ultralytics.models.yolo import YOLO

from nncf.common.graph.graph import NNCFNodeName
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.utils.os import safe_open
from nncf.experimental.torch.fx.nncf_graph_builder import GraphConverter
from nncf.torch.dynamic_graph.patch_pytorch import disable_patching
from tests.shared.paths import TEST_ROOT
from tests.torch.test_compressed_graph import check_graph

FX_DIR_NAME = "fx"


@dataclass
class ModelCase:
    model_builder: Callable[[], torch.nn.Module]
    model_id: str
    input_shape: Tuple[int]


def torchvision_model_case(model_id: str, input_shape: Tuple[int,]):
    model = getattr(models, model_id)
    return ModelCase(partial(model, weights=None), model_id, input_shape)


def yolo_v8_case(model_id, input_shape):
    def get_model() -> torch.nn.Module:
        model_config = model_id + ".yaml"
        model = YOLO(model_config)
        model = model.model
        model.eval()
        # Warmup model
        model(torch.ones(input_shape))
        return model

    return ModelCase(get_model, model_id, input_shape)


TEST_MODELS = (
    torchvision_model_case("resnet18", (1, 3, 224, 224)),
    torchvision_model_case("mobilenet_v3_small", (1, 3, 224, 224)),
    torchvision_model_case("vit_b_16", (1, 3, 224, 224)),
    torchvision_model_case("swin_v2_s", (1, 3, 224, 224)),
    yolo_v8_case("yolov8n", (1, 3, 224, 224)),
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
    model_name: str, model_metatypes: Dict[NNCFNodeName, Type[OperatorMetatype]]
) -> Dict[NNCFNodeName, Type[OperatorMetatype]]:

    model_json_name = get_json_filename(model_name)
    complete_path = get_full_path_to_json(model_json_name)

    json_parent_dir = Path(complete_path).parent

    if os.getenv("NNCF_TEST_REGEN_JSON") is not None:
        if not os.path.exists(json_parent_dir):
            os.makedirs(json_parent_dir)
        with safe_open(complete_path, "w") as file:
            json.dump(model_metatypes, file)

    with safe_open(complete_path, "r") as file:
        return json.load(file)


@pytest.mark.parametrize("test_case", TEST_MODELS, ids=[m.model_id for m in TEST_MODELS])
def test_models(test_case: ModelCase):
    with disable_patching():
        device = torch.device("cpu")
        model_name = test_case.model_id
        model = test_case.model_builder()
        model.to(device)

        with torch.no_grad():
            ex_input = torch.ones(test_case.input_shape)
            model.eval()
            exported_model = capture_pre_autograd_graph(model, args=(ex_input,))
        nncf_graph = GraphConverter.create_nncf_graph(exported_model)

        # Check NNCFGrpah
        dot_filename = get_dot_filename(model_name)
        check_graph(nncf_graph, dot_filename, FX_DIR_NAME)

        # Check metatypes
        model_metatypes = {n.node_name: n.metatype.name for n in nncf_graph.get_all_nodes()}
        ref_metatypes = get_ref_metatypes_from_json(model_name, model_metatypes)
        assert model_metatypes == ref_metatypes
