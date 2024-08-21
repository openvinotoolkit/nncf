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
from typing import Any, Dict

import torchvision.models as models

import nncf
from nncf import AdvancedQuantizationParameters
from nncf import AdvancedSmoothQuantParameters
from nncf import QuantizationPreset
from nncf.parameters import ModelType
from tests.torch.fx.performance_check.model_builders.base import BaseModelBuilder
from tests.torch.fx.performance_check.model_builders.stable_diffusion import StableDiffusion2UnetBuilder
from tests.torch.fx.performance_check.model_builders.torchvision import TorchvisionModelBuilder
from tests.torch.fx.performance_check.model_builders.ultralytics import UltralyticsModelBuilder


@dataclass
class ModelConfig:
    model_builder: BaseModelBuilder
    quantization_params: Dict[str, Any]
    num_iters: int = 500


MODEL_SCOPE = {
    "vit_b_16": ModelConfig(
        TorchvisionModelBuilder(models.vit_b_16, models.ViT_B_16_Weights.DEFAULT),
        {
            "model_type": ModelType.TRANSFORMER,
        },
    ),
    "swin_v2_s": ModelConfig(
        TorchvisionModelBuilder(models.swin_v2_s, models.Swin_V2_S_Weights.DEFAULT),
        {
            "model_type": ModelType.TRANSFORMER,
        },
    ),
    "resnet18": ModelConfig(TorchvisionModelBuilder(models.resnet18, models.ResNet18_Weights.DEFAULT), {}, 2000),
    "resnet50": ModelConfig(TorchvisionModelBuilder(models.resnet50, models.ResNet50_Weights.DEFAULT), {}),
    "mobilenet_v2": ModelConfig(
        TorchvisionModelBuilder(models.mobilenet_v2, models.MobileNet_V2_Weights.DEFAULT),
        {
            "preset": QuantizationPreset.MIXED,
            "fast_bias_correction": True,
        },
    ),
    "mobilenet_v3_small": ModelConfig(
        TorchvisionModelBuilder(models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.DEFAULT),
        {
            "preset": QuantizationPreset.MIXED,
            "fast_bias_correction": True,
        },
    ),
    "sd2_unet": ModelConfig(
        StableDiffusion2UnetBuilder(),
        {
            "model_type": ModelType.TRANSFORMER,
            "advanced_parameters": AdvancedQuantizationParameters(
                smooth_quant_alphas=AdvancedSmoothQuantParameters(-1, -1)
            ),
            "ignored_scope": nncf.IgnoredScope(
                types=["linear"],
            ),
        },
        num_iters=1,
    ),
    "yolov8n": ModelConfig(
        UltralyticsModelBuilder("yolov8n"),
        {
            "ignored_scope": nncf.IgnoredScope(
                types=["mul", "sub", "sigmoid"],
                subgraphs=[
                    nncf.Subgraph(
                        inputs=["cat_13", "cat_14", "cat_15"],
                        outputs=["output"],
                    )
                ],
            )
        },
        num_iters=10,
    ),
}
