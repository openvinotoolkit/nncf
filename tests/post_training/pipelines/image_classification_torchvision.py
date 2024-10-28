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
from typing import Any, Callable, Tuple

import numpy as np
import onnx
import openvino as ov
import torch
from torch._export import capture_pre_autograd_graph
from torchvision import models

from nncf.torch import disable_patching
from tests.post_training.pipelines.base import PT_BACKENDS
from tests.post_training.pipelines.base import BackendType
from tests.post_training.pipelines.image_classification_base import ImageClassificationBase


def _capture_pre_autograd_module(model: torch.nn.Module, args: Tuple[Any, ...]) -> torch.fx.GraphModule:
    return capture_pre_autograd_graph(model, args)


def _export_graph_module(model: torch.nn.Module, args: Tuple[Any, ...]) -> torch.fx.GraphModule:
    return torch.export.export(model, args).module()


@dataclass
class VisionModelParams:
    weights: models.WeightsEnum
    export_fn: Callable[[torch.nn.Module, Tuple[Any, ...]], torch.fx.GraphModule]


class ImageClassificationTorchvision(ImageClassificationBase):
    """Pipeline for Image Classification model from torchvision repository"""

    models_vs_model_params = {
        models.resnet18: VisionModelParams(models.ResNet18_Weights.DEFAULT, _capture_pre_autograd_module),
        models.mobilenet_v3_small: VisionModelParams(
            models.MobileNet_V3_Small_Weights.DEFAULT, _capture_pre_autograd_module
        ),
        models.vit_b_16: VisionModelParams(models.ViT_B_16_Weights.DEFAULT, _export_graph_module),
        models.swin_v2_s: VisionModelParams(models.Swin_V2_S_Weights.DEFAULT, _export_graph_module),
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_params: VisionModelParams = None
        self.input_name: str = None

    def prepare_model(self) -> None:
        model_cls = models.__dict__.get(self.model_id)
        self.model_params = self.models_vs_model_params[model_cls]
        model = model_cls(weights=self.model_params.weights)
        model.eval()

        default_input_size = [self.batch_size, 3, 224, 224]
        self.dummy_tensor = self.model_params.weights.transforms()(torch.rand(default_input_size))
        self.static_input_size = list(self.dummy_tensor.shape)

        self.input_size = self.static_input_size.copy()
        if self.batch_size > 1:  # Dynamic batch_size shape export
            self.input_size[0] = -1

        if self.backend == BackendType.FX_TORCH:
            with torch.no_grad():
                with disable_patching():
                    self.model = self.model_params.export_fn(model, (self.dummy_tensor,))

        elif self.backend in PT_BACKENDS:
            self.model = model

        if self.backend == BackendType.ONNX:
            onnx_path = self.fp32_model_dir / "model_fp32.onnx"
            additional_kwargs = {}
            if self.batch_size > 1:
                additional_kwargs["input_names"] = ["image"]
                additional_kwargs["dynamic_axes"] = {"image": {0: "batch"}}
            torch.onnx.export(
                model, self.dummy_tensor, onnx_path, export_params=True, opset_version=13, **additional_kwargs
            )
            self.model = onnx.load(onnx_path)
            self.input_name = self.model.graph.input[0].name

        elif self.backend in [BackendType.OV, BackendType.FP32]:
            with torch.no_grad():
                with disable_patching():
                    m = torch.export.export(model, args=(self.dummy_tensor,))
                self.model = ov.convert_model(m, example_input=self.dummy_tensor, input=self.input_size)
            self.input_name = list(inp.get_any_name() for inp in self.model.inputs)[0]

        self._dump_model_fp32()

        # Set device after dump fp32 model
        if self.backend == BackendType.CUDA_TORCH:
            self.model.cuda()
            self.dummy_tensor = self.dummy_tensor.cuda()

    def _dump_model_fp32(self) -> None:
        """Dump IRs of fp32 models, to help debugging."""
        if self.backend in PT_BACKENDS:
            with disable_patching():
                ov_model = ov.convert_model(
                    torch.export.export(self.model, args=(self.dummy_tensor,)),
                    example_input=self.dummy_tensor,
                    input=self.input_size,
                )
            ov.serialize(ov_model, self.fp32_model_dir / "model_fp32.xml")

        if self.backend == BackendType.FX_TORCH:
            exported_model = torch.export.export(self.model, (self.dummy_tensor,))
            ov_model = ov.convert_model(exported_model, example_input=self.dummy_tensor, input=self.input_size)
            ov.serialize(ov_model, self.fp32_model_dir / "fx_model_fp32.xml")

        if self.backend in [BackendType.FP32, BackendType.OV]:
            ov.serialize(self.model, self.fp32_model_dir / "model_fp32.xml")

    def prepare_preprocessor(self) -> None:
        self.transform = self.model_params.weights.transforms()

    def get_transform_calibration_fn(self):
        if self.backend in [BackendType.FX_TORCH] + PT_BACKENDS:
            device = torch.device("cuda" if self.backend == BackendType.CUDA_TORCH else "cpu")

            def transform_fn(data_item):
                images, _ = data_item
                return images.to(device)

        else:

            def transform_fn(data_item):
                images, _ = data_item
                return {self.input_name: np.array(images, dtype=np.float32)}

        return transform_fn
