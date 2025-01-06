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

import numpy as np
import onnx
import openvino as ov
import timm
import torch
from timm.data.transforms_factory import transforms_imagenet_eval
from timm.layers.config import set_fused_attn

from tests.post_training.pipelines.base import OV_BACKENDS
from tests.post_training.pipelines.base import PT_BACKENDS
from tests.post_training.pipelines.base import BackendType
from tests.post_training.pipelines.image_classification_base import ImageClassificationBase

# Disable using aten::scaled_dot_product_attention
set_fused_attn(False, False)


class ImageClassificationTimm(ImageClassificationBase):
    """Pipeline for Image Classification model from timm repository"""

    def prepare_model(self) -> None:
        timm_model = timm.create_model(self.model_id, num_classes=1000, in_chans=3, pretrained=True, checkpoint_path="")
        timm_model.eval()
        self.model_cfg = timm_model.default_cfg
        self.input_size = [self.batch_size] + list(timm_model.default_cfg["input_size"])
        self.dummy_tensor = torch.rand(self.input_size)
        if self.batch_size > 1:  # Dynamic batch_size shape export
            self.input_size[0] = -1

        if self.backend in PT_BACKENDS:
            self.model = timm_model

        if self.backend == BackendType.ONNX:
            onnx_path = self.fp32_model_dir / "model_fp32.onnx"
            additional_kwargs = {}
            if self.batch_size > 1:
                additional_kwargs["input_names"] = ["image"]
                additional_kwargs["dynamic_axes"] = {"image": {0: "batch"}}
            torch.onnx.export(
                timm_model, self.dummy_tensor, onnx_path, export_params=True, opset_version=13, **additional_kwargs
            )
            self.model = onnx.load(onnx_path)
            self.input_name = self.model.graph.input[0].name

        if self.backend in OV_BACKENDS + [BackendType.FP32]:
            self.model = ov.convert_model(timm_model, example_input=self.dummy_tensor, input=self.input_size)
            self.input_name = list(inp.get_any_name() for inp in self.model.inputs)[0]

        self._dump_model_fp32()

        # Set device after dump fp32 model
        if self.backend == BackendType.CUDA_TORCH:
            self.model.cuda()
            self.dummy_tensor = self.dummy_tensor.cuda()

    def _dump_model_fp32(self) -> None:
        """Dump IRs of fp32 models, to help debugging."""
        if self.backend in PT_BACKENDS:
            ov_model = ov.convert_model(self.model, example_input=self.dummy_tensor, input=self.input_size)
            ov.serialize(ov_model, self.fp32_model_dir / "model_fp32.xml")

        if self.backend == BackendType.ONNX:
            onnx_path = self.fp32_model_dir / "model_fp32.onnx"
            ov_model = ov.convert_model(onnx_path)
            ov.serialize(ov_model, self.fp32_model_dir / "model_fp32.xml")

        if self.backend in OV_BACKENDS + [BackendType.FP32]:
            ov.serialize(self.model, self.fp32_model_dir / "model_fp32.xml")

    def prepare_preprocessor(self) -> None:
        config = self.model_cfg
        self.transform = transforms_imagenet_eval(
            img_size=config["input_size"][-2:],
            crop_pct=config["crop_pct"],
            crop_mode=config["crop_mode"],
            interpolation=config["interpolation"],
            use_prefetcher=False,
            mean=config["mean"],
            std=config["std"],
        )

    def get_transform_calibration_fn(self):
        if self.backend in PT_BACKENDS:
            device = torch.device("cuda" if self.backend == BackendType.CUDA_TORCH else "cpu")

            def transform_fn(data_item):
                images, _ = data_item
                return images.to(device)

        else:

            def transform_fn(data_item):
                images, _ = data_item
                return {self.input_name: np.array(images, dtype=np.float32)}

        return transform_fn
