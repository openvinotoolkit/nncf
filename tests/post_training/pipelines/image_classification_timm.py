# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import os

import numpy as np
import onnx
import openvino.runtime as ov
import timm
import torch
import tqdm
from openvino.tools.mo import convert_model
from sklearn.metrics import accuracy_score
from timm.layers.config import set_fused_attn
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import nncf
from nncf.experimental.torch.replace_custom_modules.timm_custom_modules import (
    replace_timm_custom_modules_with_torch_native,
)
from tests.post_training.pipelines.base import DEFAULT_VAL_THREADS
from tests.post_training.pipelines.base import OV_BACKENDS
from tests.post_training.pipelines.base import PT_BACKENDS
from tests.post_training.pipelines.base import BackendType
from tests.post_training.pipelines.base import BaseTestPipeline

# Disable using aten::scaled_dot_product_attention
set_fused_attn(False, False)


class ImageClassificationTimm(BaseTestPipeline):
    """Pipeline for Image Classification model from timm repository"""

    def prepare_model(self) -> None:
        timm_model = timm.create_model(self.model_id, num_classes=1000, in_chans=3, pretrained=True, checkpoint_path="")
        timm_model = replace_timm_custom_modules_with_torch_native(timm_model)
        self.model_cfg = timm_model.default_cfg
        self.input_size = [1] + list(timm_model.default_cfg["input_size"])
        self.dummy_tensor = torch.rand(self.input_size)

        if self.backend in PT_BACKENDS:
            self.model = timm_model

        if self.backend == BackendType.ONNX:
            onnx_path = self.output_model_dir / "model_fp32.onnx"
            torch.onnx.export(
                timm_model,
                self.dummy_tensor,
                onnx_path,
                export_params=True,
                opset_version=13,
                do_constant_folding=False,
            )
            self.model = onnx.load(onnx_path)
            self.input_name = self.model.graph.input[0].name

        if self.backend in OV_BACKENDS:
            self.model = convert_model(timm_model, example_input=self.dummy_tensor, input_shape=self.input_size)
            self.input_name = list(inp.get_any_name() for inp in self.model.inputs)[0]

        self._dump_model_fp32()

    def _dump_model_fp32(self) -> None:
        """Dump IRs of fp32 models, to help debugging."""
        if self.backend in PT_BACKENDS:
            ov_model = convert_model(self.model, example_input=self.dummy_tensor, input_shape=self.input_size)
            ov.serialize(ov_model, self.output_model_dir / "model_fp32.xml")

        if self.backend == BackendType.ONNX:
            onnx_path = self.output_model_dir / "model_fp32.onnx"
            ov_model = convert_model(onnx_path)
            ov.serialize(ov_model, self.output_model_dir / "model_fp32.xml")

        if self.backend in OV_BACKENDS:
            ov.serialize(self.model, self.output_model_dir / "model_fp32.xml")

    def prepare_preprocessor(self) -> None:
        config = self.model_cfg
        transformations_list = []
        normalize = transforms.Normalize(mean=config["mean"], std=config["std"])
        input_size = config["input_size"]

        RESIZE_MODE_MAP = {
            "bilinear": InterpolationMode.BILINEAR,
            "bicubic": InterpolationMode.BICUBIC,
            "nearest": InterpolationMode.NEAREST,
        }

        if "fixed_input_size" in config and not config["fixed_input_size"]:
            resize_size = tuple(int(x / config["crop_pct"]) for x in input_size[-2:])
            resize = transforms.Resize(resize_size, interpolation=RESIZE_MODE_MAP[config["interpolation"]])
            transformations_list.append(resize)
        transformations_list.extend([transforms.CenterCrop(input_size[-2:]), transforms.ToTensor(), normalize])

        self.transform = transforms.Compose(transformations_list)

    def get_transform_calibration_fn(self):
        if self.backend in PT_BACKENDS:

            def transform_fn(data_item):
                images, _ = data_item
                return images

        else:

            def transform_fn(data_item):
                images, _ = data_item
                return {self.input_name: np.array(images, dtype=np.float32)}

        return transform_fn

    def prepare_calibration_dataset(self):
        dataset = datasets.ImageFolder(root=self.data_dir / "imagenet" / "val", transform=self.transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=2, shuffle=False)

        self.calibration_dataset = nncf.Dataset(loader, self.get_transform_calibration_fn())

    def _validate(self):
        val_dataset = datasets.ImageFolder(root=self.data_dir / "imagenet" / "val", transform=self.transform)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=2, shuffle=False)

        dataset_size = len(val_loader)

        predictions = [0] * dataset_size
        references = [-1] * dataset_size

        core = ov.Core()

        if os.environ.get("CPU_THREADS_NUM"):
            # Set CPU_THREADS_NUM for OpenVINO inference
            cpu_threads_num = os.environ.get("CPU_THREADS_NUM")
            core.set_property("CPU", properties={"CPU_THREADS_NUM": str(cpu_threads_num)})

        ov_model = core.read_model(self.path_quantized_ir)
        compiled_model = core.compile_model(ov_model)

        jobs = int(os.environ.get("NUM_VAL_THREADS", DEFAULT_VAL_THREADS))
        infer_queue = ov.AsyncInferQueue(compiled_model, jobs)

        # Disable tqdm for Jenkins
        disable_tqdm = os.environ.get("JENKINS_HOME") is not None

        with tqdm.tqdm(total=dataset_size, desc="Validation", disable=disable_tqdm) as pbar:

            def process_result(request, userdata):
                output_data = request.get_output_tensor().data
                predicted_label = np.argmax(output_data, axis=1)
                predictions[userdata] = [predicted_label]
                pbar.update()

            infer_queue.set_callback(process_result)

            for i, (images, target) in enumerate(val_loader):
                # W/A for memory leaks when using torch DataLoader and OpenVINO
                image_copies = copy.deepcopy(images.numpy())
                infer_queue.start_async(image_copies, userdata=i)
                references[i] = target

            infer_queue.wait_all()

        predictions = np.concatenate(predictions, axis=0)
        references = np.concatenate(references, axis=0)
        acc_top1 = accuracy_score(predictions, references)

        self.run_info.metric_name = "Acc@1"
        self.run_info.metric_value = acc_top1
