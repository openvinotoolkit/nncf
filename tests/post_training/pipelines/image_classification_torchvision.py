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

import copy
import os

import numpy as np
import openvino as ov
import torch
from sklearn.metrics import accuracy_score
from torch._export import capture_pre_autograd_graph
from torchvision import datasets
from torchvision import models

import nncf
from nncf.common.logging.track_progress import track
from nncf.torch import disable_patching
from tests.post_training.pipelines.base import DEFAULT_VAL_THREADS
from tests.post_training.pipelines.base import PT_BACKENDS
from tests.post_training.pipelines.base import BackendType
from tests.post_training.pipelines.base import PTQTestPipeline


class ImageClassificationTorchvision(PTQTestPipeline):
    """Pipeline for Image Classification model from torchvision repository"""

    models_vs_imagenet_weights = {
        models.resnet18: models.ResNet18_Weights.DEFAULT,
        models.mobilenet_v3_small: models.MobileNet_V3_Small_Weights.DEFAULT,
        models.vit_b_16: models.ViT_B_16_Weights.DEFAULT,
        models.swin_v2_s: models.Swin_V2_S_Weights.DEFAULT,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_weights: models.WeightsEnum = None
        self.input_name: str = None

    def prepare_model(self) -> None:
        if self.backend not in [BackendType.FP32, BackendType.FX_TORCH, BackendType.OV] + PT_BACKENDS:
            raise RuntimeError(
                f"Torchvision classification models does not support quantization for {self.backend} backend."
            )

        model_cls = models.__dict__.get(self.model_id)
        self.model_weights = self.models_vs_imagenet_weights[model_cls]
        model = model_cls(weights=self.model_weights)
        model.eval()

        self.input_size = [self.batch_size, 3, 224, 224]
        self.dummy_tensor = torch.rand(self.input_size)

        if self.backend == BackendType.FX_TORCH:
            with torch.no_grad():
                with disable_patching():
                    self.model = capture_pre_autograd_graph(model, (torch.ones(self.input_size),))

        elif self.backend in PT_BACKENDS:
            self.model = model

        elif self.backend in [BackendType.OV, BackendType.FP32]:
            with torch.no_grad():
                self.model = ov.convert_model(model, example_input=self.dummy_tensor, input=self.input_size)
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
        self.transform = self.model_weights.transforms()

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

    def prepare_calibration_dataset(self):
        dataset = datasets.ImageFolder(root=self.data_dir / "imagenet" / "val", transform=self.transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=2, shuffle=False)

        self.calibration_dataset = nncf.Dataset(loader, self.get_transform_calibration_fn())

    def _validate(self):
        val_dataset = datasets.ImageFolder(root=self.data_dir / "imagenet" / "val", transform=self.transform)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=2, shuffle=False)

        dataset_size = len(val_loader)

        # Initialize result tensors for async inference support.
        predictions = np.zeros((dataset_size))
        references = -1 * np.ones((dataset_size))

        core = ov.Core()

        if os.environ.get("INFERENCE_NUM_THREADS"):
            # Set CPU_THREADS_NUM for OpenVINO inference
            inference_num_threads = os.environ.get("INFERENCE_NUM_THREADS")
            core.set_property("CPU", properties={"INFERENCE_NUM_THREADS": str(inference_num_threads)})

        ov_model = core.read_model(self.path_compressed_ir)
        compiled_model = core.compile_model(ov_model, device_name="CPU")

        jobs = int(os.environ.get("NUM_VAL_THREADS", DEFAULT_VAL_THREADS))
        infer_queue = ov.AsyncInferQueue(compiled_model, jobs)

        with track(total=dataset_size, description="Validation") as pbar:

            def process_result(request, userdata):
                output_data = request.get_output_tensor().data
                predicted_label = np.argmax(output_data, axis=1)
                predictions[userdata] = predicted_label
                pbar.progress.update(pbar.task, advance=1)

            infer_queue.set_callback(process_result)

            for i, (images, target) in enumerate(val_loader):
                # W/A for memory leaks when using torch DataLoader and OpenVINO
                image_copies = copy.deepcopy(images.numpy())
                infer_queue.start_async(image_copies, userdata=i)
                references[i] = target

            infer_queue.wait_all()

        acc_top1 = accuracy_score(predictions, references)

        self.run_info.metric_name = "Acc@1"
        self.run_info.metric_value = acc_top1
