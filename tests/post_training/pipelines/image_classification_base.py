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

import copy
import os

import numpy as np
import openvino as ov
import torch
from sklearn.metrics import accuracy_score
from torchvision import datasets

import nncf
from nncf.common.logging.track_progress import track
from tests.post_training.pipelines.base import DEFAULT_VAL_THREADS
from tests.post_training.pipelines.base import FX_BACKENDS
from tests.post_training.pipelines.base import PTQTestPipeline


class ImageClassificationBase(PTQTestPipeline):
    """Base pipeline for Image Classification models"""

    def prepare_calibration_dataset(self):
        dataset = datasets.ImageFolder(root=self.data_dir / "imagenet" / "val", transform=self.transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=2, shuffle=False)

        self.calibration_dataset = nncf.Dataset(loader, self.get_transform_calibration_fn())

    def _validate_ov(
        self,
        val_loader: torch.utils.data.DataLoader,
        predictions: np.ndarray,
        references: np.ndarray,
        dataset_size: int,
    ):
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
                pbar.update(advance=1)

            infer_queue.set_callback(process_result)

            for i, (images, target) in enumerate(val_loader):
                # W/A for memory leaks when using torch DataLoader and OpenVINO
                image_copies = copy.deepcopy(images.numpy())
                infer_queue.start_async(image_copies, userdata=i)
                references[i] = target

            infer_queue.wait_all()
        return predictions, references

    def _validate_torch_compile(
        self, val_loader: torch.utils.data.DataLoader, predictions: np.ndarray, references: np.ndarray
    ):
        compiled_model = torch.compile(self.compressed_model.cpu(), backend="openvino", options={"aot_autograd": True})
        for i, (images, target) in enumerate(val_loader):
            # W/A for memory leaks when using torch DataLoader and OpenVINO
            pred = compiled_model(images)
            pred = torch.argmax(pred, dim=1)
            predictions[i] = pred.numpy()
            references[i] = target.numpy()
        return predictions, references

    def _validate(self) -> None:
        val_dataset = datasets.ImageFolder(root=self.data_dir / "imagenet" / "val", transform=self.transform)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=2, shuffle=False)

        dataset_size = len(val_loader)

        # Initialize result tensors for async inference support.
        predictions = np.zeros(dataset_size)
        references = -1 * np.ones(dataset_size)

        if self.backend in FX_BACKENDS:
            predictions, references = self._validate_torch_compile(val_loader, predictions, references)
        else:
            predictions, references = self._validate_ov(val_loader, predictions, references, dataset_size)

        acc_top1 = accuracy_score(predictions, references)

        self.run_info.metric_name = "Acc@1"
        self.run_info.metric_value = acc_top1
