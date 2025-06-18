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

os.environ["TORCHINDUCTOR_FREEZING"] = "1"

from itertools import islice

import numpy as np
import openvino as ov
import torch
from sklearn.metrics import accuracy_score
from torch.ao.quantization.quantize_pt2e import convert_pt2e
from torch.ao.quantization.quantize_pt2e import prepare_pt2e
from torch.ao.quantization.quantizer.quantizer import Quantizer as TorchAOQuantizer
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
from torch.ao.quantization.quantizer.x86_inductor_quantizer import get_default_x86_inductor_quantization_config
from torchvision import datasets

import nncf
from nncf import AdvancedQuantizationParameters
from nncf.common.logging.track_progress import track
from nncf.experimental.torch.fx import OpenVINOQuantizer
from nncf.experimental.torch.fx import quantize_pt2e
from nncf.torch import disable_patching
from tests.post_training.pipelines.base import DEFAULT_VAL_THREADS
from tests.post_training.pipelines.base import FX_BACKENDS
from tests.post_training.pipelines.base import BackendType
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
        if self.backend in [
            BackendType.FX_TORCH,
            BackendType.CUDA_FX_TORCH,
            BackendType.OV_QUANTIZER_AO,
            BackendType.OV_QUANTIZER_NNCF,
        ]:
            compiled_model = torch.compile(
                self.compressed_model.cpu(), backend="openvino", options={"aot_autograd": True}
            )
        else:
            compiled_model = torch.compile(self.compressed_model)
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
        return []

    def _compress_torch_ao(self, quantizer):
        with torch.no_grad(), disable_patching():
            prepared_model = prepare_pt2e(self.model, quantizer)
            subset_size = self.compression_params.get("subset_size", 300)
            for data in islice(self.calibration_dataset.get_inference_data(), subset_size):
                prepared_model(data)
            self.compressed_model = convert_pt2e(prepared_model, fold_quantize=False)

    def _compress_nncf_pt2e(self, quantizer):
        pt2e_kwargs = {}
        for key in (
            "subset_size",
            "fast_bias_correction",
        ):
            if key in self.compression_params:
                pt2e_kwargs[key] = self.compression_params[key]

        advanced_parameters: AdvancedQuantizationParameters = self.compression_params.get(
            "advanced_parameters", AdvancedQuantizationParameters()
        )

        sq_params = advanced_parameters.smooth_quant_alphas
        sq_alpha = advanced_parameters.smooth_quant_alpha
        if sq_alpha is not None:
            if sq_alpha < 0:
                sq_params.convolution = -1
                sq_params.matmul = -1
            else:
                sq_params.matmul = sq_alpha
        pt2e_kwargs["smooth_quant_params"] = sq_params
        pt2e_kwargs["bias_correction_params"] = advanced_parameters.bias_correction_params
        pt2e_kwargs["activations_range_estimator_params"] = advanced_parameters.activations_range_estimator_params
        pt2e_kwargs["weights_range_estimator_params"] = advanced_parameters.weights_range_estimator_params

        smooth_quant = False
        if self.compression_params.get("model_type", False):
            smooth_quant = self.compression_params["model_type"] == nncf.ModelType.TRANSFORMER

        with disable_patching(), torch.no_grad():
            self.compressed_model = quantize_pt2e(
                self.model,
                quantizer,
                self.calibration_dataset,
                smooth_quant=smooth_quant,
                fold_quantize=isinstance(quantizer, OpenVINOQuantizer),
                **pt2e_kwargs,
            )

    def _compress(self):
        """
        Quantize self.model
        """
        if self.backend not in FX_BACKENDS:
            super()._compress()

            return
        if self.backend in [BackendType.FX_TORCH, BackendType.CUDA_FX_TORCH]:
            with disable_patching(), torch.no_grad():
                super()._compress()
                return

        quantizer = self._build_quantizer()

        if self.backend in [BackendType.OV_QUANTIZER_NNCF, BackendType.X86_QUANTIZER_NNCF]:
            self._compress_nncf_pt2e(quantizer)
        else:
            self._compress_torch_ao(quantizer)

    def _build_quantizer(self) -> TorchAOQuantizer:
        if self.backend in [BackendType.X86_QUANTIZER_AO, BackendType.X86_QUANTIZER_NNCF]:
            quantizer = X86InductorQuantizer()
            quantizer.set_global(get_default_x86_inductor_quantization_config())
            return quantizer
        quantizer_kwargs = {}
        for key in (
            "mode",
            "preset",
            "target_device",
            "model_type",
            "ignored_scope",
        ):
            if key in self.compression_params:
                quantizer_kwargs[key] = self.compression_params[key]
        advanced_parameters: AdvancedQuantizationParameters = self.compression_params.get(
            "advanced_parameters", AdvancedQuantizationParameters()
        )
        quantizer_kwargs["overflow_fix"] = advanced_parameters.overflow_fix
        quantizer_kwargs["quantize_outputs"] = advanced_parameters.quantize_outputs
        quantizer_kwargs["activations_quantization_params"] = advanced_parameters.activations_quantization_params
        quantizer_kwargs["weights_quantization_params"] = advanced_parameters.weights_quantization_params
        quantizer_kwargs["quantizer_propagation_rule"] = advanced_parameters.quantizer_propagation_rule

        return OpenVINOQuantizer(**quantizer_kwargs)
