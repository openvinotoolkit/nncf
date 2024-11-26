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

from pathlib import Path
from typing import Dict, Tuple

import openvino as ov
import torch
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset
from ultralytics.engine.validator import BaseValidator as Validator
from ultralytics.utils.torch_utils import de_parallel

import nncf
from nncf.torch import disable_patching
from tests.post_training.pipelines.base import OV_BACKENDS
from tests.post_training.pipelines.base import BackendType
from tests.post_training.pipelines.base import PTQTestPipeline


class UltralyticsDetection(PTQTestPipeline):
    """Pipeline for Yolo detection models from the Ultralytics repository"""

    def prepare_model(self) -> None:
        if self.batch_size != 1:
            msg = "Batch size > 1 is not supported"
            raise RuntimeError(msg)

        model_path = f"{self.fp32_model_dir}/{self.model_id}"
        yolo = YOLO(f"{model_path}.pt")
        self.validator, self.data_loader = self._prepare_validation(yolo, "coco128.yaml")
        self.dummy_tensor = torch.ones((1, 3, 640, 640))

        if self.backend in OV_BACKENDS + [BackendType.FP32]:
            onnx_model_path = Path(f"{model_path}.onnx")
            ir_model_path = self.fp32_model_dir / "model_fp32.xml"
            yolo.export(format="onnx", dynamic=True, half=False)
            ov.save_model(ov.convert_model(onnx_model_path), ir_model_path)
            self.model = ov.Core().read_model(ir_model_path)

        if self.backend == BackendType.FX_TORCH:
            pt_model = yolo.model
            # Run mode one time to initialize all
            # internal variables
            pt_model(self.dummy_tensor)

            with torch.no_grad():
                with disable_patching():
                    self.model = torch.export.export(pt_model, args=(self.dummy_tensor,), strict=False).module()

    def prepare_preprocessor(self) -> None:
        pass

    @staticmethod
    def _validate_fx(
        model: ov.Model, data_loader: torch.utils.data.DataLoader, validator: Validator, num_samples: int = None
    ) -> Tuple[Dict, int, int]:
        compiled_model = torch.compile(model, backend="openvino")
        for batch_i, batch in enumerate(data_loader):
            if num_samples is not None and batch_i == num_samples:
                break
            batch = validator.preprocess(batch)
            preds = compiled_model(batch["img"])
            preds = validator.postprocess(preds)
            validator.update_metrics(preds, batch)
        stats = validator.get_stats()
        return stats, validator.seen, validator.nt_per_class.sum()

    @staticmethod
    def _validate_ov(
        model: ov.Model, data_loader: torch.utils.data.DataLoader, validator: Validator, num_samples: int = None
    ) -> Tuple[Dict, int, int]:
        model.reshape({0: [1, 3, -1, -1]})
        compiled_model = ov.compile_model(model)
        output_layer = compiled_model.output(0)
        for batch_i, batch in enumerate(data_loader):
            if num_samples is not None and batch_i == num_samples:
                break
            batch = validator.preprocess(batch)
            preds = torch.from_numpy(compiled_model(batch["img"])[output_layer])
            preds = validator.postprocess(preds)
            validator.update_metrics(preds, batch)
        stats = validator.get_stats()
        return stats, validator.seen, validator.nt_per_class.sum()

    def get_transform_calibration_fn(self):
        def transform_func(batch):
            return self.validator.preprocess(batch)["img"]

        return transform_func

    def prepare_calibration_dataset(self):
        self.calibration_dataset = nncf.Dataset(self.data_loader, self.get_transform_calibration_fn())

    @staticmethod
    def _prepare_validation(model: YOLO, data: str) -> Tuple[Validator, torch.utils.data.DataLoader]:
        custom = {"rect": False, "batch": 1}  # method defaults
        args = {**model.overrides, **custom, "mode": "val"}  # highest priority args on the right

        validator = model._smart_load("validator")(args=args, _callbacks=model.callbacks)
        stride = 32  # default stride
        validator.stride = stride  # used in get_dataloader() for padding
        validator.data = check_det_dataset(data)
        validator.init_metrics(de_parallel(model))

        data_loader = validator.get_dataloader(validator.data.get(validator.args.split), validator.args.batch)

        return validator, data_loader

    def _validate(self):
        if self.backend == BackendType.FP32:
            stats, _, _ = self._validate_ov(self.model, self.data_loader, self.validator)
        elif self.backend in OV_BACKENDS:
            stats, _, _ = self._validate_ov(self.compressed_model, self.data_loader, self.validator)
        elif self.backend == BackendType.FX_TORCH:
            stats, _, _ = self._validate_fx(self.compressed_model, self.data_loader, self.validator)
        else:
            msg = f"Backend {self.backend} is not supported in UltralyticsDetection"
            raise RuntimeError(msg)

        self.run_info.metric_name = "mAP50(B)"
        self.run_info.metric_value = stats["metrics/mAP50(B)"]
