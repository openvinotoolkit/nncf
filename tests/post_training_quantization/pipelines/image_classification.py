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

import numpy as np
import torch
import transformers
from optimum.intel import OVQuantizer
from optimum.intel.openvino import OVModelForImageClassification
from optimum.onnxruntime import ORTModelForImageClassification

import nncf
from tests.post_training_quantization.pipelines.base import OV_BACKENDS
from tests.post_training_quantization.pipelines.base import PT_BACKENDS
from tests.post_training_quantization.pipelines.base import BackendType
from tests.post_training_quantization.pipelines.base import BaseHFTestPipeline


class ImageClassificationHF(BaseHFTestPipeline):
    """Pipeline for Image Classification model from Hugging Face repository"""

    def post_init(self):
        if "onnx_model_class" not in self.params:
            self.params["onnx_model_class"] = ORTModelForImageClassification
        if "ov_model_class" not in self.params:
            self.params["ov_model_class"] = OVModelForImageClassification

    def prepare_preprocessor(self) -> None:
        self.preprocessor = transformers.AutoFeatureExtractor.from_pretrained(self.model_id)

    def get_transform_calibration_fn(self):
        if self.backend in PT_BACKENDS:

            def transform_func(x):
                d = self.preprocessor(x["image"])
                for k in d:
                    d[k] = torch.Tensor(d[k])
                return d["pixel_values"]

            return transform_func
        if self.backend == BackendType.ONNX:

            def transform_func(x):
                return self.preprocessor(x["image"])

            return transform_func

        if self.backend in OV_BACKENDS:

            def transform_func(x):
                d = self.preprocessor(x["image"])
                for k in d:
                    d[k] = np.array(d[k])
                return d["pixel_values"]

            return transform_func

    def prepare_calibration_dataset(self):
        quantizer = OVQuantizer.from_pretrained(self.model_hf)
        calibration_dataset = quantizer.get_calibration_dataset(
            "huggingface/cats-image",
            num_samples=self.num_samples,
            preprocess_function=None,
            dataset_split="test",
        )

        self.calibration_dataset = nncf.Dataset(calibration_dataset, self.get_transform_calibration_fn())

    def _validate(self):
        pass
