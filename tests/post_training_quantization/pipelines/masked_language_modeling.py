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
from optimum.intel.openvino import OVModelForSequenceClassification
from optimum.onnxruntime import ORTModelForSequenceClassification

import nncf
from tests.post_training_quantization.pipelines.base import OV_BACKENDS
from tests.post_training_quantization.pipelines.base import PT_BACKENDS
from tests.post_training_quantization.pipelines.base import BackendType
from tests.post_training_quantization.pipelines.base import BaseHFTestPipeline


class MaskedLanguageModelingHF(BaseHFTestPipeline):
    """Pipeline for Image Classification model from Hugging Face repository"""

    def post_init(self):
        if "pt_model_class" not in self.params:
            self.params["pt_model_class"] = transformers.AutoModelForSequenceClassification
        if "ov_model_class" not in self.params:
            self.params["ov_model_class"] = OVModelForSequenceClassification
        if "onnx_model_class" not in self.params:
            self.params["onnx_model_class"] = ORTModelForSequenceClassification

    def prepare_preprocessor(self) -> None:
        self.preprocessor = transformers.AutoTokenizer.from_pretrained(self.model_id)

    def get_transform_calibration_fn(self):
        def transform_func(examples):
            return self.preprocessor(examples["sentence"], padding=True, truncation=True, max_length=128)

        return transform_func

    def prepare_calibration_dataset(self):
        quantizer = OVQuantizer.from_pretrained(self.model_hf)

        calibration_dataset = quantizer.get_calibration_dataset(
            "glue",
            dataset_config_name="sst2",
            preprocess_function=self.get_transform_calibration_fn(),
            num_samples=self.num_samples,
            dataset_split="train",
            preprocess_batch=True,
        )

        if self.backend == BackendType.OPTIMUM:
            self.calibration_dataset = calibration_dataset
        else:
            # TODO: not works
            def transform_fn(x):
                return x["input_ids"], x["token_type_ids"], x["attention_mask"]

            self.calibration_dataset = nncf.Dataset(calibration_dataset, transform_fn)

    def _validate(self):
        pass
