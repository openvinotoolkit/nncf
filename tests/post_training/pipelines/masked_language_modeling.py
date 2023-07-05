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
import onnx
import torch
import transformers
from optimum.intel import OVQuantizer
from optimum.intel.openvino import OVModelForSequenceClassification
from optimum.onnxruntime import ORTModelForSequenceClassification

import nncf
from tests.post_training.pipelines.base import OV_BACKENDS
from tests.post_training.pipelines.base import PT_BACKENDS
from tests.post_training.pipelines.base import BackendType
from tests.post_training.pipelines.base import BaseTestPipeline


class MaskedLanguageModelingHF(BaseTestPipeline):
    """Pipeline for masked language models from Hugging Face repository"""

    def prepare_model(self) -> None:
        if self.backend in PT_BACKENDS:
            self.model_hf = transformers.AutoModelForSequenceClassification.from_pretrained(self.model_id)
            self.model = self.model_hf
            self.dummy_tensor = self.model_hf.dummy_inputs["input_ids"]
        if self.backend in OV_BACKENDS:
            self.model_hf = OVModelForSequenceClassification.from_pretrained(self.model_id, export=True, compile=False)
            self.model = self.model_hf.model

        if self.backend == BackendType.ONNX:
            self.model_hf = ORTModelForSequenceClassification.from_pretrained(self.model_id, export=True)
            self.model = onnx.load(self.model_hf.model_path)

    def prepare_preprocessor(self) -> None:
        self.preprocessor = transformers.AutoTokenizer.from_pretrained(self.model_id)

    def get_transform_calibration_fn(self):
        if self.backend in PT_BACKENDS:

            def transform_func(data):
                return torch.Tensor([data["input_ids"]]).type(dtype=torch.LongTensor)

        else:

            def transform_func(data):
                return {
                    "input_ids": np.expand_dims(data["input_ids"], axis=0),
                    "token_type_ids": np.expand_dims(data["token_type_ids"], axis=0),
                    "attention_mask": np.expand_dims(data["attention_mask"], axis=0),
                }

        return transform_func

    def prepare_calibration_dataset(self):
        quantizer = OVQuantizer.from_pretrained(self.model_hf)

        num_samples = self.ptq_params.get("subset_size", 300)

        def preprocess_function(examples):
            return self.preprocessor(examples["sentence"], padding=True, truncation=True, max_length=128)

        calibration_dataset = quantizer.get_calibration_dataset(
            "glue",
            dataset_config_name="sst2",
            preprocess_function=preprocess_function,
            num_samples=num_samples,
            dataset_split="validation",
            preprocess_batch=True,
        )

        if self.backend == BackendType.OPTIMUM:
            self.calibration_dataset = calibration_dataset
        else:
            self.calibration_dataset = nncf.Dataset(calibration_dataset, self.get_transform_calibration_fn())

    def _validate(self):
        pass
