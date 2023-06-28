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


import onnx
import transformers
from optimum.intel import OVQuantizer
from optimum.intel.openvino import OVModelForSequenceClassification
from optimum.onnxruntime import ORTModelForSequenceClassification

from tests.post_training_quantization.pipelines.base import BackendType
from tests.post_training_quantization.pipelines.base import BaseTestPipeline


class MaskedLanguageModelingHF(BaseTestPipeline):
    """Pipeline for Image Classification model from Hugging Face repository"""

    def prepare_model(self) -> None:
        if self.backend in [BackendType.TORCH, BackendType.LEGACY_TORCH]:
            self.model_hf = transformers.AutoModelForSequenceClassification.from_pretrained(self.model_id)
            self.model = self.model_hf

        if self.backend in [BackendType.OV, BackendType.POT, BackendType.OPTIMUM]:
            self.model_hf = OVModelForSequenceClassification.from_pretrained(self.model_id, export=True, compile=False)
            self.model = self.model_hf.model

        if self.backend in [BackendType.ONNX]:
            self.model_hf = ORTModelForSequenceClassification.from_pretrained(self.model_id, export=True)
            self.model = onnx.load(self.model_hf.model_path)

    def prepare_preprocessor(self) -> None:
        self.preprocessor = transformers.AutoTokenizer.from_pretrained(self.model_id)

    def get_transform_calibration_fn(self):
        def transform_func(examples):
            return self.preprocessor(examples["sentence"], padding=True, truncation=True, max_length=128)

        return transform_func

    def prepare_calibration_dataset(self):
        quantizer = OVQuantizer.from_pretrained(self.model_hf)

        num_samples = self.ptq_params.get("subset_size", 300)
        calibration_dataset = quantizer.get_calibration_dataset(
            "glue",
            dataset_config_name="sst2",
            preprocess_function=self.get_transform_calibration_fn(),
            num_samples=num_samples,
            dataset_split="train",
            preprocess_batch=True,
        )

        if self.backend == BackendType.OPTIMUM:
            self.calibration_dataset = calibration_dataset

    def _validate(self):
        pass
