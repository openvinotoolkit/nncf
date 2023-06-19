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


class AutomaticSpeechRecognitionHF(BaseHFTestPipeline):
    def prepare_preprocessor(self) -> None:
        self.preprocessor = transformers.WhisperProcessor.from_pretrained(self.model_id)

    def get_transform_calibration_fn(self):
        if self.backend in PT_BACKENDS:

            def transform_func(sample):
                sample = sample["audio"]
                input_features = self.preprocessor(
                    sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt"
                ).input_features
                return input_features

            return transform_func

    def prepare_calibration_dataset(self):
        quantizer = OVQuantizer.from_pretrained(self.model_hf)
        calibration_dataset = quantizer.get_calibration_dataset(
            "hf-internal-testing/librispeech_asr_dummy",
            dataset_config_name="clean",
            num_samples=self.num_samples,
            preprocess_function=None,
            dataset_split="validation",
        )

        self.calibration_dataset = nncf.Dataset(calibration_dataset, self.get_transform_calibration_fn())

    def _validate(self):
        pass
