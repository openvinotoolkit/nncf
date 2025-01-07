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


import openvino.runtime as ov
import transformers
from optimum.intel import OVQuantizer
from optimum.intel.openvino import OVModelForCausalLM

from tests.post_training.pipelines.base import OV_BACKENDS
from tests.post_training.pipelines.base import BackendType
from tests.post_training.pipelines.base import PTQTestPipeline


class CausalLMHF(PTQTestPipeline):
    """Pipeline for causal language models from Hugging Face repository"""

    def prepare_model(self) -> None:
        is_stateful = self.params.get("is_stateful", False)
        if self.backend in OV_BACKENDS + [BackendType.FP32]:
            self.model_hf = OVModelForCausalLM.from_pretrained(
                self.model_id, export=True, compile=False, stateful=is_stateful
            )
            self.model = self.model_hf.model
            ov.serialize(self.model, self.fp32_model_dir / "model_fp32.xml")

    def prepare_preprocessor(self) -> None:
        self.preprocessor = transformers.AutoTokenizer.from_pretrained(self.model_id)

    def get_transform_calibration_fn(self):
        def transform_func(examples):
            data = self.preprocessor(examples["sentence"])
            return data

        return transform_func

    def prepare_calibration_dataset(self):
        quantizer = OVQuantizer.from_pretrained(self.model_hf)

        num_samples = self.compression_params.get("subset_size", 300)
        calibration_dataset = quantizer.get_calibration_dataset(
            "glue",
            dataset_config_name="sst2",
            preprocess_function=self.get_transform_calibration_fn(),
            num_samples=num_samples,
            dataset_split="validation",
            preprocess_batch=True,
        )

        if self.backend == BackendType.OPTIMUM:
            self.calibration_dataset = calibration_dataset

    def _validate(self):
        pass
