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


import numpy as np
import openvino.runtime as ov
import torch
import transformers
from optimum.intel import OVQuantizer
from optimum.intel.openvino import OVModelForCausalLM
from optimum.intel.openvino.configuration import OVQuantizationConfigBase

import nncf
from tests.post_training.pipelines.base import OV_BACKENDS
from tests.post_training.pipelines.base import PT_BACKENDS
from tests.post_training.pipelines.base import BackendType
from tests.post_training.pipelines.base import PTQTestPipeline


class GPT(PTQTestPipeline):
    """Pipeline for causal language models from Hugging Face repository"""

    def prepare_model(self) -> None:
        if self.backend in PT_BACKENDS:
            self.model_hf = transformers.AutoModelForCausalLM.from_pretrained(self.model_id)
            self.model = self.model_hf
            self.model.config.torchscript = True  # Set to export by convert_model via torch.jit.trace
            self.dummy_tensor = self.model_hf.dummy_inputs["input_ids"]

        elif self.backend in OV_BACKENDS + [BackendType.FP32]:
            self.model_hf = OVModelForCausalLM.from_pretrained(self.model_id, export=True)
            self.model = self.model_hf.model
            ov.serialize(self.model, self.fp32_model_dir / "model_fp32.xml")

        # Set device after dump fp32 model
        if self.backend == BackendType.CUDA_TORCH:
            self.model.cuda()
            self.dummy_tensor = self.dummy_tensor.cuda()

    def prepare_preprocessor(self) -> None:
        self.preprocessor = transformers.AutoTokenizer.from_pretrained(self.model_id)
        # Fails with default pad_token
        self.preprocessor.pad_token = self.preprocessor.eos_token

    def get_transform_calibration_fn(self):
        if self.backend in PT_BACKENDS:
            device = torch.device("cuda" if self.backend == BackendType.CUDA_TORCH else "cpu")

            def transform_func(data):
                inputs = {
                    "input_ids": torch.tensor([data["input_ids"]], dtype=torch.int64, device=device),
                    "attention_mask": torch.tensor([data["attention_mask"]], dtype=torch.int64, device=device),
                }
                return inputs

        else:

            def transform_func(data):
                ids = np.expand_dims(data["input_ids"], axis=0)
                inputs = {
                    "input_ids": ids,
                    "attention_mask": np.expand_dims(data["attention_mask"], axis=0),
                    "position_ids": np.ones(ids.shape, dtype=np.int64),
                    "beam_idx": np.zeros((1,), dtype=np.int64),
                }
                return inputs

        return transform_func

    def prepare_calibration_dataset(self):
        quantizer = OVQuantizer.from_pretrained(self.model_hf)

        num_samples = self.compression_params.get("subset_size", 300)
        calibration_dataset = quantizer._prepare_causal_lm_dataset(
            OVQuantizationConfigBase(
                dataset="wikitext2",
                num_samples=num_samples,
                tokenizer=self.model_id,
            )
        )

        if self.backend == BackendType.OPTIMUM:
            self.calibration_dataset = calibration_dataset
        else:
            self.calibration_dataset = nncf.Dataset(calibration_dataset, self.get_transform_calibration_fn())
