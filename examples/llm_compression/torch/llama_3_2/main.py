# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time

import torch
from datasets import load_dataset
from optimum.exporters.openvino.convert import export_from_model
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

import nncf

MODEL_ID = "unsloth/Llama-3.2-1B"
OUTPUT_DIR = "compressed"
device = "cuda" if torch.cuda.is_available() else "cpu"


def quantize(model, dataset):
    quantization_dataset = nncf.Dataset(dataset)
    compressed_model = nncf.compress_weights(
        model,
        dataset=quantization_dataset,
        mode=nncf.CompressWeightsMode.INT4_SYM,
        ratio=0.8,
        sensitivity_metric=nncf.SensitivityMetric.HESSIAN_INPUT_ACTIVATION,
    )
    compressed_model.to("cpu")  # issue with cuda export
    export_from_model(compressed_model, OUTPUT_DIR, stateful=False, compression_option="fp32", device="cpu")


def validate(tokenizer, model):
    input_ids = tokenizer("What is PyTorch?", return_tensors="pt").to(device=model.device)

    start_t = time.time()
    output = model.generate(**input_ids, max_new_tokens=100)
    print("Elapsed time: ", time.time() - start_t)

    output_text = tokenizer.decode(output[0])
    print(output_text)
    return output_text


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, load_in_8bit=False).to(device)
    model.eval()

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    # dataset = dataset.filter(lambda example: len(example["text"]) > 128)  # THIS LEADS TO A WORSE RESULT ON VALIDATION

    def transform_fn(data):
        tokenized_text = tokenizer(data["text"], return_tensors="pt").to(device)
        return tokenized_text.data  # NEED TO RETURN ONE OF THE FORMATS of ENGINE (DICT)

    dataset = dataset.map(transform_fn).with_format("torch", device=device)

    quantize(model, dataset)
    model = OVModelForCausalLM.from_pretrained(
        OUTPUT_DIR, ov_config={"DYNAMIC_QUANTIZATION_GROUP_SIZE": "0", "KV_CACHE_PRECISION": "f16"}
    )
    validate(tokenizer, model)


if __name__ == "__main__":
    main()
