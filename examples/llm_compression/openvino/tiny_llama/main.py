# Copyright (c) 2026 Intel Corporation
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
from functools import partial

import numpy as np
from datasets import load_dataset
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer

import nncf


def main():
    MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    OUTPUT_DIR = "tinyllama_compressed"

    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = OVModelForCausalLM.from_pretrained(MODEL_ID, export=True, load_in_8bit=False, compile=False)

    def transform_fn(data, tokenizer):
        tokenized_text = tokenizer(data["text"], return_tensors="np")
        input_ids = tokenized_text["input_ids"]
        attention_mask = tokenized_text["attention_mask"]

        inputs = {}
        inputs["input_ids"] = input_ids
        inputs["attention_mask"] = tokenized_text["attention_mask"]
        position_ids = np.cumsum(attention_mask, axis=1) - 1
        position_ids[attention_mask == 0] = 1
        inputs["position_ids"] = position_ids

        batch_size = input_ids.shape[0]
        inputs["beam_idx"] = np.arange(batch_size, dtype=int)

        return inputs

    quantization_dataset = nncf.Dataset(dataset, partial(transform_fn, tokenizer=tokenizer))

    # Comment this text to turn off model optimization and measure performance of baseline model
    model.model = nncf.compress_weights(
        model.model,
        dataset=quantization_dataset,
        mode=nncf.CompressWeightsMode.INT4_SYM,
        ratio=0.8,
        sensitivity_metric=nncf.SensitivityMetric.HESSIAN_INPUT_ACTIVATION,
    )
    model.save_pretrained(OUTPUT_DIR)

    model = OVModelForCausalLM.from_pretrained(OUTPUT_DIR)

    messages = [{"role": "user", "content": "What is PyTorch?"}]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(device=model.device)

    start_t = time.time()
    output = model.generate(input_ids, max_new_tokens=100)
    print("Elapsed time: ", time.time() - start_t)

    output_text = tokenizer.decode(output[0])
    print(output_text)
    return output_text


if __name__ == "__main__":
    main()
