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
import time
from functools import partial

import numpy as np
import openvino as ov
from datasets import load_dataset
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer

import nncf


def main():
    MODEL_ID = "PY007/TinyLlama-1.1B-Chat-v0.3"
    OUTPUT_DIR = "tinyllama_compressed"

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = OVModelForCausalLM.from_pretrained(MODEL_ID, export=True, load_in_8bit=False, compile=False, stateful=False)

    def transform_fn(data, model, tokenizer):
        tokenized_text = tokenizer(data["text"], return_tensors="np")
        input_ids = tokenized_text["input_ids"]
        attention_mask = tokenized_text["attention_mask"]

        inputs = {}
        inputs["input_ids"] = input_ids
        inputs["attention_mask"] = tokenized_text["attention_mask"]
        position_ids = np.cumsum(attention_mask, axis=1) - 1
        position_ids[attention_mask == 0] = 1

        # The magic forms KV cache as model inputs
        batch_size = input_ids.shape[0]
        for input_name in model.key_value_input_names:
            model_inputs = model.model.input(input_name)
            shape = model_inputs.get_partial_shape()
            shape[0] = batch_size
            if shape[2].is_dynamic:
                shape[2] = 0
            else:
                shape[1] = 0
            inputs[input_name] = ov.Tensor(model_inputs.get_element_type(), shape.get_shape())

        inputs["position_ids"] = position_ids
        return inputs

    quantization_dataset = nncf.Dataset(dataset, partial(transform_fn, model=model, tokenizer=tokenizer))

    # Comment this text to turn off model optimization and measure performance of baseline model
    model.model = nncf.compress_weights(
        model.model,
        dataset=quantization_dataset,
        mode=nncf.CompressWeightsMode.INT4_SYM,
        ratio=0.8,
        sensitivity_metric=nncf.SensitivityMetric.HESSIAN_INPUT_ACTIVATION,
    )
    model.save_pretrained(OUTPUT_DIR)

    model = OVModelForCausalLM.from_pretrained(
        OUTPUT_DIR, ov_config={"DYNAMIC_QUANTIZATION_GROUP_SIZE": "0", "KV_CACHE_PRECISION": "f16"}
    )
    input_ids = tokenizer("What is PyTorch?", return_tensors="pt").to(device=model.device)

    start_t = time.time()
    output = model.generate(**input_ids, max_new_tokens=100)
    print("Elapsed time: ", time.time() - start_t)

    output_text = tokenizer.decode(output[0])
    print(output_text)
    return output_text


if __name__ == "__main__":
    main()
