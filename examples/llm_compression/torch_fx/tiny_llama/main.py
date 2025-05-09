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

import torch
from datasets import load_dataset
from fx_utils import FXAutoModelForCausalLM
from fx_utils import convert_and_export_with_cache
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

import nncf


def main():
    MODEL_ID = "PY007/TinyLlama-1.1B-Chat-v0.3"

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model_hf = AutoModelForCausalLM.from_pretrained(MODEL_ID)

    def transform_fn(data, tokenizer):
        tokenized_text = tokenizer(data["text"], return_tensors="pt")
        input_ids = tokenized_text["input_ids"]
        attention_mask = tokenized_text["attention_mask"]

        inputs = ()
        position_ids = torch.cumsum(attention_mask, axis=1) - 1
        position_ids[attention_mask == 0] = 1

        inputs = (
            input_ids,
            position_ids.squeeze(0),
        )

        return inputs

    quantization_dataset = nncf.Dataset(dataset, partial(transform_fn, tokenizer=tokenizer))

    model, model_config, gen_config = convert_and_export_with_cache(model_hf)
    model = model.module()
    # Comment this text to turn off model optimization and measure performance of baseline model
    model = nncf.compress_weights(
        model,
        dataset=quantization_dataset,
        mode=nncf.CompressWeightsMode.INT4_SYM,
        ratio=0.8,
        sensitivity_metric=nncf.SensitivityMetric.HESSIAN_INPUT_ACTIVATION,
    )
    compressed_model_hf = FXAutoModelForCausalLM(model, model_config, generation_config=gen_config)

    input_ids = tokenizer("What is PyTorch?", return_tensors="pt")
    output = compressed_model_hf.generate(input_ids["input_ids"])

    start_t = time.time()
    output = compressed_model_hf.generate(input_ids["input_ids"], generation_config=gen_config)
    print("Elapsed time: ", time.time() - start_t)

    output_text = tokenizer.decode(output[0])
    print(output_text)

    return output_text


if __name__ == "__main__":
    main()
