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

from functools import partial

import datasets
import numpy as np
import openvino as ov
import torch
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer
from whowhatbench import Evaluator

import nncf

SEED = 0


def transform_func(text, tokenizer, ov_model):
    input_dtypes = {inp.get_any_name(): inp.get_element_type() for inp in ov_model.inputs}
    tokens = tokenizer(text)
    input_ids = np.expand_dims(np.array(tokens["input_ids"]), 0)
    attention_mask = np.expand_dims(np.array(tokens["attention_mask"]), 0)
    position_ids = np.cumsum(attention_mask, axis=1) - 1
    position_ids[attention_mask == 0] = 1
    res = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids.reshape(*attention_mask.shape),
    }

    def gen_pkv(num_heads, head_dim, num_layers):
        res = {}
        shape = (1, num_heads, 0, head_dim)
        for i in range(num_layers):
            key_name = f"past_key_values.{i}.key"
            val_name = f"past_key_values.{i}.value"
            res[key_name] = ov.Tensor(shape=shape, type=input_dtypes[key_name])
            res[val_name] = ov.Tensor(shape=shape, type=input_dtypes[val_name])
        return res

    res.update(gen_pkv(4, 64, 22))
    return res


def compress_model(model, tokenizer, dataset):
    quantization_dataset = nncf.Dataset(dataset, partial(transform_func, tokenizer=tokenizer, ov_model=model.model))

    optimized_model = nncf.compress_weights(
        model.model.clone(),
        dataset=quantization_dataset,
        mode=nncf.CompressWeightsMode.INT4_SYM,
        ratio=1.0,
        scale_estimation=True,
    )
    return optimized_model


def validate_model(evaluator, hf_model, optimized_model, original_ov_model):
    hf_model.model = optimized_model
    hf_model.request = None
    _, all_metrics = evaluator.score(hf_model)
    hf_model.model = original_ov_model
    hf_model.request = None
    return all_metrics["similarity"][0]


def main():
    MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    hf_model = OVModelForCausalLM.from_pretrained(
        MODEL_ID, export=True, load_in_8bit=False, compile=False, stateful=False
    )

    original_ov_model = hf_model.model.clone()
    evaluator = Evaluator(hf_model, tokenizer=tokenizer, metrics=("similarity",))

    # Wikitext-based compression
    wikitext_dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    wikitext_dataset = [d["text"] for d in wikitext_dataset]
    wikitext_optimized_model = compress_model(hf_model, tokenizer, wikitext_dataset)

    # Synthetic-based compression
    saved_seed = torch.seed()
    torch.manual_seed(SEED)
    synthetic_dataset = nncf.data.generate_text_data(hf_model, tokenizer)
    torch.manual_seed(saved_seed)
    synthetic_optimized_model = compress_model(hf_model, tokenizer, synthetic_dataset)

    # Similarity comparison between Wikitext-based & Synthetic-based compressed models
    wikitext_based_similarity = validate_model(evaluator, hf_model, wikitext_optimized_model, original_ov_model)
    print(f"Wikitext-quantized model similarity: {wikitext_based_similarity}")

    synthetic_based_similarity = validate_model(evaluator, hf_model, synthetic_optimized_model, original_ov_model)
    print(f"Synthetic-quantized model similarity: {synthetic_based_similarity}")
    return wikitext_based_similarity, synthetic_based_similarity


if __name__ == "__main__":
    main()
