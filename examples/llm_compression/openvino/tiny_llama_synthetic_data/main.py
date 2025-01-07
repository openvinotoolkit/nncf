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

from functools import partial

import numpy as np
import openvino as ov
import torch
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer

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


def main():
    MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    hf_model = OVModelForCausalLM.from_pretrained(
        MODEL_ID, export=True, load_in_8bit=False, compile=False, stateful=False
    )

    dataset_size = 100

    # Synthetic-based compression
    saved_seed = torch.seed()
    torch.manual_seed(SEED)
    synthetic_dataset = nncf.data.generate_text_data(hf_model, tokenizer, dataset_size=dataset_size)
    quantization_dataset = nncf.Dataset(
        synthetic_dataset, partial(transform_func, tokenizer=tokenizer, ov_model=hf_model.model)
    )
    hf_model.request = None
    torch.manual_seed(saved_seed)

    optimized_model = nncf.compress_weights(
        hf_model.model.clone(),
        dataset=quantization_dataset,
        mode=nncf.CompressWeightsMode.INT4_SYM,
        ratio=1.0,
        scale_estimation=True,
    )

    # Verify the model output in comparison to floating-point one
    input_ids = tokenizer("What is Python? ", return_tensors="pt").to(device=hf_model.device)
    max_new_tokens = 100

    hf_model.model = optimized_model
    hf_model.request = None
    opt_output = hf_model.generate(**input_ids, max_new_tokens=max_new_tokens)
    opt_output_text = tokenizer.decode(opt_output[0])

    print(f"Optimized model output: {opt_output_text}\n")
    return opt_output_text


if __name__ == "__main__":
    main()
