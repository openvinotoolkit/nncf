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
from pathlib import Path

import numpy as np
import onnx
from datasets import load_dataset
from optimum.intel.openvino import OVModelForCausalLM
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

import nncf
from nncf.onnx.quantization.backend_parameters import BackendParameters

ROOT = Path(__file__).parent.resolve()
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = ROOT / "tinyllama_compressed"


def tiny_llama_transform_func(item, tokenizer, onnx_model: onnx.ModelProto):
    input_name_to_np_dtype = {
        i.name: onnx.helper.tensor_dtype_to_np_dtype(i.type.tensor_type.elem_type) for i in onnx_model.graph.input
    }

    tokens = tokenizer(item["text"])
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
            res[key_name] = np.zeros(shape, dtype=input_name_to_np_dtype[key_name])
            res[val_name] = np.zeros(shape, dtype=input_name_to_np_dtype[val_name])
        return res

    res.update(gen_pkv(4, 64, 22))
    return res


def main():
    # Export the pretrained model in ONNX format. The OUTPUT_DIR directory
    # will contain model.onnx, model.onnx_data, and some metadata files.
    model = ORTModelForCausalLM.from_pretrained(MODEL_ID, export=True)
    model.save_pretrained(OUTPUT_DIR)

    # Load the exported pretrained model as an ONNX model. For models larger than 2GB,
    # set `load_external_data=False` to load only the model's topology without the weights.
    # The weights will be loaded on the fly during compression. To enable this, specify the
    # `BackendParameters.EXTERNAL_DATA_DIR` parameter, which should be the absolute path to
    # the directory containing the model’s external data files.
    onnx_model = onnx.load(OUTPUT_DIR / "model.onnx", load_external_data=False)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Prepare calibration dataset train[:1000]
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    dataset = dataset.filter(lambda example: len(example["text"]) > 128)
    transform_func = partial(tiny_llama_transform_func, tokenizer=tokenizer, onnx_model=onnx_model)
    calibration_dataset = nncf.Dataset(dataset, transform_func)

    optimized_model = nncf.compress_weights(
        onnx_model,
        dataset=calibration_dataset,
        mode=nncf.CompressWeightsMode.INT4_SYM,
        ratio=1.0,
        all_layers=True,
        ignored_scope=nncf.IgnoredScope(types=["Gather"]),
        scale_estimation=True,
        advanced_parameters=nncf.AdvancedCompressionParameters(
            backend_params={BackendParameters.EXTERNAL_DATA_DIR: OUTPUT_DIR}
        ),
    )

    # Replace the original model with the compressed model.
    onnx.save(optimized_model, OUTPUT_DIR / "model.onnx", save_as_external_data=True)

    # Infer Model.
    ov_model = OVModelForCausalLM.from_pretrained(OUTPUT_DIR, from_onnx=True)
    messages = [{"role": "user", "content": "What is PyTorch?"}]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(device=model.device)

    start_t = time.time()
    output = ov_model.generate(input_ids, max_new_tokens=100)
    output_text = tokenizer.decode(output[0])
    print("Elapsed time: ", time.time() - start_t)

    print(output_text)
    return output_text


if __name__ == "__main__":
    main()
