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

from functools import partial
from pathlib import Path

import numpy as np
import onnx
from datasets import load_dataset
from optimum.intel.openvino import OVModelForCausalLM
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

import nncf


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

    return inputs


def generate_answers(questions, model, tokenizer, max_new_tokens=50):
    messages = [
        {"role": "system", "content": "You are a chatbot who always responds as short as possible."},
        {"role": "user", "content": "What is the capital of Spain?"},
        {"role": "assistant", "content": "Madrid."},
    ]
    answers_by_questions = {}
    model.request = None

    for question in questions:
        messages.append({"role": "user", "content": question})
        batch_feature = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        input_ids = batch_feature["input_ids"]
        input_len = len(input_ids[0])

        output = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)[0]
        answer = tokenizer.decode(output[input_len:], skip_special_tokens=True)
        answers_by_questions[question] = answer
        messages.append({"role": "assistant", "content": answer})

    model.request = None
    return answers_by_questions


def main():
    ROOT = Path(__file__).parent.resolve()
    MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
    OUTPUT_DIR = ROOT / "smollm2_360m_compressed"

    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    # Filtering to remove empty samples from the dataset
    dataset = dataset.filter(lambda example: len(example["text"]) > 1)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = ORTModelForCausalLM.from_pretrained(MODEL_ID, export=True, use_cache=False)
    model.save_pretrained(OUTPUT_DIR)

    questions = [
        "What is the capital of France?",
        "What is the highest peak in the Alps?",
        "What is the largest city in Canada?",
        "What is the most visited city in Japan?",
    ]

    answers_by_questions = generate_answers(questions, model, tokenizer)
    print(f"Non-optimized model outputs:\n{answers_by_questions}\n")

    quantization_dataset = nncf.Dataset(dataset, partial(transform_fn, tokenizer=tokenizer))

    onnx_model = onnx.load(OUTPUT_DIR / "model.onnx", load_external_data=False)
    target_version = 21
    if onnx_model.opset_import[0].version != target_version:
        onnx_model = onnx.version_converter.convert_version(onnx_model, target_version)

    compressed_onnx_model = nncf.quantize(
        onnx_model,
        calibration_dataset=quantization_dataset,
        # Only PERFORMANCE preset supports in combination with FP8 quantization mode
        preset=nncf.QuantizationPreset.PERFORMANCE,
        mode=nncf.QuantizationMode.FP8_E4M3,
        model_type=nncf.ModelType.TRANSFORMER,
        # SmoothQuant algorithm is not needed for FP8 quantization
        advanced_parameters=nncf.AdvancedQuantizationParameters(
            smooth_quant_alphas=nncf.AdvancedSmoothQuantParameters(matmul=-1)
        ),
        ignored_scope=nncf.IgnoredScope(
            names=[
                "/model/rotary_emb/MatMul",
                "/model/layers.0/self_attn/MatMul_1",
                "/model/layers.1/self_attn/MatMul_1",
                "/model/layers.2/self_attn/MatMul_1",
                "/model/layers.3/self_attn/MatMul_1",
                "/model/layers.4/self_attn/MatMul_1",
                "/model/layers.5/self_attn/MatMul_1",
                "/model/layers.6/self_attn/MatMul_1",
                "/model/layers.7/self_attn/MatMul_1",
                "/model/layers.8/self_attn/MatMul_1",
                "/model/layers.9/self_attn/MatMul_1",
                "/model/layers.10/self_attn/MatMul_1",
                "/model/layers.11/self_attn/MatMul_1",
                "/model/layers.12/self_attn/MatMul_1",
                "/model/layers.13/self_attn/MatMul_1",
                "/model/layers.14/self_attn/MatMul_1",
                "/model/layers.15/self_attn/MatMul_1",
                "/model/layers.16/self_attn/MatMul_1",
                "/model/layers.17/self_attn/MatMul_1",
                "/model/layers.18/self_attn/MatMul_1",
                "/model/layers.19/self_attn/MatMul_1",
                "/model/layers.20/self_attn/MatMul_1",
                "/model/layers.21/self_attn/MatMul_1",
                "/model/layers.22/self_attn/MatMul_1",
                "/model/layers.23/self_attn/MatMul_1",
                "/model/layers.24/self_attn/MatMul_1",
                "/model/layers.25/self_attn/MatMul_1",
                "/model/layers.26/self_attn/MatMul_1",
                "/model/layers.27/self_attn/MatMul_1",
                "/model/layers.28/self_attn/MatMul_1",
                "/model/layers.29/self_attn/MatMul_1",
                "/model/layers.30/self_attn/MatMul_1",
                "/model/layers.31/self_attn/MatMul_1",
            ],
        ),
    )

    # Replace the original model with the compressed model.
    onnx.save(compressed_onnx_model, OUTPUT_DIR / "model.onnx", save_as_external_data=True)
    tokenizer.save_pretrained(OUTPUT_DIR)

    model = OVModelForCausalLM.from_pretrained(
        OUTPUT_DIR, ov_config={"INFERENCE_PRECISION_HINT": "f32"}, from_onnx=True, use_cache=False
    )
    answers_by_questions = generate_answers(questions, model, tokenizer)
    print(f"Optimized model outputs:\n{answers_by_questions}\n")
    return answers_by_questions


if __name__ == "__main__":
    main()
