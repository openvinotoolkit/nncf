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

import numpy as np
from datasets import load_dataset
from optimum.intel.openvino import OVModelForCausalLM
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

    batch_size = input_ids.shape[0]
    inputs["beam_idx"] = np.arange(batch_size, dtype=int)

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
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(device=model.device)
        input_len = len(input_ids[0])

        output = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)[0]
        answer = tokenizer.decode(output[input_len:], skip_special_tokens=True)
        answers_by_questions[question] = answer
        messages.append({"role": "assistant", "content": answer})

    model.request = None
    return answers_by_questions


def main():
    MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
    OUTPUT_DIR = "smollm2_360m_compressed"

    dataset = load_dataset("Safetensor/wikitext", "wikitext-2-raw-v1", split="test")
    # Filtering to remove empty samples from the dataset
    dataset = dataset.filter(lambda example: len(example["text"]) > 1)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = OVModelForCausalLM.from_pretrained(
        MODEL_ID,
        export=True,
        load_in_8bit=False,
        compile=False,
        ov_config={"INFERENCE_PRECISION_HINT": "f32"},
    )

    questions = [
        "What is the capital of France?",
        "What is the highest peak in the Alps?",
        "What is the largest city in Canada?",
        "What is the most visited city in Japan?",
    ]

    answers_by_questions = generate_answers(questions, model, tokenizer)
    print(f"Non-optimized model outputs:\n{answers_by_questions}\n")

    quantization_dataset = nncf.Dataset(dataset, partial(transform_fn, tokenizer=tokenizer))

    model.model = nncf.quantize(
        model.model,
        calibration_dataset=quantization_dataset,
        # Only PERFORMANCE preset supports in combination with FP8 quantization mode
        preset=nncf.QuantizationPreset.PERFORMANCE,
        mode=nncf.QuantizationMode.FP8_E4M3,
        model_type=nncf.ModelType.TRANSFORMER,
        # SmoothQuant algorithm is not needed for FP8 quantization
        advanced_parameters=nncf.AdvancedQuantizationParameters(
            smooth_quant_alphas=nncf.AdvancedSmoothQuantParameters(matmul=-1)
        ),
    )
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    model = OVModelForCausalLM.from_pretrained(OUTPUT_DIR, ov_config={"INFERENCE_PRECISION_HINT": "f32"})
    answers_by_questions = generate_answers(questions, model, tokenizer)
    print(f"Optimized model outputs:\n{answers_by_questions}\n")
    return answers_by_questions


if __name__ == "__main__":
    main()
