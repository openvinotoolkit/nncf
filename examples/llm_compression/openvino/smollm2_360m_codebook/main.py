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
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer
from transformers import logging

import nncf

logging.set_verbosity_error()


def generate_answers(questions, model, tokenizer, max_new_tokens=50):
    """Generate answers for a list of questions using the provided model and tokenizer.

    Args:
        questions : List of questions to be answered.
        model : The model to use for generating answers.
        tokenizer : The tokenizer to use for processing the input and output.
        max_new_tokens (int, optional): Maximum number of new tokens to generate for each answer. Defaults to 50.

    Returns:
        dict: A dictionary mapping each question to its corresponding answer.
    """
    messages = [
        {"role": "system", "content": "You are a chatbot who always responds as short as possible."},
        {"role": "user", "content": "What is the capital of Spain?"},
        {"role": "assistant", "content": "Madrid."},
    ]
    answers_by_questions = {}

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

    return answers_by_questions


def print_answers(header, answers_by_questions):
    print(header)
    for question, answer in answers_by_questions.items():
        print(f"Q: {question}\nA: {answer}\n")


QUESTIONS = [
    "What is the capital of France?",
    "What is the highest peak in the Alps?",
    "What is the largest city in Canada?",
    "What is the most visited city in Japan?",
]


def load_model_and_tokenizer(model_id, export=True):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    model = OVModelForCausalLM.from_pretrained(
        model_id,
        export=export,
        load_in_8bit=False,
    )
    return model, tokenizer


def default_codebook_example(model_id, output_dir):
    model, tokenizer = load_model_and_tokenizer(model_id)
    answers_by_questions = generate_answers(QUESTIONS, model, tokenizer)
    print_answers("Non-optimized model outputs:\n", answers_by_questions)

    model.model = nncf.compress_weights(model.model, mode=nncf.CompressWeightsMode.CB4_F8E4M3, ratio=1.0, group_size=64)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    model, tokenizer = load_model_and_tokenizer(output_dir, False)
    answers_by_questions = generate_answers(QUESTIONS, model, tokenizer)
    print_answers("Optimized model outputs:\n", answers_by_questions)

    return list(answers_by_questions.values())


def custom_codebook_example(model_id, output_dir):
    model, tokenizer = load_model_and_tokenizer(model_id)

    answers_by_questions = generate_answers(QUESTIONS, model, tokenizer)
    print_answers("Non-optimized model outputs:\n", answers_by_questions)

    codebook_params = nncf.CodebookParameters(
        np.array([-64, -32, -16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16, 32, 64], dtype=np.int8)
    )

    model.model = nncf.compress_weights(
        model.model,
        mode=nncf.CompressWeightsMode.CODEBOOK,
        ratio=1.0,
        group_size=-1,
        advanced_parameters=nncf.AdvancedCompressionParameters(codebook_params=codebook_params),
    )
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    model, tokenizer = load_model_and_tokenizer(output_dir, False)
    answers_by_questions = generate_answers(QUESTIONS, model, tokenizer)
    print_answers("Optimized model outputs:\n", answers_by_questions)

    return list(answers_by_questions.values())


def main():
    model_id = "HuggingFaceTB/SmolLM2-360M-Instruct"
    output_dir = "smollm2_360m_compressed_codebook"

    res = default_codebook_example(model_id, output_dir)
    res += custom_codebook_example(model_id, output_dir + "_custom")
    return res


if __name__ == "__main__":
    main()
