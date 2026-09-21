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

import warnings

import numpy as np
import openvino as ov
from optimum.intel.openvino import OVModelForVisualCausalLM
from torch.jit import TracerWarning
from transformers import AutoTokenizer
from transformers import logging

import nncf

logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=TracerWarning)


MODEL_ID = "optimum-intel-internal-testing/tiny-random-qwen3.5-moe"
COMPRESSED_MODEL_ID = "tiny-random-qwen3.5-moe_compressed"


def generate_answers(
    questions: list[str], model: OVModelForVisualCausalLM, tokenizer: AutoTokenizer, max_new_tokens: int = 10
) -> dict[str, str]:
    """
    Generate answers for a list of questions using the provided model and tokenizer.

    :param questions: List of questions to be answered.
    :param model: The model to use for generating answers.
    :param tokenizer: The tokenizer to use for processing the input and output.
    :param max_new_tokens: Maximum number of new tokens to generate for each answer. Defaults to 10.
    :return: A dictionary mapping each question to its corresponding answer.
    """
    messages = [
        {"role": "system", "content": "You are a chatbot who always responds as short as possible."},
        {"role": "user", "content": "What is the capital of Spain?"},
        {"role": "assistant", "content": "Madrid."},
    ]
    answers_by_questions = {}

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

    return answers_by_questions


def print_answers(header: str, answers_by_questions: dict[str, str]) -> None:
    """
    Print the answers to the console.

    :param header: Header to print before the answers.
    :param answers_by_questions: Dictionary mapping questions to their answers.
    """
    print(header)
    for question, answer in answers_by_questions.items():
        print(f"Q: {question}\nA: {answer}\n")


QUESTIONS = [
    "What is the capital of France?",
    "What is the highest peak in the Alps?",
    "What is the largest city in Canada?",
    "What is the most visited city in Japan?",
]


def load_model_and_tokenizer(model_id: str, export: bool = True) -> tuple[OVModelForVisualCausalLM, AutoTokenizer]:
    """
    Load the model and tokenizer from the specified model ID.

    :param model_id: The identifier of the model to load.
    :param export: Whether to export the model for OpenVINO. Defaults to True.
    :return: A tuple containing the loaded model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    model = OVModelForVisualCausalLM.from_pretrained(
        model_id,
        export=export,
        load_in_8bit=False,
    )
    return model, tokenizer


def get_dummy_dataset(model: ov.Model, seq_len: int = 16) -> nncf.Dataset:
    """
    Build a dummy calibration dataset holding a single all-ones sample.

    :param model: The OV model
    :param seq_len: Sequence length of the synthetic sample.
    :return: An NNCF dataset with a single sample.
    """
    # The language model of a visual causal LM is fed with embeddings, not with token ids.
    hidden_size = model.input("inputs_embeds").get_partial_shape()[2].get_length()

    inputs = {
        "inputs_embeds": np.ones((1, seq_len, hidden_size), dtype=np.float32),
        "attention_mask": np.ones((1, seq_len), dtype=np.int64),
        "position_ids": np.ones((4, 1, seq_len), dtype=np.int64),
        "beam_idx": np.zeros(1, dtype=np.int32),
    }

    return nncf.Dataset([inputs])


def main() -> None:
    model, tokenizer = load_model_and_tokenizer(MODEL_ID)

    answers_by_questions = generate_answers(QUESTIONS, model, tokenizer)
    print_answers("Non-optimized model outputs:\n", answers_by_questions)

    calibration_dataset = get_dummy_dataset(model.language_model.model)

    model.language_model.model = nncf.compress_weights(
        model.language_model.model,
        mode=nncf.CompressWeightsMode.INT4_ASYM,
        scale_estimation=True,
        awq=True,
        dataset=calibration_dataset,
        group_size=16,
        advanced_parameters=nncf.AdvancedCompressionParameters(
            group_size_fallback_mode=nncf.GroupSizeFallbackMode.ADJUST
        ),
    )

    model.save_pretrained(COMPRESSED_MODEL_ID)
    tokenizer.save_pretrained(COMPRESSED_MODEL_ID)

    model, tokenizer = load_model_and_tokenizer(COMPRESSED_MODEL_ID, False)
    answers_by_questions = generate_answers(QUESTIONS, model, tokenizer)
    print_answers("Optimized model outputs:\n", answers_by_questions)


if __name__ == "__main__":
    main()
