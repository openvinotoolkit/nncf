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
from functools import partial

import datasets
import numpy as np
from optimum.intel.openvino import OVModelForCausalLM
from scipy.stats import norm
from torch.jit import TracerWarning
from transformers import AutoTokenizer
from transformers import logging

import nncf
from nncf.quantization.advanced_parameters import AdvancedAdaptiveCodebookParameters

logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=TracerWarning)


MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
COMPRESSED_MODEL_ID = "smollm2_360m_compressed_codebook"


def get_dataset(model, tokenizer):
    def transform_func(item, tokenizer, input_shapes, max_tokens=128):
        text = item["text"]
        tokens = tokenizer(text)

        res = {
            "input_ids": np.expand_dims(np.array(tokens["input_ids"][:max_tokens]), 0),
            "attention_mask": np.expand_dims(np.array(tokens["attention_mask"][:max_tokens]), 0),
        }

        if "position_ids" in input_shapes:
            position_ids = np.cumsum(res["attention_mask"], axis=1) - 1
            position_ids[res["attention_mask"] == 0] = 1
            res["position_ids"] = position_ids
        batch_size = res["input_ids"].shape[0]

        if "beam_idx" in input_shapes:
            res["beam_idx"] = np.arange(batch_size, dtype=int)

        return res

    def get_input_shapes(model, batch_size=1):
        inputs = {}

        for val in model.model.inputs:
            name = val.any_name
            shape = list(val.partial_shape.get_min_shape())
            shape[0] = batch_size
            inputs[name] = shape

        return inputs

    input_shapes = get_input_shapes(model, batch_size=1)

    dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    dataset = dataset.filter(lambda example: len(example["text"]) > 128)

    def preprocess_fn(example):
        return {"text": tokenizer.apply_chat_template(example["messages"], add_generation_prompt=False, tokenize=False)}

    num_samples = 2048
    ds = datasets.load_dataset("neuralmagic/LLM_compression_calibration", split="train")
    ds = ds.shuffle(seed=42).select(range(num_samples))
    ds = ds.map(preprocess_fn)
    dataset = ds

    quantization_dataset = nncf.Dataset(
        dataset, partial(transform_func, tokenizer=tokenizer, input_shapes=input_shapes)
    )
    return quantization_dataset


def create_normal_distributed_values(n_levels=8) -> np.ndarray:
    probs = (np.arange(n_levels) + 0.5) / n_levels

    # Inverse CDF (quantiles) of standard normal distribution
    values = norm.ppf(probs)

    # Normalize to [-1, 1]
    values = values / np.max(np.abs(values))

    return values.astype(np.float32)


def generate_answers(
    questions: list[str], model: OVModelForCausalLM, tokenizer: AutoTokenizer, max_new_tokens: int = 50
) -> dict[str, str]:
    """
    Generate answers for a list of questions using the provided model and tokenizer.

    :param questions: List of questions to be answered.
    :param model: The model to use for generating answers.
    :param tokenizer: The tokenizer to use for processing the input and output.
    :param max_new_tokens: Maximum number of new tokens to generate for each answer. Defaults to 50.
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
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(device=model.device)
        input_len = len(input_ids[0])

        output = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)[0]
        answer = tokenizer.decode(output[input_len:], skip_special_tokens=True)
        answers_by_questions[question] = answer
        messages.append({"role": "assistant", "content": answer})

    return answers_by_questions


def print_answers(header: str, answers_by_questions: list[str]) -> None:
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


def load_model_and_tokenizer(model_id: str, export=True) -> tuple[OVModelForCausalLM, AutoTokenizer]:
    """
    Load the model and tokenizer from the specified model ID.

    :param model_id: The identifier of the model to load.
    :param export: Whether to export the model for OpenVINO. Defaults to True.
    :return: A tuple containing the loaded model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    model = OVModelForCausalLM.from_pretrained(
        model_id,
        export=export,
        load_in_8bit=False,
    )
    return model, tokenizer


def codebook_example(
    model_id: str, compressed_model_id: str, adaptive_codebook: bool = False, num_elements: int = 10
) -> list[str]:
    """
    Example of using the custom codebook compression.

    :param model_id: The identifier of the model to load.
    :param compressed_model_id: The identifier for the compressed model to save.
    :param adaptive_codebook: Whether to use adaptive codebook compression. Defaults to False.
    :param num_parameters: Number of parameters in the codebook. Defaults to 8.
    :return: A list of answers generated by the model after compression.
    """
    model, tokenizer = load_model_and_tokenizer(model_id)

    answers_by_questions = generate_answers(QUESTIONS, model, tokenizer)
    print_answers("Non-optimized model outputs:\n", answers_by_questions)

    codebook = create_normal_distributed_values(num_elements)

    adaptive_codebook_params = AdvancedAdaptiveCodebookParameters(
        num_elements=num_elements, value_type=nncf.tensor.TensorDataType.float16, per_block=False
    )
    quantization_dataset = get_dataset(model, tokenizer)

    model.model = nncf.compress_weights(
        model.model,
        mode=nncf.CompressWeightsMode.ADAPTIVE_CODEBOOK if adaptive_codebook else nncf.CompressWeightsMode.CODEBOOK,
        ratio=1.0,
        group_size=-1,
        scale_estimation=True,
        dataset=quantization_dataset,
        advanced_parameters=nncf.AdvancedCompressionParameters(
            codebook=codebook, adaptive_codebook_params=adaptive_codebook_params if adaptive_codebook else None
        ),
    )
    model.save_pretrained(compressed_model_id)
    tokenizer.save_pretrained(compressed_model_id)

    model, tokenizer = load_model_and_tokenizer(compressed_model_id, False)
    answers_by_questions = generate_answers(QUESTIONS, model, tokenizer)
    print_answers("Optimized model outputs:\n", answers_by_questions)

    return list(answers_by_questions.values())


def main():
    res = []  # codebook_example(MODEL_ID, COMPRESSED_MODEL_ID)
    res += codebook_example(MODEL_ID, COMPRESSED_MODEL_ID + "_adaptive", adaptive_codebook=True)

    return res


if __name__ == "__main__":
    main()
