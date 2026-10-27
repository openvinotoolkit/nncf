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
import os
import time
import warnings
from functools import partial
from itertools import chain

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import PreTrainedModel
from transformers.integrations.executorch import TorchExportableModuleForDecoderOnlyLM
from transformers.integrations.executorch import convert_and_export_with_cache
from transformers.models.llama import LlamaTokenizerFast

import nncf

warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_CACHE_LEN = 512
MAX_NEW_TOKENS = 100


def transform_fn(data: str, tokenizer: LlamaTokenizerFast) -> tuple[torch.Tensor, torch.Tensor]:
    tokenized_text = tokenizer(data["text"], max_length=MAX_CACHE_LEN, truncation=True, return_tensors="pt")
    input_ids = tokenized_text["input_ids"]
    cache_position = torch.arange(input_ids.shape[1], dtype=torch.long)
    return {"input_ids": input_ids, "cache_position": cache_position}


@torch.no_grad()
def export_with_static_cache(model: PreTrainedModel) -> torch.export.ExportedProgram:
    """
    Export a `PreTrainedModel` with a static KV cache via `torch.export`,
    with `input_ids` and `cache_position` inputs of dynamic sequence length.
    """
    model.generation_config.use_cache = True
    model.generation_config.cache_implementation = "static"
    model.generation_config.cache_config = {"batch_size": 1, "max_cache_len": MAX_CACHE_LEN}

    exported_program = convert_and_export_with_cache(
        model,
        example_input_ids=torch.ones(1, 8, dtype=torch.long),
        example_cache_position=torch.arange(8, dtype=torch.long),
        dynamic_shapes={"input_ids": {1: torch.export.Dim.DYNAMIC}, "cache_position": {0: torch.export.Dim.DYNAMIC}},
        strict=True,
    )
    return exported_program


class TextGenerator:
    """
    Wrapper `TorchExportableModuleForDecoderOnlyLM.generate` to provide a simpler interface for text generation.
    """

    def __init__(self, model: torch.nn.Module, tokenizer: LlamaTokenizerFast, max_new_tokens: int):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    def module(self) -> torch.nn.Module:
        # `generate` expects an ExportedProgram and only calls its `.module()`.
        return self.model

    @torch.no_grad()
    def generate(self, prompt: str) -> str:
        return TorchExportableModuleForDecoderOnlyLM.generate(
            self,
            self.tokenizer,
            prompt,
            max_new_tokens=self.max_new_tokens,
        )


def model_size(model: torch.nn.Module) -> int:
    """Calculate the size of the model in Mebibytes."""
    param_size = sum(param.nelement() * param.element_size() for param in chain(model.parameters(), model.buffers()))
    size_all_mb = param_size / 1000**2
    return round(size_all_mb)


@torch.no_grad
def run_text_generation(model, tokenizer):
    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot who always responds in the style of a pirate",
        },
        {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    text_generator = TextGenerator(model, tokenizer, max_new_tokens=MAX_NEW_TOKENS)

    # Single forward to trigger torch.compile so compilation time is excluded from the measurement.
    model(input_ids=torch.zeros(1, 1, dtype=torch.long), cache_position=torch.zeros(1, dtype=torch.long))

    start_t = time.time()
    output_text = text_generator.generate(prompt)
    elapsed = round(time.time() - start_t)
    print(f"Elapsed time: {elapsed} seconds")
    print(f"Model size:   {model_size(model)} MiB")
    print("Output: ")
    print(output_text)
    return output_text


def main() -> str:

    ###############################################################################
    # Step 1: Prepare model and dataset
    print(os.linesep + "[Step 1] Prepare model and dataset")

    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model_hf = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    quantization_dataset = nncf.Dataset(dataset, partial(transform_fn, tokenizer=tokenizer))

    exported_model = export_with_static_cache(model_hf)

    run_text_generation(exported_model.module(), tokenizer)

    ###############################################################################
    # Step 2: Compress the model
    print(os.linesep + "[Step 2] Compressing the model")
    compressed_model = nncf.compress_weights(
        exported_model.module(),
        dataset=quantization_dataset,
        mode=nncf.CompressWeightsMode.INT4_SYM,
        ratio=0.8,
        sensitivity_metric=nncf.SensitivityMetric.HESSIAN_INPUT_ACTIVATION,
    )
    compiled_model = torch.compile(compressed_model, backend="openvino")

    ###############################################################################
    # Step 3 Running text generation with compressed model
    print(os.linesep + "[Step 3] Running text generation with compressed model")
    output_text = run_text_generation(compiled_model, tokenizer)

    return output_text


if __name__ == "__main__":
    main()
