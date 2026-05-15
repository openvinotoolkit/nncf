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

import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import GenerationMixin
from transformers.tokenization_utils import PreTrainedTokenizer

import nncf
from nncf.common.utils.helpers import create_table
from nncf.experimental.torch.gptqmodel.convertor import convert_model

MODEL_ID = "facebook/opt-125m"
BENCH_NUM_ITERS = 100


@torch.no_grad()
def benchmark(model: GenerationMixin, tokenizer: PreTrainedTokenizer, example_input: torch.Tensor) -> tuple[str, float]:
    """
    Simple benchmark model and check generation.

    Warm up: generate text from example input.
    Benchmark: run inference on random data.

    :param model: The language model for generation.
    :param tokenizer: The tokenizer for decoding generated tokens.
    :param example_input: The example input tensor.
    :returns: Generated text and time taken for benchmark.
    """
    torch.manual_seed(0)

    print("")
    print("Generation:")
    output = model.generate(example_input, max_new_tokens=30)
    answer = tokenizer.decode(output[0])
    print(answer)

    # Benchmark
    print("Benchmarking...")
    dataset = [torch.randint_like(example_input, 0, 10000) for _ in range(BENCH_NUM_ITERS)]
    start_time = time.time()
    for x in dataset:
        model(x)
    duration = time.time() - start_time
    print(f"Time taken: {duration:.3f} seconds")

    return answer, duration


def main() -> str:
    if not torch.cuda.is_available():
        msg = "This example requires a GPU with CUDA support."
        raise RuntimeError(msg)

    ###############################################################################
    # Step 1: Prepare model and dataset
    print(f"[Step 1] Prepare {MODEL_ID} model and dataset")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.cuda()
    example_input = tokenizer.encode("Who is Mark Twain?", return_tensors="pt").cuda()
    _, time_orig = benchmark(model, tokenizer, example_input)

    ###############################################################################
    # Step 2: Compress model by NNCF
    print(os.linesep + "[Step 2] Compress model by NNCF")

    nncf_model = nncf.compress_weights(
        model,
        dataset=nncf.Dataset([example_input]),
        mode=nncf.CompressWeightsMode.INT8_ASYM,
        ignored_scope=nncf.IgnoredScope(
            patterns=[
                "embed_tokens/.*",
                "embed_positions/.*",
                "lm_head/linear/0",
            ],
        ),
    )
    _, time_nncf = benchmark(model, tokenizer, example_input)

    ###############################################################################
    # Step 3: Convert to GPTQModel format
    print(os.linesep + "[Step 3] Convert to GPTQModel format")

    converted_model = convert_model(nncf_model)
    answer, time_gptq = benchmark(converted_model, tokenizer, example_input)

    ###############################################################################
    # Step 4: Summary
    print(os.linesep + "[Step 4] Summary")
    tabular_data = [
        ["Original", time_orig],
        ["Compressed by NNCF", time_nncf],
        ["Converted to GPTQModel", time_gptq],
    ]
    print(create_table(["Time (s)"], tabular_data))

    print(len(answer.split()))
    return answer


if __name__ == "__main__":
    main()
