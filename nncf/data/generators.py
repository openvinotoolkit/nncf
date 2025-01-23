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

from typing import List, TypeVar

import nncf
from nncf.common.logging.track_progress import track
from nncf.common.utils.api_marker import api
from nncf.telemetry import tracked_function
from nncf.telemetry.events import NNCF_CATEGORY
from nncf.telemetry.extractors import FunctionCallTelemetryExtractor

BASE_VOCAB_SIZE = 12000

TModel = TypeVar("TModel")
TTokenizer = TypeVar("TTokenizer")


@api(canonical_alias="nncf.data.generate_text_data")
@tracked_function(
    category=NNCF_CATEGORY,
    extractors=[
        FunctionCallTelemetryExtractor("nncf.data.generate_text_data"),
    ],
)
def generate_text_data(
    model: TModel, tokenizer: TTokenizer, seq_len: int = 32, dataset_size: int = 128, unique_tokens_lower_limit: int = 5
) -> List[str]:
    """
    Generates text dataset based on the model output.

    Since the model is required to be the instance of the PreTrainedModel
    and the tokenizer is required to be the instance of the PreTrainedTokenizerBase,
    environment must have `transformers` & `torch` modules installed to run this method.

    :param model: Model instance.
    :param tokenizer: Tokenizer instance.
    :param seq_len: Sequence length for generation.
    :param dataset_size: Size of the data.
    :return: List of the text data ready to use.
    """

    try:
        import torch
    except ImportError:
        raise nncf.ModuleNotFoundError("torch is required in order to generate text data: `pip install torch`.")

    try:
        from transformers import PreTrainedModel  # type: ignore
        from transformers import PreTrainedTokenizerBase
        from transformers.utils import logging  # type: ignore

        logging.set_verbosity_error()
    except ImportError:
        raise nncf.ModuleNotFoundError(
            "transformers is required in order to generate text data: `pip install transformers`."
        )

    if not isinstance(model, PreTrainedModel.__bases__):
        raise nncf.ValidationError("Model should be instance of the `transformers.PreTrainedModel`.")

    if not isinstance(tokenizer, PreTrainedTokenizerBase.__bases__):
        raise nncf.ValidationError("tokenizer should be instance of the `transformers.PreTrainedTokenizerBase`.")

    generated_data: List[str] = []

    vocab_size_names = ["padded_vocab_size", "vocab_size"]
    vocab_size = BASE_VOCAB_SIZE
    for vocab_size_name in vocab_size_names:
        if hasattr(model.config, vocab_size_name):
            vocab_size = getattr(model.config, vocab_size_name)

    step_num = max(1, vocab_size // dataset_size)
    ids_counter = 0

    with track[None](total=dataset_size, description="Generating text data") as pbar:
        while len(generated_data) < dataset_size:
            # Creating the input for pre-generate step
            input_ids = torch.tensor([[ids_counter % vocab_size]]).to(model.device)

            # Collecting data from the pre & post generate steps
            outputs_prep = model.generate(input_ids, do_sample=False, max_length=seq_len // 2)
            outputs_post = model.generate(outputs_prep, do_sample=True, max_length=seq_len + seq_len // 2)
            gen_text = tokenizer.batch_decode(outputs_post[:, outputs_prep.shape[1] :], skip_special_tokens=True)

            if len(set(gen_text[0])) < unique_tokens_lower_limit:
                ids_counter += 1
                continue

            ids_counter += step_num

            pbar.update(advance=1)
            generated_data.extend(gen_text)

    return generated_data
