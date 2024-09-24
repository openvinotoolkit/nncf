# Copyright (c) 2024 Intel Corporation
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

BASE_VOCAB_SIZE = 12000

TModel = TypeVar("TModel")
TTokenizer = TypeVar("TTokenizer")


def generate_text_data(
    model: TModel, tokenizer: TTokenizer, seq_len: int = 32, dataset_size: int = 128, unique_tokens_lower_limit: int = 5
) -> List[str]:
    """
    Generates text dataset based on the model output.

    :param model: Model instance.
    :param tokenizer: Tokenizer instance.
    :param seq_len: Sequence length for generation.
    :param dataset_size: Size of the data.
    :return: List of the text data ready to use.
    """

    try:
        import torch
        from transformers import PreTrainedModel
        from transformers import PreTrainedTokenizerBase
        from transformers.utils import logging

        logging.set_verbosity_error()
    except ImportError:
        raise nncf.ModuleNotFoundError("Install `nncf/helpers/requirements.txt` before using `nncf.helpers` module.")

    if not isinstance(model, PreTrainedModel.__bases__) or not isinstance(tokenizer, PreTrainedTokenizerBase.__bases__):
        raise nncf.ValidationError(
            "Model and tokenizer should be instance of the "
            "`transformers.PreTrainedModel` and `transformers.PreTrainedTokenizerBase` respectively."
        )

    generated_data = []

    vocab_size_names = ["padded_vocab_size", "vocab_size"]
    vocab_size = BASE_VOCAB_SIZE
    for vocab_size_name in vocab_size_names:
        if hasattr(model.config, vocab_size_name):
            vocab_size = getattr(model.config, vocab_size_name)

    step_num = max(1, vocab_size // dataset_size)
    ids_counter = 0

    while len(generated_data) < dataset_size:
        # Creating the input for pre-generate step
        input_ids = torch.tensor([[ids_counter % vocab_size]])

        # Collecting data from the pre & post generate steps
        outputs_prep = model.generate(input_ids, do_sample=False, max_length=seq_len // 2)
        outputs_post = model.generate(outputs_prep, do_sample=True, max_length=seq_len + seq_len // 2)
        gen_text = tokenizer.batch_decode(outputs_post[:, outputs_prep.shape[1] :], skip_special_tokens=True)

        if len(set(gen_text[0])) < unique_tokens_lower_limit:
            ids_counter += 1
            continue

        ids_counter += step_num

        generated_data.extend(gen_text)

    return generated_data
