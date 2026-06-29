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

import pytest
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

import nncf
from nncf.data import generate_text_data
from tests.cross_fw.shared.json import load_json
from tests.cross_fw.shared.paths import TEST_ROOT
from tests.torch.helpers import set_torch_seed

BASE_TEST_MODEL_ID = "hf-internal-testing/tiny-random-gpt2"
GENERATED_TEXT_REF = TEST_ROOT / "torch" / "data" / "ref_generated_data.json"


@pytest.mark.parametrize(
    "with_model, with_tokenizer, usage_error",
    [
        (False, False, True),
        (True, False, True),
        (False, True, True),
        (True, True, False),
    ],
    ids=["no_model_no_tokenizer", "model_only", "tokenizer_only", "model_and_tokenizer"],
)
def test_generate_text_data_usage(with_model: bool, with_tokenizer: bool, usage_error: bool, _seed: None):
    model = AutoModelForCausalLM.from_pretrained(BASE_TEST_MODEL_ID) if with_model else None
    tokenizer = AutoTokenizer.from_pretrained(BASE_TEST_MODEL_ID) if with_tokenizer else None

    if usage_error:
        with pytest.raises(nncf.ValidationError):
            generate_text_data(model, tokenizer, seq_len=32, dataset_size=1)
    else:
        generate_text_data(model, tokenizer, seq_len=32, dataset_size=1)


@pytest.mark.parametrize("model_cls", [AutoModelForCausalLM, OVModelForCausalLM])
def test_generate_text_data_functional(model_cls):
    seq_len = 12
    max_seq_len = seq_len + seq_len // 2
    dataset_size = 6

    model = model_cls.from_pretrained(BASE_TEST_MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(BASE_TEST_MODEL_ID)

    with set_torch_seed(0):
        generated_data = generate_text_data(
            model,
            tokenizer,
            seq_len=seq_len,
            dataset_size=dataset_size,
        )

    assert len(generated_data) == dataset_size
    generated_data = [tokenizer.encode(d) for d in generated_data]

    # Uncomment lines below to generate reference for new models.
    # from tests.cross_fw.shared.json import dump_to_json
    # dump_to_json(GENERATED_TEXT_REF, generated_data)

    reference_data = load_json(GENERATED_TEXT_REF)
    for ref_data, gen_data in zip(reference_data, generated_data):
        assert len(gen_data) <= max_seq_len
        assert ref_data == gen_data
