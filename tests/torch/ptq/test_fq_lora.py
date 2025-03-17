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

import pytest
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

import nncf
from nncf.data.dataset import Dataset
from nncf.errors import ValidationError
from nncf.parameters import CompressionFormat
from nncf.parameters import CompressWeightsMode
from nncf.quantization.advanced_parameters import AdvancedCompressionParameters
from nncf.quantization.quantize_model import compress_weights
from nncf.scopes import IgnoredScope
from nncf.torch import load_from_config
from nncf.torch.quantization.layers import AsymmetricQuantizer as AQ
from nncf.torch.quantization.layers import LoraMixin
from nncf.torch.quantization.layers import SymmetricQuantizer as SQ
from tests.torch.test_models.synthetic import LinearModel


@pytest.mark.parametrize(
    "compression_kwargs",
    (dict(scale_estimation=True, awq=True), dict(scale_estimation=False, awq=False)),
    ids=["se_awq", "data_free"],
)
@pytest.mark.parametrize(
    ("mode", "backup_mode", "ref_num_trainable"),
    (
        (nncf.CompressWeightsMode.INT4_ASYM, nncf.CompressWeightsMode.INT8_ASYM, 4 + 2),
        (nncf.CompressWeightsMode.INT4_SYM, nncf.CompressWeightsMode.INT8_SYM, 3 + 1),
    ),
    ids=["asym", "sym"],
)
def test_fq_lora_tuning(mode, backup_mode, compression_kwargs, ref_num_trainable, _seed):
    model_id = "facebook/opt-125m"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    inputs = tokenizer("overfit " * 10, return_tensors="pt").to(device)

    except_lm_head_and_5th_vproj = (
        r"^(?!.*(OPTDecoderLayer\[5\]/OPTSdpaAttention\[self_attn\]/Linear\[v_proj\]/l|lm_head).*$).*$"
    )
    model = nncf.compress_weights(
        model,
        group_size=64,
        mode=mode,
        backup_mode=backup_mode,
        dataset=nncf.Dataset([dict(inputs)]),
        compression_format=nncf.CompressionFormat.FQ_LORA,
        ignored_scope=nncf.IgnoredScope(patterns=[except_lm_head_and_5th_vproj]),
        **compression_kwargs,
    )

    expected_names = {LoraMixin.LORA_A_PARAM_NAME, LoraMixin.LORA_B_PARAM_NAME}
    if mode == nncf.CompressWeightsMode.INT4_ASYM:
        expected_names.update([AQ.INPUT_LOW_PARAM_NAME, AQ._INPUT_RANGE_PARAM_STORAGE_ATTR])
    else:
        expected_names.add(SQ._SCALE_PARAM_STORAGE_ATTR)
    actual_names = {name.split(".")[-1] for name, param in model.named_parameters() if param.requires_grad}
    assert actual_names == expected_names
    actual_num_trainable = sum(1 for param in model.parameters() if param.requires_grad)
    assert actual_num_trainable == ref_num_trainable

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    model_kwargs = dict(
        input_ids=inputs["input_ids"][:, :-1],
        attention_mask=inputs["attention_mask"][:, :-1],
        labels=inputs["input_ids"][:, 1:],
    )
    for i in range(5):
        optimizer.zero_grad()
        loss = model(**model_kwargs).loss
        if i == 0:
            first_loss = float(loss)
        loss.backward()
        optimizer.step()

    assert first_loss > 8
    assert float(loss) < 1


def test_checkpoint_loading(tmp_path):
    model_id = "hf-internal-testing/tiny-random-GPTNeoXForCausalLM"
    if not torch.cuda.is_available():
        pytest.skip("Skipping CUDA test case for CPU only setups.")
    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    example_input = tokenizer("dummy", return_tensors="pt").to(device)
    except_lm_head_and_5th_vproj = (
        r"^(?!.*(GPTNeoXLayer\[2\]/GPTNeoXSdpaAttention\[attention\]/Linear\[query_key_value\]/l|embed_out).*$).*$"
    )
    model = compress_weights(
        model,
        group_size=32,
        mode=CompressWeightsMode.INT4_ASYM,
        backup_mode=CompressWeightsMode.INT8_ASYM,
        dataset=Dataset([dict(example_input)]),
        compression_format=CompressionFormat.FQ_LORA,
        ignored_scope=IgnoredScope(patterns=[except_lm_head_and_5th_vproj]),
        advanced_parameters=AdvancedCompressionParameters(lora_adapter_rank=2),
    )
    ref_output = tokenizer.decode(
        model.generate(**example_input, do_sample=False, max_new_tokens=20)[0], skip_special_tokens=True
    )

    # save checkpoint
    ckpt_path = tmp_path / "nncf_ckpt.pth"
    torch.save(
        {
            "nncf_state_dict": model.nncf.state_dict(),
            "nncf_config": model.nncf.get_config(),
        },
        ckpt_path,
    )
    del model

    # load checkpoint
    nncf_ckpt = torch.load(ckpt_path, weights_only=False)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
    model = load_from_config(model, nncf_ckpt["nncf_config"], example_input=dict(example_input))
    model.nncf.load_state_dict(nncf_ckpt["nncf_state_dict"])

    actual_output = tokenizer.decode(
        model.generate(**example_input, do_sample=False, max_new_tokens=20)[0],
        skip_special_tokens=True,
    )
    assert actual_output == ref_output


def test_invalid_lora_rank():
    too_big_rank = 4
    model = LinearModel(torch.ones(2, 2))
    with pytest.raises(ValidationError):
        compress_weights(
            model,
            mode=CompressWeightsMode.INT4_ASYM,
            group_size=2,
            all_layers=True,
            dataset=Dataset([torch.ones(2, 2)]),
            compression_format=CompressionFormat.FQ_LORA,
            advanced_parameters=AdvancedCompressionParameters(lora_adapter_rank=too_big_rank),
        )
