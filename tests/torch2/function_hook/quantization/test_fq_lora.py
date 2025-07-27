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

from pathlib import Path

import pytest
import torch
from networkx.drawing.nx_pydot import to_pydot
from optimum.exporters.openvino.convert import export_from_model
from optimum.intel.openvino import OVModelForCausalLM
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

import nncf
from nncf.data.dataset import Dataset
from nncf.errors import ValidationError
from nncf.parameters import CompressionFormat
from nncf.parameters import CompressWeightsMode
from nncf.parameters import StripFormat
from nncf.quantization.advanced_parameters import AdvancedCompressionParameters
from nncf.quantization.quantize_model import compress_weights
from nncf.torch import load_from_config
from nncf.torch.function_hook.nncf_graph.nncf_graph_builder import build_nncf_graph
from nncf.torch.model_creation import get_config
from nncf.torch.model_creation import wrap_model
from nncf.torch.quantization.layers import AsymmetricQuantizer as AQ
from nncf.torch.quantization.layers import LoraMixin
from nncf.torch.quantization.layers import SymmetricQuantizer as SQ
from tests.cross_fw.shared.paths import TEST_ROOT
from tests.torch.test_models.synthetic import LinearModel
from tests.torch.test_models.synthetic import ShortTransformer
from tests.torch2.utils import compare_with_reference_file
from tests.torch2.utils import to_comparable_nx_graph

REF_DIR = TEST_ROOT / "torch2" / "data" / "function_hook" / "compress_weights" / "fq_lora"


class ValidationMock:
    def __init__(self) -> None:
        model_id = "sentence-transformers/all-mpnet-base-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = SentenceTransformer(
            model_id, tokenizer_kwargs={"pad_token": self.tokenizer.pad_token}, trust_remote_code=True
        )

    def calculate_similarity(self, gold: str, prediction: str) -> torch.Tensor:
        embeddings = self.model.encode([gold, prediction])
        cos_sim = util.cos_sim(embeddings, embeddings)
        return torch.mean(cos_sim)

    @property
    def validation_ref(self) -> torch.Tensor:
        return torch.tensor(1.0)


def generate_control_output(model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> torch.Tensor:
    control_input = tokenizer("What is Pytorch?", return_tensors="pt")
    control_input = control_input.to(model.device)
    control_output = model.generate(**control_input, do_sample=False)
    return tokenizer.batch_decode(control_output, skip_special_tokens=True)[0]


def get_ov_model(model: AutoModelForCausalLM, tmp_path: str) -> OVModelForCausalLM:
    model = model.cpu()

    # TODO(AlexanderDokuchaev): WA for optimum patcher, remove after update optimum-intel>1.22.0
    saved_ops = torch.ops._prepare_4d_causal_attention_mask_for_sdpa
    export_from_model(model, tmp_path)
    torch.ops._prepare_4d_causal_attention_mask_for_sdpa = saved_ops
    return OVModelForCausalLM.from_pretrained(
        model_id=tmp_path,
        trust_remote_code=True,
        load_in_8bit=False,
        compile=True,
    )


@pytest.mark.xfail(reason="issue-171197")
@pytest.mark.cuda
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
def test_fq_lora_tuning(tmp_path, mode, backup_mode, compression_kwargs, ref_num_trainable, _seed):
    model_id = "facebook/opt-125m"
    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    example_inputs = dict(tokenizer("overfit " * 10, return_tensors="pt").to(device))

    except_lm_head_and_5th_vproj = r"^(?!(self_attn/v_proj/linear/5|lm_head/linear/0)$).*"

    model = nncf.compress_weights(
        model,
        group_size=64,
        mode=mode,
        backup_mode=backup_mode,
        dataset=nncf.Dataset([example_inputs]),
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
        input_ids=example_inputs["input_ids"][:, :-1],
        attention_mask=example_inputs["attention_mask"][:, :-1],
        labels=example_inputs["input_ids"][:, 1:],
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

    if compression_kwargs["awq"]:
        return  # Skip test for strip for awq + se initialization. Cases with data-free methods are enough.

    with torch.no_grad():
        tuned_output = generate_control_output(model, tokenizer)

        # Workaround till export from the optimum would be fixed - CVS-164159
        model = model.to(torch.float32)

        model = nncf.strip(model, do_copy=False, strip_format=StripFormat.DQ, example_input=example_inputs)
        stripped_output = generate_control_output(model, tokenizer)

        model = get_ov_model(model, tmp_path)
        stripped_ov_output = generate_control_output(model, tokenizer)

        vm = ValidationMock()
        tuned_vs_stripped = vm.calculate_similarity(tuned_output, stripped_output)
        tuned_vs_stripped_ov = vm.calculate_similarity(tuned_output, stripped_ov_output)

        # torch.compiled version of FQ+LoRA leads to a small error
        atol = 1e-2
        assert torch.allclose(tuned_vs_stripped, vm.validation_ref, atol)
        assert torch.allclose(tuned_vs_stripped_ov, vm.validation_ref, atol)


def test_checkpoint_loading(tmp_path: Path, use_cuda: bool):
    device = "cuda" if use_cuda else "cpu"
    model = ShortTransformer(8, 16, share_weights=True).to(device)
    input_ids = torch.randint(0, 10, (8,)).to(device)

    model = compress_weights(
        model,
        group_size=4,
        mode=CompressWeightsMode.INT4_ASYM,
        backup_mode=CompressWeightsMode.INT8_ASYM,
        dataset=Dataset([input_ids]),
        compression_format=CompressionFormat.FQ_LORA,
        advanced_parameters=AdvancedCompressionParameters(lora_adapter_rank=2),
    )
    with torch.no_grad():
        ref_output = model(input_ids)

    # save checkpoint
    ckpt_path = tmp_path / "nncf_ckpt.pth"
    torch.save(
        {
            "nncf_config": get_config(model),
            "model_state_dict": model.state_dict(),
        },
        ckpt_path,
    )
    del model

    # load checkpoint
    nncf_ckpt = torch.load(ckpt_path, weights_only=False)
    model = ShortTransformer(8, 16, share_weights=True).to(device)
    model = load_from_config(model, nncf_ckpt["nncf_config"], example_input=input_ids)
    model.load_state_dict(nncf_ckpt["model_state_dict"])

    with torch.no_grad():
        actual_output = model(input_ids)
    assert torch.all(actual_output == ref_output)


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


@pytest.mark.parametrize("all_layers", [True, False])
def test_compress_shared_weights(all_layers, regen_ref_data):
    model = ShortTransformer(8, 16, share_weights=True)

    input_ids = torch.randint(0, 10, (8,))
    wrapped_model = wrap_model(model, example_input=input_ids, trace_parameters=True)

    compressed_model = compress_weights(
        wrapped_model,
        mode=CompressWeightsMode.INT4_SYM,
        all_layers=all_layers,
        group_size=4,
        compression_format=CompressionFormat.FQ_LORA,
        advanced_parameters=AdvancedCompressionParameters(lora_adapter_rank=4),
    )
    nncf_graph = build_nncf_graph(compressed_model, example_input=input_ids)
    nx_graph = to_comparable_nx_graph(nncf_graph)
    dot_nncf_graph = to_pydot(nx_graph)
    ref_file = REF_DIR / f"shared_weights_all_layers_{all_layers}.dot"
    compare_with_reference_file(str(dot_nncf_graph), ref_file, regen_ref_data)
