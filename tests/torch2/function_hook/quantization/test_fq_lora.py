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

from pathlib import Path

import openvino as ov
import pytest
import torch
from networkx.drawing.nx_pydot import to_pydot

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
from tests.torch2.function_hook.quantization.test_weights_compression import AWQLinearModel
from tests.torch2.utils import compare_with_reference_file
from tests.torch2.utils import to_comparable_nx_graph

REF_DIR = TEST_ROOT / "torch2" / "data" / "function_hook" / "compress_weights" / "fq_lora"


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("mode", "backup_mode", "ref_num_trainable"),
    (
        # LoRA A, LoRA B, input_low, input_range for single linear layer
        (nncf.CompressWeightsMode.INT4_ASYM, nncf.CompressWeightsMode.INT8_ASYM, 4),
        # LoRA A, LoRA B, scale for single linear layer
        (nncf.CompressWeightsMode.INT4_SYM, nncf.CompressWeightsMode.INT8_SYM, 3),
    ),
    ids=["asym", "sym"],
)
def test_fq_lora_tuning(mode, backup_mode, ref_num_trainable, _seed):
    """
    Tests FQ-LoRA (Fake-Quantize with Low-Rank Adaptation) fine-tuning.
    Verifies:
    1. Weight compression with FQ-LoRA properly sets up trainable parameters
    2. Model can be fine-tuned after quantization
    3. Loss decreases significantly after fine-tuning
    """
    device = "cuda"
    MODEL_DIM = 32
    model = LinearModel(torch.randn(MODEL_DIM, MODEL_DIM)).to(device)
    example_inputs = torch.randn(1, MODEL_DIM, device=device)

    model = nncf.compress_weights(
        model,
        group_size=-1,
        mode=mode,
        backup_mode=backup_mode,
        dataset=nncf.Dataset([example_inputs]),
        compression_format=nncf.CompressionFormat.FQ_LORA,
        advanced_parameters=AdvancedCompressionParameters(lora_adapter_rank=8),
    )
    # Verify the correct trainable parameters are set based on quantization mode
    trainable_params = [name.split(".")[-1] for name, param in model.named_parameters() if param.requires_grad]

    # Both modes should have LoRA A and B parameters trainable
    assert LoraMixin.LORA_A_PARAM_NAME in trainable_params
    assert LoraMixin.LORA_B_PARAM_NAME in trainable_params

    # Mode-specific parameters
    if mode == nncf.CompressWeightsMode.INT4_ASYM:
        assert AQ.INPUT_LOW_PARAM_NAME in trainable_params
        assert AQ._INPUT_RANGE_PARAM_STORAGE_ATTR in trainable_params
    else:  # Symmetric mode
        assert SQ._SCALE_PARAM_STORAGE_ATTR in trainable_params

    # Verify total number of trainable parameters
    assert len(trainable_params) == ref_num_trainable

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    target = torch.zeros(1, MODEL_DIM, device=device)
    loss_fn = torch.nn.MSELoss()
    for i in range(10):
        optimizer.zero_grad()
        output = model(example_inputs)
        loss = loss_fn(output, target)
        if i == 0:
            first_loss = float(loss)
        loss.backward()
        optimizer.step()

    assert first_loss > 30
    assert float(loss) < 10


@pytest.mark.cuda
@pytest.mark.parametrize(
    "compression_kwargs",
    (dict(scale_estimation=True, awq=True), dict(scale_estimation=False, awq=False)),
    ids=["se_awq", "data_free"],
)
def test_fq_lora_export(compression_kwargs, _seed):
    """
    Tests FQ-LoRA (Fake-Quantize with Low-Rank Adaptation) can be stripped and exported to OpenVINO.
    """
    device = "cuda"
    example_input = 0.01 * torch.arange(0, 4 * 8, device=device).reshape(1, 4, 8) + 0.02

    model = AWQLinearModel().to(device)
    model = nncf.compress_weights(
        model,
        group_size=-1,
        mode=nncf.CompressWeightsMode.INT4_ASYM,
        dataset=nncf.Dataset([example_input]),
        compression_format=nncf.CompressionFormat.FQ_LORA,
        advanced_parameters=AdvancedCompressionParameters(lora_adapter_rank=8),
        **compression_kwargs,
    )

    with torch.no_grad():
        tuned_output = model(example_input)
        model = nncf.strip(model, do_copy=False, strip_format=StripFormat.DQ, example_input=example_input)
        stripped_output = model(example_input)
        model = ov.convert_model(model, example_input=example_input)
        model = ov.compile_model(model)
        example_inputs_numpy = example_input.detach().cpu().numpy()
        stripped_ov_output = torch.tensor(model(example_inputs_numpy)[0], device=example_input.device)

        assert torch.allclose(tuned_output, stripped_output, atol=1e-1)
        assert torch.allclose(tuned_output, stripped_ov_output, atol=1e-1)


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
