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

import torch
from optimum.exporters.openvino.convert import export_from_model
from torch import nn
from transformers import AutoModelForCausalLM

import nncf
from nncf.parameters import StripFormat
from nncf.torch.function_hook.wrapper import get_hook_storage
from nncf.torch.model_creation import load_from_config
from nncf.torch.quantization.layers import SymmetricLoraQuantizer  # noqa: F401


def load_checkpoint(model: nn.Module, ckpt_file: Path) -> nn.Module:
    """
    Loads the state of a tuned model from a checkpoint. This function restores the placement of Fake Quantizers (FQs)
    with absorbable LoRA adapters and loads their parameters.

    :param model: The model to load the checkpoint into.
    :param ckpt_file: Path to the checkpoint file.
    :returns: The model with the loaded NNCF state from checkpoint.
    """
    ckpt = torch.load(ckpt_file, weights_only=False, map_location="cpu")
    model = load_from_config(model, ckpt["nncf_config"])
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    hook_storage = get_hook_storage(model)
    hook_storage.load_state_dict(ckpt["nncf_state_dict"])
    return model


pretrained = "Qwen/Qwen3-4B"
ckpt_file = "nncf_checkpoint_epoch10.pth"
ir_dir = "u2_u4_ov_model"
with torch.no_grad():
    model_to_eval = AutoModelForCausalLM.from_pretrained(pretrained, torch_dtype=torch.float32, device_map="cpu")
    model_to_eval = load_checkpoint(model_to_eval, ckpt_file)
    model_to_eval = nncf.strip(model_to_eval, do_copy=False, strip_format=StripFormat.DQ)
    export_from_model(model_to_eval, ir_dir, device="cpu")
