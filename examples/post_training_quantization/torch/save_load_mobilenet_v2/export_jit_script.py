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

import os
from copy import deepcopy

import openvino as ov
import torch
from common import QUANTIZED_CHECKPOINT_FILE_NAME
from common import ROOT
from common import get_data_loader
from common import get_mobilenet_v2
from common import validate

import nncf.torch

###############################################################################
# Recover the quantized model, benchmark performance, calculate compression rate and validate accuracy

device = torch.device("cpu")
torch_model = get_mobilenet_v2(device)
dummy_input = torch.randn(1, 3, 224, 224).to(device)

quantized_checkpoint_path = ROOT / QUANTIZED_CHECKPOINT_FILE_NAME
print(f"[1/8] Recover INT8 PyTorch model model from the checkpoint: {quantized_checkpoint_path }")
if not os.path.isfile(quantized_checkpoint_path):
    raise RuntimeError(
        f"File {quantized_checkpoint_path} is not found."
        " Please quantize the model first by running quantize.py script."
    )

quantized_checkpoint = torch.load(quantized_checkpoint_path)
torch_quantized_model = nncf.torch.load_from_config(
    deepcopy(torch_model), quantized_checkpoint["nncf_config"], dummy_input
)
torch_quantized_model.load_state_dict(quantized_checkpoint["model_state_dict"])

jit_model = ov.convert_model(torch_model, example_input=dummy_input)
jit_quantized_model = ov.convert_model(torch_quantized_model, example_input=dummy_input)

val_data_loader = get_data_loader()

# TODO: add benchmarking

print("[6/8] Validate OpenVINO FP32 model:")
fp32_top1 = validate(jit_model, val_data_loader)
print(f"Accuracy @ top1: {fp32_top1:.3f}")

print("[7/8] Validate OpenVINO INT8 model:")
int8_top1 = validate(jit_quantized_model, val_data_loader)
print(f"Accuracy @ top1: {int8_top1:.3f}")

print("[8/8] Report:")
print(f"Accuracy drop: {fp32_top1 - int8_top1:.3f}")
# https://docs.openvino.ai/latest/openvino_docs_optimization_guide_dldt_optimization_guide.html
