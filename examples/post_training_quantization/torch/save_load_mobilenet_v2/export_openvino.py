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
from pathlib import Path

import openvino as ov
import torch

import nncf.torch
from examples.post_training_quantization.torch.save_load_mobilenet_v2.common import QUANTIZED_CHECKPOINT_FILE_NAME
from examples.post_training_quantization.torch.save_load_mobilenet_v2.common import ROOT
from examples.post_training_quantization.torch.save_load_mobilenet_v2.common import get_data_loader
from examples.post_training_quantization.torch.save_load_mobilenet_v2.common import get_mobilenet_v2
from examples.post_training_quantization.torch.save_load_mobilenet_v2.common import run_benchmark
from examples.post_training_quantization.torch.save_load_mobilenet_v2.common import validate


def get_model_size(ir_path: Path, m_type: str = "Mb", verbose: bool = True) -> float:
    xml_size = os.path.getsize(ir_path)
    bin_size = os.path.getsize(os.path.splitext(ir_path)[0] + ".bin")
    for t in ["bytes", "Kb", "Mb"]:
        if m_type == t:
            break
        xml_size /= 1024
        bin_size /= 1024
    model_size = xml_size + bin_size
    if verbose:
        print(f"Model graph (xml):   {xml_size:.3f} {m_type}")
        print(f"Model weights (bin): {bin_size:.3f} {m_type}")
        print(f"Model size:          {model_size:.3f} {m_type}")
    return model_size


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

ov_model = ov.convert_model(torch_model, example_input=dummy_input)
ov_quantized_model = ov.convert_model(torch_quantized_model, example_input=dummy_input)

fp32_ir_path = ROOT / "mobilenet_v2_fp32.xml"
ov.save_model(ov_model, fp32_ir_path, compress_to_fp16=False)
print(f"[2/8] Save FP32 model: {fp32_ir_path}")
fp32_model_size = get_model_size(fp32_ir_path, verbose=True)

int8_ir_path = ROOT / "mobilenet_v2_int8.xml"
ov.save_model(ov_quantized_model, int8_ir_path, compress_to_fp16=False)
print(f"[3/8] Save INT8 model: {int8_ir_path}")
int8_model_size = get_model_size(int8_ir_path, verbose=True)

print("[4/8] Benchmark FP32 model:")
fp32_fps = run_benchmark(fp32_ir_path, shape=[1, 3, 224, 224], verbose=True)
print("[5/8] Benchmark INT8 model:")
int8_fps = run_benchmark(int8_ir_path, shape=[1, 3, 224, 224], verbose=True)

val_data_loader = get_data_loader()

print("[6/8] Validate OpenVINO FP32 model:")
fp32_top1 = validate(ov_model, val_data_loader)
print(f"Accuracy @ top1: {fp32_top1:.3f}")

print("[7/8] Validate OpenVINO INT8 model:")
int8_top1 = validate(ov_quantized_model, val_data_loader)
print(f"Accuracy @ top1: {int8_top1:.3f}")

print("[8/8] Report:")
print(f"Accuracy drop: {fp32_top1 - int8_top1:.3f}")
print(f"Model compression rate: {fp32_model_size / int8_model_size:.3f}")
# https://docs.openvino.ai/latest/openvino_docs_optimization_guide_dldt_optimization_guide.html
print(f"Performance speed up (throughput mode): {int8_fps / fp32_fps:.3f}")
