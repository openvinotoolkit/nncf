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

import re
import subprocess
from functools import partial
from pathlib import Path
from typing import List, Tuple

import numpy as np
import openvino as ov
import torch
from fastdownload import FastDownload
from rich.progress import track
from sklearn.metrics import accuracy_score
from torchvision import datasets
from torchvision import models
from torchvision import transforms

import nncf

ROOT = Path(__file__).parent.resolve()
CHECKPOINT_URL = "https://huggingface.co/alexsu52/mobilenet_v2_imagenette/resolve/main/pytorch_model.bin"
DATASET_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
DATASET_PATH = Path().home() / ".cache" / "nncf" / "datasets"
DATASET_CLASSES = 10


def download_dataset() -> Path:
    downloader = FastDownload(base=DATASET_PATH.resolve(), archive="downloaded", data="extracted")
    return downloader.get(DATASET_URL)


def load_checkpoint(model: torch.nn.Module) -> torch.nn.Module:
    checkpoint = torch.hub.load_state_dict_from_url(CHECKPOINT_URL, map_location=torch.device("cpu"), progress=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def validate(model: ov.Model, val_loader: torch.utils.data.DataLoader) -> float:
    predictions = []
    references = []

    compiled_model = ov.compile_model(model, device_name="CPU")
    output = compiled_model.outputs[0]

    for images, target in track(val_loader, description="Validating"):
        pred = compiled_model(images)[output]
        predictions.append(np.argmax(pred, axis=1))
        references.append(target)

    predictions = np.concatenate(predictions, axis=0)
    references = np.concatenate(references, axis=0)
    return accuracy_score(predictions, references)


def run_benchmark(model_path: Path, shape: List[int]) -> float:
    command = [
        "benchmark_app",
        "-m", model_path.as_posix(),
        "-d", "CPU",
        "-api", "async",
        "-t", "15",
        "-shape", str(shape),
    ]  # fmt: skip
    cmd_output = subprocess.check_output(command, text=True)  # nosec
    print(*cmd_output.splitlines()[-8:], sep="\n")
    match = re.search(r"Throughput\: (.+?) FPS", cmd_output)
    return float(match.group(1))


def get_model_size(ir_path: Path, m_type: str = "Mb") -> float:
    xml_size = ir_path.stat().st_size
    bin_size = ir_path.with_suffix(".bin").stat().st_size
    for t in ["bytes", "Kb", "Mb"]:
        if m_type == t:
            break
        xml_size /= 1024
        bin_size /= 1024
    model_size = xml_size + bin_size
    print(f"Model graph (xml):   {xml_size:.3f} {m_type}")
    print(f"Model weights (bin): {bin_size:.3f} {m_type}")
    print(f"Model size:          {model_size:.3f} {m_type}")
    return model_size


###############################################################################
# Create a PyTorch model and dataset

dataset_path = download_dataset()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
val_dataset = datasets.ImageFolder(
    root=dataset_path / "val",
    transform=transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    ),
)
batch_size = 128
val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

torch_model = models.mobilenet_v2(num_classes=DATASET_CLASSES)
torch_model = load_checkpoint(torch_model)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch_model.to(device)
torch_model.eval()

###############################################################################
# Quantize a PyTorch model

# The transformation function transforms a data item into model input data.
#
# To validate the transform function use the following code:
# >> for data_item in val_loader:
# >>    model(transform_fn(data_item, device))


def transform_fn(data_item: Tuple[torch.Tensor, int], device: torch.device) -> torch.Tensor:
    images, _ = data_item
    return images.to(device)


# The calibration dataset is a small, no label, representative dataset
# (~100-500 samples) that is used to estimate the range, i.e. (min, max) of all
# floating point activation tensors in the model, to initialize the quantization
# parameters.

# The easiest way to define a calibration dataset is to use a training or
# validation dataset and a transformation function to remove labels from the data
# item and prepare model input data. The quantize method uses a small subset
# (default: 300 samples) of the calibration dataset.

# Recalculation default subset_size parameter based on batch_size.
subset_size = 300 // batch_size
calibration_dataset = nncf.Dataset(val_data_loader, partial(transform_fn, device=device))
torch_quantized_model = nncf.quantize(torch_model, calibration_dataset, subset_size=subset_size)

###############################################################################
# Benchmark performance, calculate compression rate and validate accuracy

dummy_input = torch.randn(1, 3, 224, 224)
ov_model = ov.convert_model(torch_model.cpu(), example_input=dummy_input)
ov_quantized_model = ov.convert_model(torch_quantized_model.cpu(), example_input=dummy_input)

fp32_ir_path = ROOT / "mobilenet_v2_fp32.xml"
ov.save_model(ov_model, fp32_ir_path, compress_to_fp16=False)
print(f"[1/7] Save FP32 model: {fp32_ir_path}")
fp32_model_size = get_model_size(fp32_ir_path)

int8_ir_path = ROOT / "mobilenet_v2_int8.xml"
ov.save_model(ov_quantized_model, int8_ir_path)
print(f"[2/7] Save INT8 model: {int8_ir_path}")
int8_model_size = get_model_size(int8_ir_path)

print("[3/7] Benchmark FP32 model:")
fp32_fps = run_benchmark(fp32_ir_path, shape=[1, 3, 224, 224])
print("[4/7] Benchmark INT8 model:")
int8_fps = run_benchmark(int8_ir_path, shape=[1, 3, 224, 224])

print("[5/7] Validate OpenVINO FP32 model:")
fp32_top1 = validate(ov_model, val_data_loader)
print(f"Accuracy @ top1: {fp32_top1:.3f}")

print("[6/7] Validate OpenVINO INT8 model:")
int8_top1 = validate(ov_quantized_model, val_data_loader)
print(f"Accuracy @ top1: {int8_top1:.3f}")

print("[7/7] Report:")
print(f"Accuracy drop: {fp32_top1 - int8_top1:.3f}")
print(f"Model compression rate: {fp32_model_size / int8_model_size:.3f}")
# https://docs.openvino.ai/latest/openvino_docs_optimization_guide_dldt_optimization_guide.html
print(f"Performance speed up (throughput mode): {int8_fps / fp32_fps:.3f}")
