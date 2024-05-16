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

import re
import subprocess
from pathlib import Path
from typing import List, Optional

import numpy as np
import openvino as ov
import torch
from fastdownload import FastDownload
from sklearn.metrics import accuracy_score
from torchvision import datasets
from torchvision import models
from torchvision import transforms

from nncf.common.logging.track_progress import track

ROOT = Path(__file__).parent.resolve()
QUANTIZED_CHECKPOINT_FILE_NAME = "mobilenet_v2_int8_checkpoint.pt"
CHECKPOINT_URL = "https://huggingface.co/alexsu52/mobilenet_v2_imagenette/resolve/main/pytorch_model.bin"
DATASET_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
DATASET_PATH = "~/.cache/nncf/datasets"
DATASET_CLASSES = 10


def download_dataset() -> Path:
    downloader = FastDownload(base=DATASET_PATH, archive="downloaded", data="extracted")
    return downloader.get(DATASET_URL)


def load_baseline_checkpoint(model: torch.nn.Module) -> torch.nn.Module:
    checkpoint = torch.hub.load_state_dict_from_url(CHECKPOINT_URL, map_location=torch.device("cpu"), progress=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def get_data_loader(batch_size=128) -> torch.utils.data.DataLoader:
    dataset_path = download_dataset()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transformations = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]

    val_dataset = datasets.ImageFolder(
        root=dataset_path / "val",
        transform=transforms.Compose(transformations),
    )
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    return val_data_loader


def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_mobilenet_v2(device: torch.device):
    torch_model = models.mobilenet_v2(num_classes=DATASET_CLASSES)
    torch_model = load_baseline_checkpoint(torch_model)
    torch_model.to(device)
    return torch_model


def validate(model: ov.Model, val_loader: torch.utils.data.DataLoader) -> float:
    predictions = []
    references = []

    compiled_model = ov.compile_model(model)
    output = compiled_model.outputs[0]

    for images, target in track(val_loader, description="Validating"):
        pred = compiled_model(images)[output]
        predictions.append(np.argmax(pred, axis=1))
        references.append(target)

    predictions = np.concatenate(predictions, axis=0)
    references = np.concatenate(references, axis=0)
    return accuracy_score(predictions, references)


def run_benchmark(model_path: Path, shape: Optional[List[int]] = None, verbose: bool = True) -> float:
    command = f"benchmark_app -m {model_path} -d CPU -api async -t 15"
    if shape is not None:
        command += f' -shape=[{",".join(str(x) for x in shape)}]'
    cmd_output = subprocess.check_output(command, shell=True)  # nosec
    if verbose:
        print(*str(cmd_output).split("\\n")[-9:-1], sep="\n")
    match = re.search(r"Throughput\: (.+?) FPS", str(cmd_output))
    return float(match.group(1))
