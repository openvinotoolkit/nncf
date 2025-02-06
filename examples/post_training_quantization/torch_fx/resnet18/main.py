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

import os
from pathlib import Path
from time import time
from typing import Tuple

# We need to import openvino.torch for torch.compile() with openvino backend to work.
import openvino.torch  # noqa
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from fastdownload import FastDownload
from rich.progress import track
from torch._dynamo.exc import BackendCompilerFailed

import nncf
import nncf.torch
from nncf.common.utils.helpers import create_table
from nncf.common.utils.os import is_windows
from nncf.torch import disable_patching

IMAGE_SIZE = 64


ROOT = Path(__file__).parent.resolve()
BEST_CKPT_NAME = "resnet18_int8_best.pt"
CHECKPOINT_URL = (
    "https://storage.openvinotoolkit.org/repositories/nncf/openvino_notebook_ckpts/302_resnet18_fp32_v1.pth"
)
DATASET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
DATASET_PATH = Path().home() / ".cache" / "nncf" / "datasets"


def download_dataset() -> Path:
    downloader = FastDownload(base=DATASET_PATH.resolve(), archive="downloaded", data="extracted")
    return downloader.get(DATASET_URL)


def load_checkpoint(model: torch.nn.Module) -> torch.nn.Module:
    checkpoint = torch.hub.load_state_dict_from_url(CHECKPOINT_URL, map_location=torch.device("cpu"), progress=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint["acc1"]


def get_resnet18_model(device: torch.device) -> torch.nn.Module:
    num_classes = 200  # 200 is for Tiny ImageNet, default is 1000 for ImageNet
    model = models.resnet18(weights=None)
    # Update the last FC layer for Tiny ImageNet number of classes.
    model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
    model.to(device)
    return model


def measure_latency(model, example_inputs, num_iters=2000) -> float:
    with torch.no_grad():
        model(example_inputs)
        total_time = 0
        for _ in range(num_iters):
            start_time = time()
            model(example_inputs)
            total_time += time() - start_time
        average_time = (total_time / num_iters) * 1000
    return average_time


def validate(val_loader: torch.utils.data.DataLoader, model: torch.nn.Module, device: torch.device) -> float:
    top1_sum = 0.0

    with torch.no_grad():
        for images, target in track(val_loader, total=len(val_loader), description="Validation:"):
            images = images.to(device)
            target = target.to(device)

            # Compute output.
            output = model(images)

            # Measure accuracy and record loss.
            [acc1] = accuracy(output, target, topk=(1,))
            top1_sum += acc1.item()

        num_samples = len(val_loader)
        top1_avg = top1_sum / num_samples
    return top1_avg


def accuracy(output: torch.Tensor, target: torch.tensor, topk: Tuple[int, ...] = (1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def create_data_loaders():
    dataset_path = download_dataset()

    prepare_tiny_imagenet_200(dataset_path)
    print(f"Successfully downloaded and prepared dataset at: {dataset_path}")

    val_dir = dataset_path / "val"

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    val_dataset = datasets.ImageFolder(
        val_dir,
        transforms.Compose(
            [
                transforms.Resize(IMAGE_SIZE),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    calibration_dataset = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
    )

    return val_loader, calibration_dataset


def prepare_tiny_imagenet_200(dataset_dir: Path):
    # Format validation set the same way as train set is formatted.
    val_data_dir = dataset_dir / "val"
    val_images_dir = val_data_dir / "images"
    if not val_images_dir.exists():
        return

    val_annotations_file = val_data_dir / "val_annotations.txt"
    with open(val_annotations_file, "r") as f:
        val_annotation_data = map(lambda line: line.split("\t")[:2], f.readlines())
    for image_filename, image_label in val_annotation_data:
        from_image_filepath = val_images_dir / image_filename
        to_image_dir = val_data_dir / image_label
        if not to_image_dir.exists():
            to_image_dir.mkdir()
        to_image_filepath = to_image_dir / image_filename
        from_image_filepath.rename(to_image_filepath)
    val_annotations_file.unlink()
    val_images_dir.rmdir()


def main():
    device = torch.device("cpu")
    print(f"Using {device} device")

    ###############################################################################
    # Step 1: Prepare model and dataset
    print(os.linesep + "[Step 1] Prepare model and dataset")

    model = get_resnet18_model(device)
    model, acc1_fp32 = load_checkpoint(model)

    print(f"Accuracy@1 of original FP32 model: {acc1_fp32:.3f}")

    val_loader, calibration_dataset = create_data_loaders()

    def transform_fn(data_item):
        return data_item[0].to(device)

    quantization_dataset = nncf.Dataset(calibration_dataset, transform_fn)

    ###############################################################################
    # Step 2: Quantize model
    print(os.linesep + "[Step 2] Quantize model")

    input_shape = (1, 3, IMAGE_SIZE, IMAGE_SIZE)
    example_input = torch.ones(*input_shape).to(device)

    with disable_patching():
        fx_model = torch.export.export_for_training(model.eval(), args=(example_input,)).module()
        quantized_fx_model = nncf.quantize(fx_model, quantization_dataset)
        quantized_fx_model = torch.compile(quantized_fx_model, backend="openvino")

        acc1_int8 = validate(val_loader, quantized_fx_model, device)

    print(f"Accuracy@1 of INT8 model: {acc1_int8:.3f}")
    print(f"Accuracy diff FP32 - INT8: {acc1_fp32 - acc1_int8:.3f}")

    ###############################################################################
    # Step 3: Run benchmarks
    print(os.linesep + "[Step 3] Run benchmarks")
    print("Benchmark FP32 model compiled with default backend ...")
    with disable_patching():
        compiled_model = torch.compile(model)
        try:
            fp32_latency = measure_latency(compiled_model, example_inputs=example_input)
        except BackendCompilerFailed as exp:
            if not is_windows():
                raise exp
            print(
                "WARNING: Torch Inductor is currently unavailable on Windows. "
                "For more information, visit https://github.com/pytorch/pytorch/issues/135954"
            )
            fp32_latency = float("nan")
    print(f"{fp32_latency:.3f} ms")

    print("Benchmark FP32 model compiled with openvino backend ...")
    with disable_patching():
        compiled_model = torch.compile(model, backend="openvino")
        fp32_ov_latency = measure_latency(compiled_model, example_inputs=example_input)
    print(f"{fp32_ov_latency:.3f} ms")

    print("Benchmark INT8 model compiled with openvino backend ...")
    with disable_patching():
        int8_latency = measure_latency(quantized_fx_model, example_inputs=example_input)
    print(f"{int8_latency:.3f} ms")

    print("[Step 4] Summary:")
    tabular_data = [
        ["default", "FP32", f"{fp32_latency:.3f}", ""],
        ["openvino", "FP32", f"{fp32_ov_latency:.3f}", f"x{fp32_latency / fp32_ov_latency:.3f}"],
        ["openvino", "INT8", f"{int8_latency:.3f}", f"x{fp32_latency / int8_latency:.3f}"],
    ]
    print(create_table(["Backend", "Precision", "Performance (ms)", "Speed up"], tabular_data))
    return acc1_fp32, acc1_int8, fp32_latency, fp32_ov_latency, int8_latency


if __name__ == "__main__":
    main()
