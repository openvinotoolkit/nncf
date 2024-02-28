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
import re
import subprocess
import warnings
from pathlib import Path
from typing import List, Tuple

import openvino as ov
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
from torch.jit import TracerWarning

import nncf
from nncf.common.logging.track_progress import track

warnings.filterwarnings("ignore", category=TracerWarning)
warnings.filterwarnings("ignore", category=UserWarning)


BASE_MODEL_NAME = "resnet18"
IMAGE_SIZE = 64
BATCH_SIZE = 128


ROOT = Path(__file__).parent.resolve()
CHECKPOINT_URL = (
    "https://storage.openvinotoolkit.org/repositories/nncf/openvino_notebook_ckpts/302_resnet18_fp32_v1.pth"
)
DATASET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
DATASET_PATH = "~/.cache/nncf/datasets"


def download_dataset() -> Path:
    downloader = FastDownload(base=DATASET_PATH, archive="downloaded", data="extracted")
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


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    init_lr = 1e-4
    model = get_resnet18_model(device)

    # Define loss function (criterion) and optimizer.
    criterion = nn.CrossEntropyLoss().to(device)

    model, acc1_fp32 = load_checkpoint(model)
    print(f"Accuracy of pre-trained FP32 model: {acc1_fp32:.3f}")

    train_loader, val_loader, calibration_dataset = create_data_loaders()

    def transform_fn(data_item):
        return data_item[0].to(device)

    quantization_dataset = nncf.Dataset(calibration_dataset, transform_fn)

    quantized_model = nncf.quantize(model, quantization_dataset)
    acc1_int8_init = validate(val_loader, quantized_model, device)
    print(f"Accuracy of initialized INT8 model: {acc1_int8_init:.3f}")
    print(f"Accuracy drop of initialized INT8 model over pre-trained FP32 model: {acc1_fp32 - acc1_int8_init:.3f}")

    compression_lr = init_lr / 10
    optimizer = torch.optim.Adam(quantized_model.parameters(), lr=compression_lr)

    # Train for one epoch with NNCF.
    train(train_loader, quantized_model, criterion, optimizer, epoch=0, device=device)

    # Evaluate on validation set after Quantization-Aware Training (QAT case).
    acc1_int8 = validate(val_loader, quantized_model, device)

    print(f"Accuracy of tuned INT8 model: {acc1_int8:.3f}")
    print(f"Accuracy drop of tuned INT8 model over pre-trained FP32 model: {acc1_fp32 - acc1_int8:.3f}")

    input_shape = (1, 3, IMAGE_SIZE, IMAGE_SIZE)
    example_input = torch.randn(*input_shape).cpu()

    # Export FP32 model to OpenVINO™ IR
    fp32_ir_path = f"{ROOT}/{BASE_MODEL_NAME}_fp32.xml"
    ov_model = ov.convert_model(model.cpu(), example_input=example_input, input=input_shape)
    ov.save_model(ov_model, fp32_ir_path, compress_to_fp16=False)
    print(f"FP32 model was exported to {fp32_ir_path}.")

    # Export INT8 model to OpenVINO™ IR
    int8_ir_path = f"{ROOT}/{BASE_MODEL_NAME}_int8.xml"
    ov_model = ov.convert_model(quantized_model.cpu(), example_input=example_input, input=input_shape)
    ov.save_model(ov_model, int8_ir_path)
    print(f"INT8 model exported to {int8_ir_path}.")

    print("Benchmark FP32 model (IR)")
    fp32_fps = run_benchmark(fp32_ir_path, shape=input_shape, verbose=True)

    print("Benchmark INT8 model (IR)")
    int8_fps = run_benchmark(int8_ir_path, shape=input_shape, verbose=True)

    print(f"Performance speed up (throughput mode): {int8_fps / fp32_fps:.3f}")

    fp32_model_size = get_model_size(fp32_ir_path, verbose=True)
    int8_model_size = get_model_size(int8_ir_path, verbose=True)
    print(f"Model compression rate: {fp32_model_size / int8_model_size:.3f}")
    return acc1_fp32, acc1_int8_init, acc1_int8, fp32_fps, int8_fps, fp32_model_size, int8_model_size


def train(
    train_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    device: torch.device,
):
    # Switch to train mode.
    model.train()

    for images, target in track(train_loader, total=len(train_loader), description=f"Training: epoch {epoch}"):
        images = images.to(device)
        target = target.to(device)

        # Compute output.
        output = model(images)
        loss = criterion(output, target)

        # Compute gradient and do opt step.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate(val_loader: torch.utils.data.DataLoader, model: torch.nn.Module, device: torch.device) -> float:
    top1_sum = 0.0
    top5_sum = 0.0

    # Switch to evaluate mode.
    model.eval()

    with torch.no_grad():
        for images, target in track(val_loader, total=len(val_loader), description="Validation:"):
            images = images.to(device)
            target = target.to(device)

            # Compute output.
            output = model(images)

            # Measure accuracy and record loss.
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1_sum += acc1.item()
            top5_sum += acc5.item()

        num_samples = len(val_loader)
        top1_avg = top1_sum / num_samples
        top5_avg = top5_sum / num_samples
        print(f" * Acc@1 {top1_avg:.3f} Acc@5 {top5_avg:.3f}")
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

    train_dir = dataset_path / "train"
    val_dir = dataset_path / "val"

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose(
            [
                transforms.Resize(IMAGE_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
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

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, sampler=None
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True
    )

    # Creating separate dataloader with batch size = 1
    # as dataloaders with batches > 1 are not supported yet.
    calibration_dataset = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
    )

    return train_loader, val_loader, calibration_dataset


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


def run_benchmark(model_path: str, shape: List[int], verbose: bool = True) -> float:
    command = f"benchmark_app -m {model_path} -d CPU -api async -t 15"
    command += f' -shape "[{",".join(str(x) for x in shape)}]"'
    cmd_output = subprocess.check_output(command, shell=True)  # nosec
    if verbose:
        print(*str(cmd_output).split("\\n")[-9:-1], sep="\n")
    match = re.search(r"Throughput\: (.+?) FPS", str(cmd_output))
    return float(match.group(1))


def get_model_size(ir_path: str, m_type: str = "Mb", verbose: bool = True) -> float:
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


if __name__ == "__main__":
    main()
