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
import warnings
from pathlib import Path

import openvino as ov
import torch
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from fastdownload import FastDownload
from rich.progress import track
from torch import nn
from torch.jit import TracerWarning
from torch.utils.data import DataLoader

import nncf
import nncf.parameters
import nncf.torch
import nncf.torch.function_hook
import nncf.torch.function_hook.prune
import nncf.torch.function_hook.prune.prune_model
from nncf.parameters import PruneMode
from nncf.torch.function_hook.prune.magnitude.schedulers import MultiStepMagnitudePruningScheduler

warnings.filterwarnings("ignore", category=TracerWarning)
warnings.filterwarnings("ignore", category=UserWarning)

BASE_MODEL_NAME = "resnet18"
IMAGE_SIZE = 64
BATCH_SIZE = 128
TRAINING_EPOCHS = 2


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


def load_checkpoint(model: nn.Module) -> tuple[nn.Module, float]:
    checkpoint = torch.hub.load_state_dict_from_url(CHECKPOINT_URL, map_location=torch.device("cpu"), progress=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint["acc1"]


def get_resnet18_model(device: torch.device) -> nn.Module:
    num_classes = 200  # 200 is for Tiny ImageNet, default is 1000 for ImageNet
    model = models.resnet18(weights=None)
    # Update the last FC layer for Tiny ImageNet number of classes.
    model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
    model.to(device)
    return model


def train_epoch(
    train_loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    # Switch to train mode.
    model.train()

    for images, target in track(train_loader, total=len(train_loader), description="Fine tuning:"):
        images = images.to(device)
        target = target.to(device)

        # Compute output.
        output = model(images)
        loss = criterion(output, target)

        # Compute gradient and do opt step.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate(val_loader: DataLoader, model: nn.Module, device: torch.device) -> float:
    top1_sum = 0.0

    # Switch to evaluate mode.
    model.eval()

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


def accuracy(output: torch.Tensor, target: torch.tensor, topk: tuple[int, ...] = (1,)):
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


def create_data_loaders() -> tuple[DataLoader, DataLoader]:
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

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, sampler=None
    )

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader


def prepare_tiny_imagenet_200(dataset_dir: Path) -> None:
    # Format validation set the same way as train set is formatted.
    val_data_dir = dataset_dir / "val"
    val_images_dir = val_data_dir / "images"
    if not val_images_dir.exists():
        return

    val_annotations_file = val_data_dir / "val_annotations.txt"
    with open(val_annotations_file) as f:
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
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    ###############################################################################
    # Step 1: Prepare model and dataset
    print(os.linesep + "[Step 1] Prepare model and dataset")

    model = get_resnet18_model(device)
    model, acc1_fp32 = load_checkpoint(model)

    print(f"Accuracy@1 of original FP32 model: {acc1_fp32}")

    train_loader, val_loader = create_data_loaders()
    example_input = torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)

    ###############################################################################
    # Step 2: Prune model
    print(os.linesep + "[Step 2] Prune model")

    # Unstructured pruning with 70% sparsity ratio
    pruned_model = nncf.prune(
        model,
        mode=PruneMode.UNSTRUCTURED_MAGNITUDE_GLOBAL,
        ratio=0.7,
        ignored_scope=nncf.IgnoredScope(),
        examples_inputs=example_input,
    )

    acc1_init = validate(val_loader, pruned_model, device)

    print(f"Accuracy@1 of pruned model with 0.7 pruning ratio without fine-tuning: {acc1_init:.3f}")

    ###############################################################################
    # Step 3: Fine tune with multi step sparsity scheduler
    print(os.linesep + "[Step 3] Fine tune with multi step sparsity scheduler")

    # Define loss function (criterion) and optimizer.
    criterion = nn.CrossEntropyLoss().to(device)
    compression_lr = 1e-5
    optimizer = torch.optim.Adam(pruned_model.parameters(), lr=compression_lr)

    # Create prune scheduler with multi steps strategy
    pruning_scheduler = MultiStepMagnitudePruningScheduler(
        pruned_model, mode=PruneMode.UNSTRUCTURED_MAGNITUDE_GLOBAL, steps={0: 0.6, 1: 0.7}
    )

    for epoch in range(2):
        print(os.linesep + f"Train epoch: {epoch}")

        pruning_scheduler.step()

        train_epoch(train_loader, pruned_model, criterion, optimizer, device=device)
        acc1 = validate(val_loader, pruned_model, device)
        # Show statistics of pruning
        print(f"Accuracy@1 of pruned model after {epoch} epoch ratio {pruning_scheduler.current_ratio}: {acc1:.3f}")

    ###############################################################################
    # Step 4: Export models
    ir_path = ROOT / f"{BASE_MODEL_NAME}_pruned.xml"
    ov_model = ov.convert_model(pruned_model.cpu(), example_input=example_input.cpu(), input=tuple(example_input.shape))
    ov.save_model(ov_model, ir_path, compress_to_fp16=False)
    print(f"Pruned model path: {ir_path}")
    return acc1


if __name__ == "__main__":
    main()
