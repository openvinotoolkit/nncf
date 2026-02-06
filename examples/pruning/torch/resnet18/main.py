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

import os
import warnings
from argparse import ArgumentParser
from argparse import RawTextHelpFormatter
from pathlib import Path

import openvino as ov
import pooch
import torch
from rich.progress import track
from torch import nn
from torch.jit import TracerWarning
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.models import resnet18

import nncf
from nncf.parameters import PruneMode
from nncf.torch.function_hook.pruning.magnitude.schedulers import MultiStepMagnitudePruningScheduler
from nncf.torch.function_hook.pruning.rb.losses import RBLoss
from nncf.torch.function_hook.pruning.rb.schedulers import MultiStepRBPruningScheduler

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
DATASET_PATH = Path().home() / ".cache" / "nncf" / "datasets"
EXTRACTED_DATASET_PATH = DATASET_PATH / "extracted"


def get_argument_parser() -> ArgumentParser:
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["mag", "mag_bn", "rb"],
        default="mag",
        help=(
            "Pruning mode to use. Choices are:\n"
            " - mag: Magnitude-based pruning with fine-tuning (default).\n"
            " - mag_bn: Magnitude-based pruning with BatchNorm adaptation without fine-tuning.\n"
            " - rb: Regularization-based pruning with fine-tuning.\n"
        ),
    )
    return parser


def download_dataset() -> Path:
    files = pooch.retrieve(
        url=DATASET_URL,
        path=DATASET_PATH / "downloaded",
        processor=pooch.Untar(extract_dir=EXTRACTED_DATASET_PATH),
    )
    # pooch.Untar returns a list of extracted files
    dataset_root = EXTRACTED_DATASET_PATH / Path(files[0]).relative_to(EXTRACTED_DATASET_PATH).parts[0]
    return dataset_root


def load_checkpoint(model: nn.Module) -> tuple[nn.Module, float]:
    checkpoint = torch.hub.load_state_dict_from_url(CHECKPOINT_URL, map_location=torch.device("cpu"), progress=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint["acc1"]


def get_resnet18_model(device: torch.device) -> nn.Module:
    model = resnet18(weights=None)
    # Update the last FC layer for Tiny ImageNet number of classes.
    # 200 is for Tiny ImageNet, default is 1000 for ImageNet
    model.fc = nn.Linear(in_features=512, out_features=200, bias=True)
    model.to(device)
    return model


def train_epoch(
    train_loader: DataLoader,
    model: nn.Module,
    rb_loss: RBLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    model.train()
    criterion = nn.CrossEntropyLoss().to(device)

    for images, target in track(train_loader, total=len(train_loader), description="Fine tuning:"):
        images = images.to(device)
        target = target.to(device)

        # Compute output.
        output = model(images)
        loss = criterion(output, target)
        if rb_loss is not None:
            loss += rb_loss()
        # Compute gradient and do opt step.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


@torch.no_grad()
def validate(val_loader: torch.utils.data.DataLoader, model: torch.nn.Module, device: torch.device) -> float:
    model.eval()

    correct = 0
    total = 0

    for images, target in track(val_loader, total=len(val_loader), description="Validation:"):
        images = images.to(device)
        target = target.to(device)

        output = model(images)

        _, preds = output.max(1)
        correct += preds.eq(target).sum().item()
        total += target.size(0)

    accuracy1 = 100.0 * correct / total
    return accuracy1


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
            [transforms.Resize(IMAGE_SIZE), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]
        ),
    )
    val_dataset = datasets.ImageFolder(
        val_dir,
        transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.ToTensor(), normalize]),
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


def main() -> float:
    args = get_argument_parser().parse_args()
    pruning_mode = args.mode

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    ###############################################################################
    # Step 1: Prepare model and dataset
    print(os.linesep + "[Step 1] Prepare model and dataset")

    model = get_resnet18_model(device)
    model, acc1_fp32 = load_checkpoint(model)

    print(f"Accuracy@1 of original FP32 model: {acc1_fp32:.2f}")

    train_loader, val_loader = create_data_loaders()
    example_input = torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)

    ###############################################################################
    # Step 2: Prune model
    print(os.linesep + "[Step 2] Prune model and specify training parameters")

    if pruning_mode == "mag_bn":
        pruned_model = nncf.prune(
            model,
            mode=PruneMode.UNSTRUCTURED_MAGNITUDE_GLOBAL,
            ratio=0.6,
            examples_inputs=example_input,
        )
    elif pruning_mode == "mag":
        pruned_model = nncf.prune(
            model,
            mode=PruneMode.UNSTRUCTURED_MAGNITUDE_GLOBAL,
            ratio=0.7,
            examples_inputs=example_input,
        )
        num_epochs = 2
        rb_loss = None
        scheduler = MultiStepMagnitudePruningScheduler(
            model=model, mode=PruneMode.UNSTRUCTURED_MAGNITUDE_GLOBAL, steps={0: 0.5, 1: 0.7}
        )
        optimizer = torch.optim.Adam(pruned_model.parameters(), lr=1e-5)
    elif pruning_mode == "rb":
        pruned_model = nncf.prune(
            model,
            mode=PruneMode.UNSTRUCTURED_REGULARIZATION_BASED,
            examples_inputs=example_input,
        )
        num_epochs = 30
        rb_loss = RBLoss(pruned_model, target_ratio=0.7, p=0.1).to(device)
        scheduler = MultiStepRBPruningScheduler(rb_loss, steps={0: 0.3, 5: 0.5, 10: 0.7})

        # Set higher lr for mask parameters to achieve the target pruning ratio faster
        mask_params = [p for n, p in pruned_model.named_parameters() if "mask" in n]
        model_params = [p for n, p in pruned_model.named_parameters() if "mask" not in n]
        optimizer = torch.optim.Adam(
            [
                {"params": model_params, "lr": 1e-5},
                {"params": mask_params, "lr": 1e-2, "weight_decay": 0.0},
            ]
        )
    else:
        msg = f"Unsupported pruning mode: {pruning_mode}, please choose from ['mag', 'mag_bn', 'rb']"
        raise ValueError(msg)

    ###############################################################################
    # Step 3: Fine tune
    print(os.linesep + "[Step 3] Fine tune with multi step pruning ratio scheduler")

    if pruning_mode == "mag_bn":
        acc1_before = validate(val_loader, pruned_model, device)
        print(f"Accuracy@1 of pruned model before BatchNorm adaptation: {acc1_before:.3f}")

        def transform_fn(batch: tuple[torch.Tensor, int]) -> torch.Tensor:
            inputs, _ = batch
            return inputs.to(device=device)

        calibration_dataset = nncf.Dataset(train_loader, transform_func=transform_fn)

        pruned_model = nncf.batch_norm_adaptation(
            pruned_model,
            calibration_dataset=calibration_dataset,
            num_iterations=200,
        )

        acc1 = validate(val_loader, pruned_model, device)
        print(f"Accuracy@1 of pruned model after BatchNorm adaptation: {acc1:.3f}")
    else:
        for epoch in range(num_epochs):
            print(os.linesep + f"Train epoch: {epoch}")
            scheduler.step()
            train_epoch(train_loader, pruned_model, rb_loss, optimizer, device=device)

            acc1 = validate(val_loader, pruned_model, device)
            print(f"Current pruning ratio: {scheduler.current_ratio:.3f}")
            print(f"Accuracy@1 of pruned model after {epoch} epoch: {acc1:.3f}")

    ###############################################################################
    # Step 4: Print per tensor pruning statistics
    print(os.linesep + "[Step 4] Pruning statistics")

    pruning_stat = nncf.pruning_statistic(pruned_model)
    print(pruning_stat)

    ###############################################################################
    # Step 5: Export models
    print(os.linesep + "[Step 5] Export models")
    ir_path = ROOT / f"{BASE_MODEL_NAME}_pruned.xml"
    ov_model = ov.convert_model(pruned_model.cpu(), example_input=example_input.cpu(), input=tuple(example_input.shape))
    ov.save_model(ov_model, ir_path, compress_to_fp16=False)
    print(f"Pruned model path: {ir_path}")
    return acc1


if __name__ == "__main__":
    main()
