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

"""
Reproducible benchmark for structured 2:4 magnitude pruning.

Compares the unpruned resnet18 model against the model pruned with
``PruneMode.STRUCTURED_MAGNITUDE_2_4`` (without fine-tuning):

* top-1 accuracy on the Tiny ImageNet validation set
* PyTorch inference latency (same procedure for both variants)
* OpenVINO export: model file sizes and inference latency of the
  baseline and the stripped pruned model
* verification of the per-group 2:4 pattern (2 zeros and 2 non-zeros
  in every group of four weights)

The benchmark intentionally measures what the pruning mode itself
provides: a fixed sparsity pattern in the weights. Acceleration of the
2:4 pattern is not expected with ordinary dense kernels; it requires an
inference backend and hardware with explicit support for the pattern.

Usage::

    python examples/pruning/torch/resnet18/structured_2_4_benchmark.py \
        --data-dir <path> [--batch-size 64] [--num-warmup 20] [--num-iters 50]

The dataset and the pretrained checkpoint are downloaded automatically
on the first run (the same sources as ``main.py`` in this directory; the
dataset download additionally requires the ``fastdownload`` package from
this example's ``requirements.txt``). Use ``--skip-accuracy`` for a fast
latency-only run without the dataset.
"""

import platform
import time
from argparse import ArgumentParser
from argparse import RawTextHelpFormatter
from pathlib import Path

import numpy as np
import openvino as ov
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import models
from torchvision import transforms

import nncf
from nncf.parameters import PruneMode
from nncf.torch.function_hook.pruning.magnitude.structured_modules import StructuredPruningMask
from nncf.torch.function_hook.wrapper import get_hook_storage

BASE_MODEL_NAME = "resnet18"
IMAGE_SIZE = 64

CHECKPOINT_URL = (
    "https://storage.openvinotoolkit.org/repositories/nncf/openvino_notebook_ckpts/302_resnet18_fp32_v1.pth"
)
DATASET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
DEFAULT_DATA_DIR = Path().home() / ".cache" / "nncf" / "datasets"


def get_argument_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Benchmark of structured 2:4 magnitude pruning on resnet18 / Tiny ImageNet.",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Dataset cache directory.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for latency measurement.")
    parser.add_argument("--num-warmup", type=int, default=20, help="Number of warmup iterations before timing.")
    parser.add_argument("--num-iters", type=int, default=50, help="Number of measured iterations.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--skip-accuracy", action="store_true", help="Skip dataset download and accuracy evaluation, time only."
    )
    return parser


def download_dataset(data_dir: Path) -> Path:
    # Imported lazily so that the latency-only mode (--skip-accuracy) does not require it.
    from fastdownload import FastDownload

    downloader = FastDownload(base=data_dir.resolve(), archive="downloaded", data="extracted")
    return downloader.get(DATASET_URL)


def prepare_tiny_imagenet_200(dataset_dir: Path) -> None:
    # Format validation set the same way as the train set is formatted.
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


def get_model(device: torch.device) -> nn.Module:
    model = models.resnet18(weights=None)
    # Update the last FC layer for the number of Tiny ImageNet classes.
    model.fc = nn.Linear(in_features=512, out_features=200, bias=True)
    checkpoint = torch.hub.load_state_dict_from_url(CHECKPOINT_URL, map_location="cpu", progress=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    return model


def get_val_loader(dataset_dir: Path, batch_size: int) -> DataLoader:
    val_dir = dataset_dir / "val"
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    val_dataset = datasets.ImageFolder(
        val_dir, transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.ToTensor(), normalize])
    )
    return DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)


@torch.no_grad()
def validate(val_loader: DataLoader, model: nn.Module, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, target in val_loader:
        images = images.to(device)
        target = target.to(device)
        output = model(images)
        preds = output.max(1)[1]
        correct += preds.eq(target).sum().item()
        total += target.size(0)
    return 100.0 * correct / total


def check_two_four_pattern(model: nn.Module) -> None:
    """
    Verifies that every complete group of four consecutive weights within one
    output channel contains exactly two retained values, for each structured
    pruning mask in the model.
    """
    hook_storage = get_hook_storage(model)
    structured_hooks = [hook for _, hook in hook_storage.named_hooks() if isinstance(hook, StructuredPruningMask)]
    if not structured_hooks:
        msg = "No structured pruning masks found in the model"
        raise RuntimeError(msg)
    for hook in structured_hooks:
        mask = hook.binary_mask
        num_out_channels = mask.shape[0]
        grouped = mask.reshape(num_out_channels, -1, 4)
        group_sums = grouped.sum(dim=-1)
        if not torch.all(group_sums == 2):
            msg = f"2:4 pattern violation: group retention sums {group_sums.unique().tolist()}"
            raise RuntimeError(msg)


def count_zero_fraction(model: nn.Module) -> float:
    zeros = 0
    total = 0
    hook_storage = get_hook_storage(model)
    for _, hook in hook_storage.named_hooks():
        if isinstance(hook, StructuredPruningMask):
            mask = hook.binary_mask
            zeros += int((mask == 0).sum())
            total += mask.numel()
    return 100.0 * zeros / total


def measure_latency(model: nn.Module, batch_size: int, num_warmup: int, num_iters: int) -> dict:
    """
    Measures average forward-pass latency per iteration.

    Both models are measured with the same procedure: eval mode, gradients
    disabled, one shared random input tensor, num_warmup unmeasured
    iterations followed by num_iters measured iterations timed with
    time.perf_counter(). For CUDA devices torch.cuda.synchronize() is called
    before starting and stopping the timer.
    """
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    input_tensor = torch.randn(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)

    with torch.no_grad():
        for _ in range(num_warmup):
            model(input_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iters):
            model(input_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    if was_training:
        model.train()
    per_iter_ms = elapsed / num_iters * 1e3
    return {"ms_per_iter": per_iter_ms, "fps": batch_size * num_iters / elapsed}


def print_environment() -> None:
    print("Environment:")
    print(f"  torch:       {torch.__version__}")
    print(f"  nncf:        {nncf.__version__}")
    print(f"  openvino:    {ov.__version__}")
    print(f"  device:      {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"  cpu:         {platform.processor() or platform.machine()}")
    print(f"  threads:     {torch.get_num_threads()}")
    print(f"  os:          {platform.system()} {platform.release()}")


def export_to_openvino(model: nn.Module, input_shape: tuple, ir_path: Path) -> float:
    model.to("cpu")
    ov_model = ov.convert_model(model, example_input=torch.zeros(input_shape), input=input_shape)
    ov.save_model(ov_model, str(ir_path), compress_to_fp16=False)
    # An OpenVINO IR consists of the .xml structure file and the .bin weights file.
    return ir_path.stat().st_size + ir_path.with_suffix(".bin").stat().st_size


def main() -> None:
    args = get_argument_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Structured 2:4 magnitude pruning benchmark ({BASE_MODEL_NAME}, Tiny ImageNet)")
    print_environment()
    print(f"  batch size:  {args.batch_size}")
    print(f"  warmup:      {args.num_warmup} iterations")
    print(f"  measured:    {args.num_iters} iterations")

    example_input = torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    print("\n[Step 1] Load pretrained model")
    baseline_model = get_model(device)

    print("\n[Step 2] Baseline accuracy and latency")
    # nncf.prune() wraps the passed model in place, so the baseline must be
    # fully measured before pruning is applied.
    baseline_latency = measure_latency(baseline_model, args.batch_size, args.num_warmup, args.num_iters)
    print(
        f"  baseline             {baseline_latency['ms_per_iter']:8.2f} ms/iter  ({baseline_latency['fps']:8.1f} fps)"
    )

    baseline_acc = None
    if not args.skip_accuracy:
        dataset_dir = download_dataset(args.data_dir)
        prepare_tiny_imagenet_200(dataset_dir)
        val_loader = get_val_loader(dataset_dir, args.batch_size)
        baseline_acc = validate(val_loader, baseline_model, device)
        print(f"Baseline top-1 accuracy: {baseline_acc:.3f}")

    print("\n[Step 3] Apply structured 2:4 pruning")
    # conv1 has 3 * 7 * 7 = 147 weights per output channel, which is not divisible
    # by the group size 4 of the 2:4 pattern, so it is excluded from pruning.
    pruned_model = nncf.prune(
        baseline_model,
        mode=PruneMode.STRUCTURED_MAGNITUDE_2_4,
        ignored_scope=nncf.IgnoredScope(names=["conv1/conv2d/0"]),
        examples_inputs=example_input.to(device),
    )
    check_two_four_pattern(pruned_model)
    sparsity = count_zero_fraction(pruned_model)
    print(f"2:4 pattern verified for all pruned weights, measured sparsity: {sparsity:.2f}%")

    if not args.skip_accuracy:
        pruned_acc = validate(val_loader, pruned_model, device)
        print(f"2:4 pruned top-1 accuracy (before fine-tuning): {pruned_acc:.3f}")

    print("\n[Step 4] Strip pruning masks into the weights")
    # The default strip_format returns an independent copy, so the pruned model
    # with pruning hooks stays available for statistics and comparison.
    stripped_model = nncf.strip(pruned_model, strip_format=nncf.StripFormat.IN_PLACE)
    if not args.skip_accuracy:
        stripped_acc = validate(val_loader, stripped_model, device)
        print(f"Stripped pruned top-1 accuracy: {stripped_acc:.3f}")

    print("\n[Step 5] PyTorch latency of the pruned model")
    pruned_latency = measure_latency(pruned_model, args.batch_size, args.num_warmup, args.num_iters)
    stripped_latency = measure_latency(stripped_model, args.batch_size, args.num_warmup, args.num_iters)
    # The baseline is measured a second time to quantify the run-to-run drift of the
    # measurement (CPU frequency and thermal effects) between the first and the last run.
    baseline_latency_recheck = measure_latency(baseline_model, args.batch_size, args.num_warmup, args.num_iters)
    for name, latency in (
        ("pruned (hooks)", pruned_latency),
        ("pruned (stripped)", stripped_latency),
        ("baseline (recheck)", baseline_latency_recheck),
    ):
        print(f"  {name:20s} {latency['ms_per_iter']:8.2f} ms/iter  ({latency['fps']:8.1f} fps)")

    print("\n[Step 6] OpenVINO export and latency")
    ir_dir = Path(__file__).parent.resolve() / "benchmark_artifacts"
    ir_dir.mkdir(exist_ok=True)
    baseline_ir = ir_dir / f"{BASE_MODEL_NAME}_baseline.xml"
    pruned_ir = ir_dir / f"{BASE_MODEL_NAME}_2_4_pruned.xml"

    # Export with the static latency input shape so that the OpenVINO latency
    # refers to the same batch size as the PyTorch latency above.
    latency_input_shape = (args.batch_size, 3, IMAGE_SIZE, IMAGE_SIZE)
    baseline_size = export_to_openvino(baseline_model, latency_input_shape, baseline_ir)
    pruned_size = export_to_openvino(stripped_model, latency_input_shape, pruned_ir)

    core = ov.Core()
    ov_latencies = {}
    for name, ir in (("baseline", baseline_ir), ("2:4 pruned", pruned_ir)):
        compiled = core.compile_model(str(ir), "CPU")
        infer_request = compiled.create_infer_request()
        input_data = np.random.rand(args.batch_size, 3, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)
        for _ in range(args.num_warmup):
            infer_request.infer(input_data)
        start = time.perf_counter()
        for _ in range(args.num_iters):
            # InferRequest.infer() is synchronous, no additional synchronization is required.
            infer_request.infer(input_data)
        elapsed = time.perf_counter() - start
        ms = elapsed / args.num_iters * 1e3
        ov_latencies[name] = ms
        print(f"  {name:20s} {ms:8.2f} ms/iter  ({args.batch_size * args.num_iters / elapsed:8.1f} fps)")

    print("\nModel file sizes (OpenVINO IR, FP32):")
    print(f"  baseline:   {baseline_size / 1e6:7.2f} MB")
    print(f"  2:4 pruned: {pruned_size / 1e6:7.2f} MB")

    print("\nSummary")
    if not args.skip_accuracy:
        print(
            f"  Top-1 accuracy: baseline {baseline_acc:.3f} -> pruned {pruned_acc:.3f} "
            f"(change {pruned_acc - baseline_acc:+.3f}, no fine-tuning)"
        )
        print(f"  Top-1 accuracy after strip: {stripped_acc:.3f}")
    speedup_torch = baseline_latency_recheck["ms_per_iter"] / stripped_latency["ms_per_iter"]
    speedup_ov = ov_latencies["baseline"] / ov_latencies["2:4 pruned"]
    print(
        f"  PyTorch latency ratio (baseline recheck / pruned-stripped): {speedup_torch:.3f}x "
        f"(initial baseline run: {baseline_latency['ms_per_iter']:.2f} ms/iter)"
    )
    print(f"  OpenVINO CPU latency ratio (baseline / pruned):     {speedup_ov:.3f}x")
    print(
        "\nNote: the pruned model carries a fixed 2:4 sparsity pattern in its weights. Execution with\n"
        "ordinary dense kernels (as in this benchmark) is not expected to be faster; acceleration\n"
        "requires an inference backend and hardware with explicit support for 2:4 structured sparsity."
    )


if __name__ == "__main__":
    main()
