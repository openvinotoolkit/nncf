import re

import torch

from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import timm

from tests.shared.command import Command

def create_timm_model(name):
    model = timm.create_model(
        name, num_classes=1000, in_chans=3, pretrained=True, checkpoint_path=""
    )
    return model


def get_model_transform(config):
    normalize = transforms.Normalize(mean=config["mean"], std=config["std"])
    input_size = config["input_size"]
    resize_size = tuple(int(x / config["crop_pct"]) for x in input_size[-2:])

    RESIZE_MODE_MAP = {
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
        "nearest": InterpolationMode.NEAREST,
    }

    transform = transforms.Compose(
        [
            transforms.Resize(
                resize_size, interpolation=RESIZE_MODE_MAP[config["interpolation"]]
            ),
            transforms.CenterCrop(input_size[-2:]),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return transform


def get_torch_dataloader(folder, transform, batch_size=1):
    val_dataset = datasets.ImageFolder(root=folder, transform=transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, num_workers=4, shuffle=False
    )
    return val_loader


def export_to_onnx(model, save_path, data_sample):
    print(f"Export to ONNX: {save_path}")
    torch.onnx.export(
        model,
        data_sample,
        save_path,
        export_params=True,
        opset_version=14,    # fix offset to 14 
        do_constant_folding=False,
    )


def export_to_ir(model_path, save_path, model_name):
    print(f"Export to IR: {save_path}")
    runner = Command(f"mo -m {model_path} -o {save_path} -n {model_name}")
    runner.run()


def run_benchmark(model_path):
    runner = Command(f"benchmark_app -m {model_path} -d CPU -niter 300")
    runner.run()
    cmd_output = " ".join(runner.output)

    match = re.search(r"Throughput\: (.+?) FPS", cmd_output)
    if match is not None:
        fps = match.group(1)
        return float(fps), cmd_output

    return None, cmd_output