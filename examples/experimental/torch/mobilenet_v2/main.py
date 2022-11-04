"""
 Copyright (c) 2022 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import os
import subprocess
import time
from pathlib import Path

import nncf
import torch
import torchvision
from openvino.runtime import Core

from examples.experimental.torch.mobilenet_v2 import utils

# Path to the `mobilenet_v2` directory.
ROOT = Path(__file__).parent.resolve()
# Path to the directory where the original and quantized models will be saved.
MODEL_DIR = ROOT / 'mobilenet_v2_quantization'
# Path to ImageNet validation dataset.
DATASET_DIR = ROOT / 'imagenet'
# Batch size
BATCH_SIZE = 125


def run_example():
    """
    Runs the MobileNetV2 quantization example.
    """
    # Step 1: Instantiate the MobileNetV2 from the PyTorch Hub.
    # For more details, please see the [link](https://pytorch.org/hub/pytorch_vision_mobilenet_v2).
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    # Step 2: Create dataset.
    data_source = create_data_source(BATCH_SIZE)

    # Step 3: Apply quantization algorithm.

    # Define the transformation method. This method should take a data item returned
    # per iteration through the `data_source` object and transform it into the model's
    # expected input that can be used for the model inference.
    def transform_fn(data_item):
        images, _ = data_item
        return images

    # Wrap framework-specific data source into the `nncf.Dataset` object.
    calibration_dataset = nncf.Dataset(data_source, transform_fn)

    # Quantization of the PyTorch model.
    quantized_model = nncf.quantize(model, calibration_dataset)

    # Step 4: Export PyTorch model to ONNX format.
    quantized_model.cpu()

    if not MODEL_DIR.exists():
        os.makedirs(MODEL_DIR)

    onnx_quantized_model_path = MODEL_DIR / 'mobilenet_v2.onnx'
    dummy_input = torch.randn(BATCH_SIZE, 3, 224, 224)
    torch.onnx.export(quantized_model, dummy_input, onnx_quantized_model_path, verbose=False)
    print(f'The quantized model is exported to: {onnx_quantized_model_path}')

    # Step 5: Run OpenVINO Model Optimizer to convert ONNX model to OpenVINO IR.
    mo_command = f'mo --framework onnx -m {onnx_quantized_model_path} --output_dir {MODEL_DIR}'
    subprocess.call(mo_command, shell=True)

    # Step 6: Compare the accuracy of the original and quantized models.
    print('Checking the accuracy of the original model:')
    result = validate_pytorch_model(model, data_source)
    print(f'The original model accuracy@top1: {result:.3f}')

    print('Checking the accuracy of the quantized model:')
    ie = Core()
    ir_model_xml = MODEL_DIR / 'mobilenet_v2.xml'
    ir_model_bin = MODEL_DIR / 'mobilenet_v2.bin'
    ir_quantized_model = ie.read_model(model=ir_model_xml, weights=ir_model_bin)
    quantized_compiled_model = ie.compile_model(ir_quantized_model, device_name='CPU')
    quantized_result = validate_openvino_model(quantized_compiled_model, data_source)
    print(f'The quantized model accuracy@top1: {quantized_result:.3f}')


def create_data_source(batch_size: int = 1) -> torch.utils.data.DataLoader:
    """
    Creates validation ImageNet data loader.
    """
    val_dir = DATASET_DIR / 'val'
    # Transformations were taken from [here](https://pytorch.org/hub/pytorch_vision_mobilenet_v2).
    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_dataset = torchvision.datasets.ImageFolder(val_dir, preprocess)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False)
    return val_dataloader


def validate_pytorch_model(model, val_loader):
    def create_forward_fn(device):
        def forward(model, images):
            images = images.to(device)
            return model(images)
        return forward

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device('cpu')

    forward_fn = create_forward_fn(device)
    return validate(model, val_loader, forward_fn)


def validate_openvino_model(model, val_loader):
    def create_forward_fn(output_layer):
        def forward(model, images):
            input_data = images.numpy()
            return torch.from_numpy(model([input_data])[output_layer])
        return forward

    output_layer = next(iter(model.outputs))
    forward_fn = create_forward_fn(output_layer)

    return validate(model, val_loader, forward_fn)


def validate(model, val_loader, forward_fn):
    def run_validate(loader, base_progress=0, print_freq=10000):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i

                output = forward_fn(model, images)

                # measure accuracy
                acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % print_freq == 0:
                    progress.display(i + 1)

    batch_time = utils.AverageMeter('Time', ':6.3f', utils.Summary.NONE)
    top1 = utils.AverageMeter('Acc@1', ':6.2f', utils.Summary.AVERAGE)
    top5 = utils.AverageMeter('Acc@5', ':6.2f', utils.Summary.AVERAGE)

    progress = utils.ProgressMeter(
        len(val_loader), [batch_time, top1, top5], prefix='Test: '
    )
    run_validate(val_loader, print_freq=10000 // BATCH_SIZE)
    progress.display_summary()

    return top1.avg


if __name__ == '__main__':
    run_example()
