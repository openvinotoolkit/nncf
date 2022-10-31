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

import subprocess
import time
from pathlib import Path
from typing import Any
from typing import Iterable

import openvino.runtime as ov
import torch
import torchvision

import nncf
from examples.experimental.openvino.mobilenet_v2 import utils

# Path to the `mobilenet_v2` directory.
ROOT = Path(__file__).parent.resolve()
# Path to the directory where the original and quantized IR will be saved.
MODEL_DIR = ROOT / 'mobilenet_v2_quantization'
# Path to ImageNet validation dataset.
DATASET_DIR = Path('/ssd/imagenet') #ROOT / 'imagenet'


ie = ov.Core()


def run_example():
    """
    Runs the MobileNetV2 quantization example.
    """
    # Step 1: Instantiate the MobileNetV2 from the PyTorch Hub.
    # For more details, please see the [link](https://pytorch.org/hub/pytorch_vision_mobilenet_v2).
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    model.eval()

    # Step 2: Converts PyTorch model to the OpenVINO model.
    ov_model = convert_torch_to_openvino(model)
    
    # Step 3: Create calibration dataset.
    data_source = create_data_source()

    # Step 4: Apply quantization algorithm.

    # Define the transformation method. This method should take a data item returned
    # per iteration through the `data_source` object and transform it into the model's
    # expected input that can be used for the model inference.
    def transform_fn(data_item):
        images, _ = data_item
        return images.numpy()

    # Wrap framework-specific data source into the `nncf.Dataset` object.
    calibration_dataset = nncf.Dataset(data_source, transform_fn)

    # Quantization of the OpenVINO model. The `quantize` method expects an
    # OpenVINO model as input and returns an OpenVINO model that has been
    # quantized using the calibration dataset.
    quantized_model = nncf.quantize(ov_model, calibration_dataset)

    # Step 5: Save the quantized model.
    ir_qmodel_xml = MODEL_DIR / 'mobilenet_v2_quantized.xml'
    ir_qmodel_bin = MODEL_DIR / 'mobilenet_v2_quantized.bin'
    ov.serialize(quantized_model, str(ir_qmodel_xml), str(ir_qmodel_bin))
    print("The quantized model has been saved in:")
    print(f"XML file: {ir_qmodel_xml}")
    print(f"BIN file: {ir_qmodel_bin}")

    # Step 6: Compare the accuracy of the original and quantized models.
    print('Checking the accuracy of the original model:')
    metric = validation_fn(ov_model, data_source)
    print(f'The original model accuracy@top1: {metric:.3f}')

    print('Checking the accuracy of the quantized model:')
    quantized_metric = validation_fn(quantized_model, data_source)
    print(f'The quantized model accuracy@top1: {quantized_metric:.3f}')

    # Step 7: Compare Performance of the original and quantized models.
    # benchmark_app -m mobilenet_v2_quantization/mobilenet_v2.xml -d CPU -api async
    # benchmark_app -m mobilenet_v2_quantization/mobilenet_v2_quantized.xml -d CPU -api async


def convert_torch_to_openvino(model: torch.nn.Module) -> ov.Model:
    """
    Converts PyTorch MobileNetV2 model to the OpenVINO IR format.
    """
    if not MODEL_DIR.exists():
        MODEL_DIR.mkdir()

    # Export PyTorch model to the ONNX format.
    onnx_model_path = MODEL_DIR / 'mobilenet_v2.onnx'
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy_input, onnx_model_path, verbose=False)

    # Run Model Optimizer to convert ONNX model to OpenVINO IR.
    mo_command = f'mo --framework onnx -m {onnx_model_path} --output_dir {MODEL_DIR}'
    subprocess.call(mo_command, shell=True)

    ir_model_xml = MODEL_DIR / 'mobilenet_v2.xml'
    ir_model_bin = MODEL_DIR / 'mobilenet_v2.bin'
    ov_model = ie.read_model(model=ir_model_xml, weights=ir_model_bin)

    return ov_model


def create_data_source() -> torch.utils.data.DataLoader:
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
                                                 batch_size=1,
                                                 shuffle=False)
    return val_dataloader


# The `validate()` method was taken as is from the pytorch repository.
# You can find it [here](https://github.com/pytorch/examples/blob/main/imagenet/main.py).
# Code regarding CUDA and training was removed.
# Some code was changed and added.

def validation_fn(model: ov.Model, validation_dataset: Iterable[Any]) -> float:
    compiled_model = ie.compile_model(model, device_name='CPU')
    output_layer = compiled_model.output(0)

    def run_validate(loader, base_progress=0):
        PRINT_FREQ = 10000
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i

                # compute output
                input_data = images.numpy()
                output = torch.from_numpy(
                    compiled_model([input_data])[output_layer]
                )

                # measure accuracy
                acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % PRINT_FREQ == 0:
                    progress.display(i + 1)

    batch_time = utils.AverageMeter('Time', ':6.3f', utils.Summary.NONE)
    top1 = utils.AverageMeter('Acc@1', ':6.2f', utils.Summary.AVERAGE)
    top5 = utils.AverageMeter('Acc@5', ':6.2f', utils.Summary.AVERAGE)

    BATCH_SIZE = 1
    NUM_BATCHES = 50000 // BATCH_SIZE
    progress = utils.ProgressMeter(NUM_BATCHES, [batch_time, top1, top5], prefix='Test: ')
    run_validate(validation_dataset)
    progress.display_summary()

    return top1.avg


if __name__ == '__main__':
    run_example()
