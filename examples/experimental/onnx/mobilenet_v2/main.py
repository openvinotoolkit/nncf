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
import time
from pathlib import Path
from typing import Dict, List

import nncf
import onnx
import onnxruntime as rt
import torch
import torchvision

from examples.experimental.onnx.mobilenet_v2 import utils
from nncf.experimental.onnx.tensor import ONNXNNCFTensor

# Path to the `mobilenet_v2` directory.
ROOT = Path(__file__).parent.resolve()
# Path to the directory where the original and quantized models will be saved.
MODEL_DIR = ROOT / 'mobilenet_v2_quantization'
# Path to ImageNet validation dataset.
DATASET_DIR = ROOT / 'imagenet'


def run_example():
    """
    Runs the MobileNetV2 quantization example.
    """
    # Step 1: Instantiate the MobileNetV2 from the PyTorch Hub.
    # For more details, please see the [link](https://pytorch.org/hub/pytorch_vision_mobilenet_v2).
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    model.eval()

    # Step 2: Export the PyTorch model to ONNX.
    if not MODEL_DIR.exists():
        os.makedirs(MODEL_DIR)

    onnx_model_path = MODEL_DIR.joinpath('mobilenet_v2.onnx')
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy_input, onnx_model_path, verbose=False)

    # Step 3: Load the ONNX model.
    onnx_model = onnx.load(onnx_model_path)

    # Step 4: Create calibration dataset.
    data_source = create_data_source()

    # Step 5: Apply quantization algorithm.

    # Define the transformation method. This method should take a data item returned
    # per iteration through the `data_source` object and transform it into the model's
    # expected input that can be used for the model inference.
    input_name = onnx_model.graph.input[0].name
    def transform_fn(data_item):
        images, _ = data_item
        return {input_name: ONNXNNCFTensor(images.numpy())}

    # Wrap framework-specific data source into the `nncf.Dataset` object.
    calibration_dataset = nncf.Dataset(data_source, transform_fn)

    # Quantization of the ONNX model. The `quantize` method expects an ONNX
    # model as input and returns an ONNX model that has been quantized using
    # the calibration dataset.
    quantized_model = nncf.quantize(onnx_model, calibration_dataset)

    # Step 6: Save the quantized model.
    quantized_model_path = MODEL_DIR / 'quantized_mobilenet_v2.onnx'
    onnx.save(quantized_model, quantized_model_path)
    print(f'The quantized model has been saved in: {quantized_model_path}')

    # Step 7: Compare the accuracy of the original and quantized models.
    print('Checking the accuracy of the original model:')
    result = validate(onnx_model,
                      data_source,
                      providers=['OpenVINOExecutionProvider'],
                      provider_options=[{'device_type' : 'CPU_FP32'}])
    print(f'The original model accuracy@top1: {result:.3f}')

    print('Checking the accuracy of the quantized model:')
    quantized_result = validate(quantized_model,
                                data_source,
                                providers=['OpenVINOExecutionProvider'],
                                provider_options=[{'device_type' : 'CPU_FP32'}])
    print(f'The quantized model accuracy@top1: {quantized_result:.3f}')


def create_data_source() -> torch.utils.data.DataLoader:
    """
    Creates validation ImageNet dataset.

    :return: The instance of the `torch.utils.data.DataLoader`.
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


def validate(model: onnx.ModelProto,
             val_loader: torch.utils.data.DataLoader,
             providers: List[str],
             provider_options: List[Dict[str, str]],
             print_freq: int = 10000):

    def run_validate(sess, input_name, output_names, loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i

                output = torch.from_numpy(
                    sess.run(output_names, {input_name: images.numpy()})[0]
                )

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

    so = rt.SessionOptions()
    so.log_severity_level = 3
    sess = rt.InferenceSession(model.SerializeToString(), so, providers, provider_options)
    input_name = sess.get_inputs()[0].name
    outputs = sess.get_outputs()
    output_names = list(map(lambda output: output.name, outputs))

    run_validate(sess, input_name, output_names, val_loader)
    progress.display_summary()

    return top1.avg


if __name__ == '__main__':
    run_example()
