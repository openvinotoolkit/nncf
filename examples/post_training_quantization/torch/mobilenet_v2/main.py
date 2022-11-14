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
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import nncf
import numpy as np
import openvino.runtime as ov
import torch
from sklearn.metrics import accuracy_score
from openvino.offline_transformations import compress_quantize_weights_transformation
from torchvision import datasets
from torchvision import models
from torchvision import transforms
from tqdm import tqdm

FOOD101_CLASSES = 101
ROOT = Path(__file__).parent.resolve()
DATASET_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'dataset'
CHECKPOINT_URL = 'https://huggingface.co/AlexKoff88/mobilenet_v2_food101/resolve/main/pytorch_model.bin'


def fix_names(state_dict):
    state_dict = {key.replace('module.', ''): value for (key, value) in state_dict.items()}
    return state_dict


def load_checkpoint(model):  
    checkpoint = torch.hub.load_state_dict_from_url(CHECKPOINT_URL, progress=False)
    weights = fix_names(checkpoint['state_dict'])
    model.load_state_dict(weights)
    return model


def validate(model, val_loader):
    predictions = []
    references = []

    output_name = model.outputs[0]

    for images, target in tqdm(val_loader):
        pred = model(images)[output_name]
        predictions.append(np.argmax(pred, axis=1))
        references.append(target)

    predictions = np.concatenate(predictions, axis=0)
    references = np.concatenate(references, axis=0)  
    return accuracy_score(predictions, references)


def ov_convert(model, args):
    onnx_model_path = f'{tempfile.gettempdir()}/model.onnx'
    torch.onnx.export(model, args, onnx_model_path, verbose=False)

    ov_model = ov.Core().read_model(onnx_model_path)
    compress_quantize_weights_transformation(ov_model)
    return ov_model


def ov_benchmark(model, verbose=True):
    ov_model_path = f'{tempfile.gettempdir()}/model.xml'
    ov.serialize(model, ov_model_path)

    command = f'benchmark_app -m {ov_model_path} -d CPU -api async -t 15'
    cmd_output = subprocess.check_output(command, shell=True)
    if verbose:
        print(*str(cmd_output).split('\\n')[-9:-1], sep='\n')
    match = re.search(r"Throughput\: (.+?) FPS", str(cmd_output))
    return float(match.group(1))


def ov_model_size(ir_path, m_type='Mb'):
    model_size = os.path.getsize(ir_path) + os.path.getsize(ir_path.replace('xml', 'bin'))
    for t in ['bytes', 'Kb', 'Mb']:
        if m_type == t:
            return model_size
        model_size /= 1024
    return model_size

###########################################################################
# Create a PyTorch model and dataset
###########################################################################
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
val_dataset = datasets.Food101(
    root=DATASET_PATH,
    split = 'test', 
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]),
    download = True
)
val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=128, num_workers=4, shuffle=False)

model = models.mobilenet_v2(num_classes=FOOD101_CLASSES) 
model.eval()
model = load_checkpoint(model)

###########################################################################
# Quantize a PyTorch model
###########################################################################

'''
Transformation function transforms a data item into model input.

To validate the transform function use the following code:
>> for data_item in val_loader:
>>    model(transform_fn(data_item))
'''
def transform_fn(data_item):
    images, _ = data_item
    return images

calibration_dataset = nncf.Dataset(val_loader, transform_fn)
quantized_model = nncf.quantize(model, calibration_dataset)

###########################################################################
# Benchmark performance, calculate compression rate and validate accuracy
###########################################################################

dummy_input = torch.randn(1, 3, 224, 224)
ov_model = ov_convert(model.cpu(), dummy_input)
ov_quantized_model = ov_convert(quantized_model.cpu(), dummy_input)

print('[1/7] Benchmark FP32 model:')
fp32_fps = ov_benchmark(ov_model, verbose=True)
print('[2/7] Benchmark INT8 model:')
int8_fps = ov_benchmark(ov_quantized_model, verbose=True)

fp32_ir_path = f'{ROOT}/mobilenet_v2_fp32.xml'
int8_ir_path = f'{ROOT}/mobilenet_v2_int8.xml'
ov.serialize(ov_model, fp32_ir_path)
ov.serialize(ov_quantized_model, int8_ir_path)

fp32_model_size = ov_model_size(fp32_ir_path)
int8_model_size = ov_model_size(int8_ir_path)
print(f'[3/7] Save FP32 model: {fp32_ir_path} ({fp32_model_size:.3f} Mb)')
print(f'[4/7] Save INT8 model: {int8_ir_path} ({int8_model_size:.3f} Mb)')

# Using [the dynamic shape feature](https://docs.openvino.ai/latest/openvino_docs_OV_UG_DynamicShapes.html)
# to compute the last batch of the validation dataset
ov_model.reshape([-1, 3, 224, 224])
ov_quantized_model.reshape([-1, 3, 224, 224])

compiled_model = ov.compile_model(ov_model)
compiled_quantized_model = ov.compile_model(ov_quantized_model)

print('[5/7] Validate OpenVINO FP32 model:')
fp32_top1 = validate(compiled_model, val_loader)
print(f'Accuracy @ top1: {fp32_top1:.3f}')

print('[6/7] Validate OpenVINO INT8 model:')
int8_top1 = validate(compiled_quantized_model, val_loader)
print(f'Accuracy @ top1: {int8_top1:.3f}')

print('[7/7] Report:')
print(f'Accuracy drop: {fp32_top1 - int8_top1:.3f}')
print(f'Compression rate: {fp32_model_size / int8_model_size:.3f}')
# https://docs.openvino.ai/2018_R5/_docs_IE_DG_Intro_to_Performance.html
print(f'Performance speed up (throughput mode): {int8_fps / fp32_fps:.3f}') 
