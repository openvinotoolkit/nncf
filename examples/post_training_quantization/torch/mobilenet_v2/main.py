"""
 Copyright (c) 2023 Intel Corporation
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
from pathlib import Path
from typing import List, Optional

import nncf
import numpy as np
import openvino.runtime as ov
import torch
from fastdownload import FastDownload
from openvino.tools import mo
from sklearn.metrics import accuracy_score
from torchvision import datasets, models, transforms
from tqdm import tqdm

ROOT = Path(__file__).parent.resolve()
CHECKPOINT_URL = 'https://huggingface.co/alexsu52/mobilenet_v2_imagenette/resolve/main/pytorch_model.bin'
DATASET_URL = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz'
DATASET_PATH = '~/.cache/nncf/datasets'
DATASET_CLASSES = 10


def download_dataset() -> Path:
    downloader = FastDownload(base=DATASET_PATH, 
                              archive='downloaded', 
                              data='extracted')
    return downloader.get(DATASET_URL)


def load_checkpoint(model: torch.nn.Module) -> torch.nn.Module:  
    checkpoint = torch.hub.load_state_dict_from_url(
        CHECKPOINT_URL, 
        map_location=torch.device('cpu'), 
        progress=False)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def validate(model: ov.Model, 
             val_loader: torch.utils.data.DataLoader) -> float:
    predictions = []
    references = []

    compiled_model = ov.compile_model(model)
    output = compiled_model.outputs[0]

    for images, target in tqdm(val_loader):
        pred = compiled_model(images)[output]
        predictions.append(np.argmax(pred, axis=1))
        references.append(target)

    predictions = np.concatenate(predictions, axis=0)
    references = np.concatenate(references, axis=0)  
    return accuracy_score(predictions, references)


def run_benchmark(model_path: str, shape: Optional[List[int]] = None, 
                  verbose: bool = True) -> float:
    command = f'benchmark_app -m {model_path} -d CPU -api async -t 15'
    if shape is not None:
        command += f' -shape [{",".join(str(x) for x in shape)}]'
    cmd_output = subprocess.check_output(command, shell=True)
    if verbose:
        print(*str(cmd_output).split('\\n')[-9:-1], sep='\n')
    match = re.search(r"Throughput\: (.+?) FPS", str(cmd_output))
    return float(match.group(1))


def get_model_size(ir_path: str, m_type: str = 'Mb', 
                   verbose: bool = True) -> float:
    xml_size = os.path.getsize(ir_path)
    bin_size = os.path.getsize(os.path.splitext(ir_path)[0] + '.bin')
    for t in ['bytes', 'Kb', 'Mb']:
        if m_type == t:
            break
        xml_size /= 1024
        bin_size /= 1024
    model_size = xml_size + bin_size 
    if verbose:
        print(f'Model graph (xml):   {xml_size:.3f} Mb')
        print(f'Model weights (bin): {bin_size:.3f} Mb')
        print(f'Model size:          {model_size:.3f} Mb')      
    return model_size

###############################################################################
# Create a PyTorch model and dataset

dataset_path = download_dataset()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
val_dataset = datasets.ImageFolder(
    root=f'{dataset_path}/val',
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
)
val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=128, num_workers=4, shuffle=False)

model = models.mobilenet_v2(num_classes=DATASET_CLASSES) 
model.eval()
model = load_checkpoint(model)

###############################################################################
# Quantize a PyTorch model

'''
The transformation function transforms a data item into model input data.

To validate the transform function use the following code:
>> for data_item in val_loader:
>>    model(transform_fn(data_item))
'''
def transform_fn(data_item):
    images, _ = data_item
    return images

'''
The calibration dataset is a small, no label, representative dataset
(~100-500 samples) that is used to estimate the range, i.e. (min, max) of all
floating point activation tensors in the model, to initialize the quantization
parameters.

The easiest way to define a calibration dataset is to use a training or
validation dataset and a transformation function to remove labels from the data
item and prepare model input data. The quantize method uses a small subset
(default: 300 samples) of the calibration dataset.
'''
calibration_dataset = nncf.Dataset(val_loader, transform_fn)
quantized_model = nncf.quantize(model, calibration_dataset)

###############################################################################
# Benchmark performance, calculate compression rate and validate accuracy

ov_model = mo.convert_model(model.cpu(), input_shape=[-1,3,224,224])
ov_quantized_model = mo.convert_model(quantized_model.cpu(),
                                      input_shape=[-1,3,224,224])

fp32_ir_path = f'{ROOT}/mobilenet_v2_fp32.xml'
ov.serialize(ov_model, fp32_ir_path)
print(f'[1/7] Save FP32 model: {fp32_ir_path}')
fp32_model_size = get_model_size(fp32_ir_path, verbose=True)

int8_ir_path = f'{ROOT}/mobilenet_v2_int8.xml'
ov.serialize(ov_quantized_model, int8_ir_path)
print(f'[2/7] Save INT8 model: {int8_ir_path}')
int8_model_size = get_model_size(int8_ir_path, verbose=True)

print('[3/7] Benchmark FP32 model:')
fp32_fps = run_benchmark(fp32_ir_path, shape=[1,3,224,224], verbose=True)
print('[4/7] Benchmark INT8 model:')
int8_fps = run_benchmark(int8_ir_path, shape=[1,3,224,224], verbose=True)

print('[5/7] Validate OpenVINO FP32 model:')
fp32_top1 = validate(ov_model, val_loader)
print(f'Accuracy @ top1: {fp32_top1:.3f}')

print('[6/7] Validate OpenVINO INT8 model:')
int8_top1 = validate(ov_quantized_model, val_loader)
print(f'Accuracy @ top1: {int8_top1:.3f}')

print('[7/7] Report:')
print(f'Accuracy drop: {fp32_top1 - int8_top1:.3f}')
print(f'Model compression rate: {fp32_model_size / int8_model_size:.3f}')
# https://docs.openvino.ai/latest/openvino_docs_optimization_guide_dldt_optimization_guide.html
print(f'Performance speed up (throughput mode): {int8_fps / fp32_fps:.3f}')
