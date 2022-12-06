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
import tempfile
from pathlib import Path
from typing import Tuple

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
DATASET_PATH = '~/.nncf'
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


def validate(model: ov.CompiledModel, 
             val_loader: torch.utils.data.DataLoader) -> float:
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


def ov_benchmark(model: ov.Model, verbose: bool = True) -> float:
    ov_model_path = f'{tempfile.gettempdir()}/model.xml'
    ov.serialize(model, ov_model_path)

    command = f'benchmark_app -m {ov_model_path} -d CPU -api async -t 15'
    cmd_output = subprocess.check_output(command, shell=True)
    if verbose:
        print(*str(cmd_output).split('\\n')[-9:-1], sep='\n')
    match = re.search(r"Throughput\: (.+?) FPS", str(cmd_output))
    return float(match.group(1))


def ov_model_size(ir_path: Path, m_type: str = 'Mb') -> Tuple[float, float]:
    xml_size = os.path.getsize(ir_path)
    bin_size = os.path.getsize(ir_path.replace('xml', 'bin'))
    for t in ['bytes', 'Kb', 'Mb']:
        if m_type == t:
            return (xml_size, bin_size)
        xml_size /= 1024
        bin_size /= 1024
    return (xml_size, bin_size)

###########################################################################
# Create a PyTorch model and dataset
###########################################################################
dataset_path = download_dataset()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
val_dataset = datasets.ImageFolder(
    root=str(dataset_path / 'val'),
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

###########################################################################
# Quantize a PyTorch model
###########################################################################

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
(default: 300 samples) of the calibration data set.
'''
calibration_dataset = nncf.Dataset(val_loader, transform_fn)
quantized_model = nncf.quantize(model, calibration_dataset)

###########################################################################
# Benchmark performance, calculate compression rate and validate accuracy
###########################################################################

dummy_input = torch.randn(1, 3, 224, 224)
ov_model = mo.convert_model(model.cpu(), example_input=dummy_input)
ov_quantized_model = mo.convert_model(quantized_model.cpu(), 
                                      example_input=dummy_input)

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
print(f'[3/7] Save FP32 model: {fp32_ir_path} ({sum(fp32_model_size):.3f} Mb = '
      f'{fp32_model_size[0]:.3f} Mb (xml) + {fp32_model_size[1]:.3f} Mb (bin))')
print(f'[4/7] Save INT8 model: {int8_ir_path} ({sum(int8_model_size):.3f} Mb = '
      f'{int8_model_size[0]:.3f} Mb (xml) + {int8_model_size[1]:.3f} Mb (bin))')

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
print(f'Model compression rate: {sum(fp32_model_size) / sum(int8_model_size):.3f}')
# https://docs.openvino.ai/latest/openvino_docs_optimization_guide_dldt_optimization_guide.html
print(f'Performance speed up (throughput mode): {int8_fps / fp32_fps:.3f}')
