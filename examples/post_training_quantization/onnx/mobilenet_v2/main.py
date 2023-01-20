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

import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import nncf
import numpy as np
import onnx
import onnxruntime
import torch
from fastdownload import FastDownload, download_url
from sklearn.metrics import accuracy_score
from torchvision import datasets, transforms
from tqdm import tqdm

ROOT = Path(__file__).parent.resolve()
MODEL_URL = 'https://huggingface.co/alexsu52/mobilenet_v2_imagenette/resolve/main/mobilenet_v2_imagenette.onnx'
DATASET_URL = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz'
DATASET_PATH = '~/.cache/nncf/datasets'
MODEL_PATH = '~/.cache/nncf/models'
DATASET_CLASSES = 10


def download_dataset() -> Path:
    downloader = FastDownload(base=DATASET_PATH,
                              archive='downloaded',
                              data='extracted')
    return downloader.get(DATASET_URL)


def download_model() -> Path:
    return download_url(MODEL_URL, Path(MODEL_PATH).resolve())


def validate(path_to_model: str, data_loader: torch.utils.data.DataLoader,
             providers: List[str], provider_options: Dict[str, str]) -> float:
    sess = onnxruntime.InferenceSession(path_to_model, providers=providers,
                                        provider_options=provider_options)
    _input_name = sess.get_inputs()[0].name
    _output_names = [sess.get_outputs()[0].name]

    predictions = []
    references = []
    for images, target in tqdm(data_loader):
        pred = sess.run(_output_names, {_input_name: images.numpy()})[0]
        predictions.append(np.argmax(pred, axis=1))
        references.append(target)

    predictions = np.concatenate(predictions, axis=0)
    references = np.concatenate(references, axis=0)
    return accuracy_score(predictions, references)


def run_benchmark(path_to_model: str, shape: Optional[List[int]] = None,
                  verbose: bool = True) -> float:
    command = f'benchmark_app -m {path_to_model} -d CPU -api async -t 15'
    if shape is not None:
        command += f' -shape [{",".join(str(x) for x in shape)}]'
    cmd_output = subprocess.check_output(command, shell=True)
    if verbose:
        print(*str(cmd_output).split('\\n')[-9:-1], sep='\n')
    match = re.search(r"Throughput\: (.+?) FPS", str(cmd_output))
    return float(match.group(1))

###############################################################################
# Create an ONNX model and dataset

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
    val_dataset, batch_size=1, shuffle=False)

model_path = download_model()
model = onnx.load(model_path)

###############################################################################
# Quantize an ONNX model

# The transformation function transforms a data item into model input data.
#
# To validate the transform function use the following code:
# >> sess = onnxruntime.InferenceSession(model_path)
# >> output_names = [output.name for output in sess.get_outputs()]
# >> for data_item in val_loader:
# >>    sess.run(output_names, input_feed=transform_fn(data_item))
input_name = model.graph.input[0].name
def transform_fn(data_item):
    images, _ = data_item
    return {input_name: images.numpy()}

# The calibration dataset is a small, no label, representative dataset
# (~100-500 samples) that is used to estimate the range, i.e. (min, max) of all
# floating point activation tensors in the model, to initialize the quantization
# parameters.

# The easiest way to define a calibration dataset is to use a training or
# validation dataset and a transformation function to remove labels from the data
# item and prepare model input data. The quantize method uses a small subset
# (default: 300 samples) of the calibration dataset.
calibration_dataset = nncf.Dataset(val_loader, transform_fn)
quantized_model = nncf.quantize(model, calibration_dataset)

###############################################################################
# Benchmark performance and validate accuracy

fp32_model_path = f'{ROOT}/mobilenet_v2_fp32.onnx'
onnx.save(model, fp32_model_path)
print(f'[1/7] Save FP32 model: {fp32_model_path}')

int8_model_path = f'{ROOT}/mobilenet_v2_int8.onnx'
onnx.save(quantized_model, int8_model_path)
print(f'[2/7] Save INT8 model: {int8_model_path}')

print('[3/7] Benchmark FP32 model:')
fp32_fps = run_benchmark(fp32_model_path, shape=[1,3,224,224], verbose=True)
print('[4/7] Benchmark INT8 model:')
int8_fps = run_benchmark(int8_model_path, shape=[1,3,224,224], verbose=True)

print('[5/7] Validate OpenVINO FP32 model:')
fp32_top1 = validate(fp32_model_path, val_loader,
                     providers = ['OpenVINOExecutionProvider'],
                     provider_options = [{'device_type' : 'CPU_FP32'}])
print(f'Accuracy @ top1: {fp32_top1:.3f}')

print('[6/7] Validate OpenVINO INT8 model:')
int8_top1 = validate(int8_model_path, val_loader,
                     providers = ['OpenVINOExecutionProvider'],
                     provider_options = [{'device_type' : 'CPU_FP32'}])
print(f'Accuracy @ top1: {int8_top1:.3f}')

print('[7/7] Report:')
print(f'Accuracy drop: {fp32_top1 - int8_top1:.3f}')
# https://docs.openvino.ai/latest/openvino_docs_optimization_guide_dldt_optimization_guide.html
print(f'Performance speed up (throughput mode): {int8_fps / fp32_fps:.3f}')
