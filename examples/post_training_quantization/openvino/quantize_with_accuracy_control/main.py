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

import json
import os
import re
import subprocess
import sys
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import nncf
import numpy as np
import openvino.runtime as ov
import torch

from anomalib.data.mvtec import MVTec
from anomalib.data.utils import download
from anomalib.post_processing.normalization.min_max import normalize
from anomalib.utils.metrics import create_metric_collection

ROOT = Path(__file__).parent.resolve()
HOME_PATH = Path.home()
MODEL_INFO = download.DownloadInfo(
    name="stfpm_mvtec_capsule",
    url='https://huggingface.co/alexsu52/stfpm_mvtec_capsule/resolve/main/openvino_model.tar',
    hash='b75ce461aa17b1b33673ebcea0f6b846')
MODEL_PATH = HOME_PATH / '.cache/nncf/models/stfpm_mvtec_capsule'

DATASET_INFO = download.DownloadInfo(
    name="mvtec_capsule",
    url='https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937454-1629951595/capsule.tar.xz',
    hash='380afc46701c99cb7b9a928edbe16eb5')
DATASET_PATH = HOME_PATH / '.cache/nncf/datasets/mvtec_capsule'

max_accuracy_drop = 0.005 if len(sys.argv) < 2 else sys.argv[1]


def download_and_extract(root: Path, info: download.DownloadInfo) -> None:
    if not root.is_dir():
        download.download_and_extract(root, info)


def get_anomaly_images(data_loader: Iterable[Any]) -> List[Dict[str, torch.Tensor]]:
    anomaly_images = []
    for data_item in data_loader:
        if data_item['label'].int() == 1:
            anomaly_images.append({'image': data_item['image']})
    return anomaly_images


def validate(model: ov.CompiledModel, 
             val_loader: Iterable[Any],
             val_params: Dict[str, float]) -> float:
    metric = create_metric_collection(['F1Score'], prefix='image_')['F1Score']
    metric.threshold = 0.5
    
    output = model.outputs[0]

    counter = 0
    for batch in val_loader:
        anomaly_maps = model(batch["image"])[output]
        pred_scores = np.max(anomaly_maps, axis=(1,2,3))
        pred_scores = normalize(pred_scores, val_params['image_threshold'],
                                val_params['min'], val_params['max'])
        metric.update(torch.from_numpy(pred_scores), batch['label'].int())
        counter+=1
           
    metric_value = metric.compute()
    print(f'Validate: dataset lenght = {counter}, '
          f'metric value = {metric_value:.3f}')
    return metric_value


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
    bin_size = os.path.getsize(ir_path.replace('xml', 'bin'))
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
# Create an OpenVINO model and dataset

download_and_extract(DATASET_PATH, DATASET_INFO)

datamodule = MVTec(root=DATASET_PATH,
                   category="capsule",
                   image_size=(256, 256),
                   train_batch_size=1,
                   eval_batch_size=1,
                   num_workers=1)
datamodule.setup()
test_loader = datamodule.test_dataloader()

download_and_extract(MODEL_PATH, MODEL_INFO)
model = ov.Core().read_model(MODEL_PATH / 'stfpm_capsule.xml')

with open(MODEL_PATH / 'meta_data_stfpm_capsule.json', 'r') as f:
    validation_params = json.load(f)

###############################################################################
# Quantize an OpenVINO model with accuracy control
#
# The transformation function transforms a data item into model input data.
#
# To validate the transform function use the following code:
# >> for data_item in val_loader:
# >>    model(transform_fn(data_item))

def transform_fn(data_item):
    return data_item["image"]

# Uses only anomaly images for calibration process
anomaly_images = get_anomaly_images(test_loader)
calibration_dataset = nncf.Dataset(anomaly_images, transform_fn)

# Whole test dataset is used for validation
validation_fn = partial(validate, val_params=validation_params)
validation_dataset = nncf.Dataset(test_loader, transform_fn)

quantized_model = nncf.quantize_with_accuracy_control(
    model=model,
    calibration_dataset = calibration_dataset,
    validation_dataset = validation_dataset,
    validation_fn=validation_fn,
    max_drop=max_accuracy_drop)

###############################################################################
# Benchmark performance, calculate compression rate and validate accuracy

fp32_ir_path = f'{ROOT}/stfpm_fp32.xml'
ov.serialize(model, fp32_ir_path)
print(f'[1/7] Save FP32 model: {fp32_ir_path}')
fp32_size = get_model_size(fp32_ir_path, verbose=True)

int8_ir_path = f'{ROOT}/stfpm_int8.xml'
ov.serialize(quantized_model, int8_ir_path)
print(f'[2/7] Save INT8 model: {int8_ir_path}')
int8_size = get_model_size(int8_ir_path, verbose=True)

print('[3/7] Benchmark FP32 model:')
fp32_fps = run_benchmark(fp32_ir_path, shape=[1,3,256,256], verbose=True)
print('[4/7] Benchmark INT8 model:')
int8_fps = run_benchmark(int8_ir_path, shape=[1,3,256,256], verbose=True)

print('[5/7] Validate OpenVINO FP32 model:')
compiled_model = ov.compile_model(model)
fp32_top1 = validate(compiled_model, test_loader, validation_params)
print(f'Accuracy @ top1: {fp32_top1:.3f}')

print('[6/7] Validate OpenVINO INT8 model:')
quantized_compiled_model = ov.compile_model(quantized_model)
int8_top1 = validate(quantized_compiled_model, test_loader, validation_params)
print(f'Accuracy @ top1: {int8_top1:.3f}')

print('[7/7] Report:')
print(f'Maximum accuracy drop:                  {max_accuracy_drop}')
print(f'Accuracy drop:                          {fp32_top1 - int8_top1:.3f}')
print(f'Model compression rate:                 {fp32_size / int8_size:.3f}')
# https://docs.openvino.ai/latest/openvino_docs_optimization_guide_dldt_optimization_guide.html
print(f'Performance speed up (throughput mode): {int8_fps / fp32_fps:.3f}')
