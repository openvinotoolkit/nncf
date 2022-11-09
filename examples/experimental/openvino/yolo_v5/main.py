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
from pathlib import Path

import openvino.runtime as ov
import torch
import nncf

from yolov5.utils.general import check_dataset
from yolov5.utils.general import download
from yolov5.utils.dataloaders import create_dataloader
from yolov5.val import run as validation_fn


# Path to the `yolo_v5` directory.
ROOT = Path(__file__).parent.resolve()
# Path to the directory where the original and quantized IR will be saved.
MODEL_DIR = ROOT / 'yolov5m_quantization'
# Path to the dataset config from the `ultralytics/yolov5` repository.
DATASET_CONFIG = ROOT / 'yolov5' / 'data' / 'coco.yaml'


ie = ov.Core()


def run_example():
    """
    Runs the YOLOv5 quantization example.
    """
    # Step 1: Initialize model from the PyTorch Hub.
    repo_path = str(ROOT.joinpath('yolov5'))
    model = torch.hub.load(repo_path, 'yolov5m', pretrained=True, source='local')
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
        images, *_ = data_item
        images = images.float()
        images = images / 255
        images = images.cpu().detach().numpy()
        return images

    # Wrap framework-specific data source into the `nncf.Dataset` object.
    calibration_dataset = nncf.Dataset(data_source, transform_fn)
    quantized_model = nncf.quantize(ov_model, calibration_dataset, preset=nncf.QuantizationPreset.MIXED)

    # Step 5: Save the quantized model.
    ir_qmodel_xml = MODEL_DIR / 'yolov5m_quantized.xml'
    ir_qmodel_bin = MODEL_DIR / 'yolov5m_quantized.bin'
    ov.serialize(quantized_model, str(ir_qmodel_xml), str(ir_qmodel_bin))

    # Step 6: Compare the accuracy of the original and quantized models.
    print('Checking the accuracy of the original model:')
    metrics = validation_fn(data=DATASET_CONFIG,
                            weights=MODEL_DIR.joinpath('yolov5m.xml'),  # Already supports.
                            batch_size=1,
                            workers=1,
                            plots=False,
                            device='cpu',
                            iou_thres=0.65)
    print(f'mAP@.5 = {metrics[0][2]}')
    print(f'mAP@.5:.95 = {metrics[0][3]}')

    print('Checking the accuracy of the quantized model:')
    metrics = validation_fn(data=DATASET_CONFIG,
                            weights=ir_qmodel_xml,  # Already supports.
                            batch_size=1,
                            workers=1,
                            plots=False,
                            device='cpu',
                            iou_thres=0.65)
    print(f'mAP@.5 = {metrics[0][2]}')
    print(f'mAP@.5:.95 = {metrics[0][3]}')

    # Step 7: Compare Performance of the original and quantized models
    # benchmark_app -m yolov5m_quantization/yolov5m.xml -d CPU -api async
    # benchmark_app -m yolov5m_quantization/yolov5m_quantized.xml -d CPU -api async


def convert_torch_to_openvino(model: torch.nn.Module) -> ov.Model:
    """
    Converts PyTorch YOLOv5 model to the OpenVINO IR format.
    """
    if not MODEL_DIR.exists():
        MODEL_DIR.mkdir()

    # Export PyTorch model to the ONNX format.
    onnx_model_path = MODEL_DIR / 'yolov5m.onnx'
    dummy_input = torch.randn(1, 3, 640, 640)
    torch.onnx.export(model, dummy_input, onnx_model_path, verbose=False)

    # Run Model Optimizer to convert ONNX model to OpenVINO IR.
    mo_command = f'mo --framework onnx -m {onnx_model_path} --output_dir {MODEL_DIR}'
    subprocess.call(mo_command, shell=True)

    ir_model_xml = MODEL_DIR / 'yolov5m.xml'
    ir_model_bin = MODEL_DIR / 'yolov5m.bin'
    ov_model = ie.read_model(model=ir_model_xml, weights=ir_model_bin)

    return ov_model


def create_data_source() -> torch.utils.data.DataLoader:
    """
    Creates COCO 2017 validation data loader. The method downloads COCO 2017
    dataset if it does not exist.
    """
    if not ROOT.joinpath('datasets', 'coco').exists():
        urls = ['https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels.zip']
        download(urls, dir=ROOT.joinpath('datasets'))

        urls = ['http://images.cocodataset.org/zips/val2017.zip']
        download(urls, dir=ROOT.joinpath('datasets', 'coco', 'images'))

    data = check_dataset(DATASET_CONFIG)
    val_dataloader = create_dataloader(data['val'],
                                       imgsz=640,
                                       batch_size=1,
                                       stride=32,
                                       pad=0.5,
                                       workers=1)[0]

    return val_dataloader


if __name__ == '__main__':
    run_example()
