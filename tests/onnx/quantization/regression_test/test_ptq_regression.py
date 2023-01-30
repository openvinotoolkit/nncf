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
from typing import Dict, List
import pytest

import nncf
import numpy as np
import onnx
from onnx import version_converter
import onnxruntime
import torch
from fastdownload import FastDownload
from sklearn.metrics import accuracy_score
from torchvision import datasets, transforms
from tqdm import tqdm
from pathlib import Path

MODELS = [
    ('https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-12.onnx',
     'mobilenetv2-12', 0.7877707006369427),
    ('https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v1-7.onnx',
     'resnet50-v1-7', 0.8101910828025478),
    ('https://github.com/onnx/models/raw/main/vision/classification/shufflenet/model/shufflenet-v2-12.onnx',
    'shufflenet-v2-12', 0.7806369426751593)
]

DATASET_URL = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz'
DATASET_PATH = '~/.cache/nncf/datasets'


def download_dataset() -> Path:
    downloader = FastDownload(base=DATASET_PATH,
                              archive='downloaded',
                              data='extracted')
    return downloader.get(DATASET_URL)


def download_model(model_url, tmp_path) -> Path:
    downloader = FastDownload(base=tmp_path)
    return downloader.download(model_url)


def validate(quantized_model: onnx.ModelProto, data_loader: torch.utils.data.DataLoader,
             providers: List[str], provider_options: Dict[str, str]) -> float:
    sess = onnxruntime.InferenceSession(quantized_model.SerializeToString(), providers=providers,
                                        provider_options=provider_options)
    _input_name = sess.get_inputs()[0].name
    _output_names = [sess.get_outputs()[0].name]

    predictions = []
    references = []

    from_imagenet_to_imageneetee = {
        0: 0,  # tench
        217: 1,  # English springer
        482: 2,  # cassette player
        491: 3,  # chain saw
        497: 4,  # church
        566: 5,  # French horn
        569: 6,  # garbage truck
        571: 7,  # gas pump
        574: 8,  # golf ball
        701: 9  # parachute
    }
    # TODO: (kshpv) add async
    for images, target in tqdm(data_loader):
        pred = sess.run(_output_names, {_input_name: images.numpy()})[0]
        pred_class = np.argmax(pred, axis=1)
        if pred_class.item() in from_imagenet_to_imageneetee:
            pred_class[0] = from_imagenet_to_imageneetee[pred_class.item()]
        predictions.append(pred_class)
        if target.item() in from_imagenet_to_imageneetee:
            target[0] = from_imagenet_to_imageneetee[target.item()]
        references.append(target)

    predictions = np.concatenate(predictions, axis=0)
    references = np.concatenate(references, axis=0)
    return accuracy_score(predictions, references)


@pytest.mark.parametrize('model_url, model_name, int8_ref_top1', MODELS, ids=[model[1] for model in MODELS])
def test_compression(tmp_path, model_url, model_name, int8_ref_top1):
    original_model_path = download_model(model_url, tmp_path)
    dataset_path = download_dataset()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_dataset = datasets.ImageFolder(
        root=str(Path(dataset_path) / 'val'),
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False)

    model = onnx.load_model(original_model_path)
    # ALL models are not in target opset
    converted_model = version_converter.convert_version(model, 13)
    input_name = converted_model.graph.input[0].name

    def transform_fn(data_item):
        images, _ = data_item
        return {input_name: images.numpy()}

    calibration_dataset = nncf.Dataset(val_loader, transform_fn)
    quantized_model = nncf.quantize(converted_model, calibration_dataset)
    int8_top1 = validate(quantized_model, val_loader,
                         providers=['OpenVINOExecutionProvider'],
                         provider_options=[{'device_type': 'CPU_FP32'}])

    assert abs(int8_top1 - int8_ref_top1) < 1e-6
