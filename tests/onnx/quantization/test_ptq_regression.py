# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import openvino as ov
import pooch
import pytest
import torch
from onnx import version_converter
from openvino import Core
from sklearn.metrics import accuracy_score
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm

import nncf


@dataclass
class TestModel:
    model_url: str
    model_name: str
    int8_ref_top1: float
    num_inputs_initial_model: int
    num_inputs_quantized_model: int

    def __str__(self) -> str:
        return self.model_name


MODELS = [
    TestModel(
        "https://github.com/onnx/models/raw/5faef4c33eba0395177850e1e31c4a6a9e634c82/vision/classification/mobilenet/model/mobilenetv2-12.onnx",
        "mobilenetv2-12",
        0.7864968152866242,
        1,
        1,
    ),
    TestModel(
        "https://github.com/onnx/models/raw/5faef4c33eba0395177850e1e31c4a6a9e634c82/vision/classification/resnet/model/resnet50-v1-7.onnx",
        "resnet50-v1-7",
        0.8114649681528663,
        300,
        246,
    ),
    TestModel(
        "https://github.com/onnx/models/raw/5faef4c33eba0395177850e1e31c4a6a9e634c82/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx",
        "efficientnet-lite4-11",
        0.8035668789808917,
        1,
        1,
    ),
]

DATASET_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"


@pytest.fixture(name="data_dir")
def data(request):
    option = request.config.getoption("--data")
    if option is None:
        return Path.home() / ".cache" / "nncf" / "datasets"
    return Path(option)


@pytest.fixture(name="model_dir")
def models(request, tmp_path):
    option = request.config.getoption("--data")
    if option is None:
        return Path(tmp_path)
    return Path(option)


def download_dataset(dataset_path: Path) -> Path:
    files = pooch.retrieve(
        url=DATASET_URL,
        path=dataset_path / "downloaded",
        processor=pooch.Untar(extract_dir=dataset_path / "extracted"),
    )
    return dataset_path / "extracted" / Path(files[0]).relative_to(dataset_path / "extracted").parts[0]


def download_model(model_url, tmp_path) -> Path:
    model_file = pooch.retrieve(url=model_url, path=tmp_path, known_hash=None)
    return Path(model_file)


def validate(quantized_model_path: Path, data_loader: torch.utils.data.DataLoader) -> float:
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
        701: 9,  # parachute
    }
    core = Core()
    compiled_model = core.compile_model(quantized_model_path, device_name="CPU")
    infer_queue = ov.AsyncInferQueue(compiled_model)

    predictions = [0] * len(data_loader)
    references = [-1] * len(data_loader)

    def res_callback(infer_request: ov.InferRequest, userdata) -> None:
        pred = infer_request.get_output_tensor().data
        pred_class = np.argmax(pred, axis=1)
        if pred_class.item() in from_imagenet_to_imageneetee:
            pred_class[0] = from_imagenet_to_imageneetee[pred_class.item()]
        predictions[userdata] = [pred_class]

    infer_queue.set_callback(res_callback)
    print("Validate")
    print(len(data_loader))
    for i, (images, target) in tqdm(enumerate(data_loader)):
        print(i, target)
        infer_queue.start_async(images, userdata=i)
        references[i] = target
    infer_queue.wait_all()

    predictions = np.concatenate(predictions, axis=0)
    references = np.concatenate(references, axis=0)
    return accuracy_score(predictions, references)


@pytest.mark.parametrize("test_model", MODELS, ids=str)
def test_compression(tmp_path, model_dir, data_dir, test_model: TestModel):
    original_model_path = download_model(test_model.model_url, model_dir)
    dataset_path = download_dataset(data_dir)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    val_dataset = datasets.ImageFolder(
        root=str(Path(dataset_path) / "val"),
        transform=transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
                transforms.Lambda(
                    lambda images: torch.moveaxis(images, 0, 2)
                    if test_model.model_name == "efficientnet-lite4-11"
                    else images
                ),
            ]
        ),
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = onnx.load_model(original_model_path)
    # ALL models are not in target opset
    converted_model = version_converter.convert_version(model, 13)
    input_name = converted_model.graph.input[0].name

    def transform_fn(data_item):
        images, _ = data_item
        return {input_name: images.numpy()}

    assert len([inp.name for inp in converted_model.graph.input]) == test_model.num_inputs_initial_model

    calibration_dataset = nncf.Dataset(val_loader, transform_fn)
    quantized_model = nncf.quantize(converted_model, calibration_dataset)

    assert len([inp.name for inp in quantized_model.graph.input]) == test_model.num_inputs_quantized_model

    int8_model_path = tmp_path / "quantized_model.onnx"
    onnx.save_model(quantized_model, str(int8_model_path))
    int8_top1 = validate(int8_model_path, val_loader)
    print(f"INT8 metrics = {int8_top1}")
    assert int8_top1 == pytest.approx(test_model.int8_ref_top1, abs=3e-3)  # 0.03 deviations
