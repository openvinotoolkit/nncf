# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
from pathlib import Path

import requests
from fastdownload import FastDownload
from openvino.tools.accuracy_checker.argparser import build_arguments_parser
from openvino.tools.accuracy_checker.config import ConfigReader
from openvino.tools.accuracy_checker.evaluators import ModelEvaluator

from nncf import Dataset
from tests.openvino.omz_helpers import OPENVINO_DATASET_DEFINITIONS_PATH

IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
IMAGENETTE_ANNOTATION_URL = (
    "https://huggingface.co/datasets/frgfm/imagenette/resolve/main/metadata/imagenette2-320/val.txt"
)
WIDER_FACE_URL = "https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_val.zip"
WIDER_FACE_ANNOTATION_URL = "https://huggingface.co/datasets/wider_face/resolve/main/data/wider_face_split.zip"


def download(url: str, path: str) -> Path:
    downloader = FastDownload(base=path, archive="downloaded", data="extracted")
    return downloader.get(url)


def preprocess_imagenette_data(dataset_path: Path, destination_path: Path) -> None:
    val_dataset_path = dataset_path / "val"
    for filename in os.listdir(val_dataset_path):
        path = val_dataset_path / filename
        if path.is_dir():
            for img_name in path.iterdir():
                img_path = path / img_name
                shutil.copy(img_path, destination_path)


def preprocess_imagenette_labels(destination_path: Path) -> None:
    labels_map = {
        "n01440764": 0,  # tench
        "n02102040": 217,  # English springer
        "n02979186": 482,  # cassette player
        "n03000684": 491,  # chain saw
        "n03028079": 497,  # church
        "n03394916": 566,  # French horn
        "n03417042": 569,  # garbage truck
        "n03425413": 571,  # gas pump
        "n03445777": 574,  # golf ball
        "n03888257": 701,  # parachute
    }
    response = requests.get(IMAGENETTE_ANNOTATION_URL, timeout=10)
    annotation_path = destination_path / "imagenette2-320_val.txt"

    with open(annotation_path, "w+", encoding="utf-8") as output_file:
        for line in response.iter_lines():
            image_path = line.decode("utf-8").split("/")
            class_name = image_path[2]
            new_path = os.path.join(image_path[0], image_path[1], image_path[2], image_path[3])
            label = labels_map[class_name]
            img_path_with_labels = f"{new_path} {label}\n"
            output_file.write(img_path_with_labels)


def convert_dataset_to_ac_format(dataset_path: Path, destination_path: Path) -> None:
    preprocess_imagenette_labels(destination_path)
    preprocess_imagenette_data(dataset_path, destination_path)


def prepare_imagenette_for_test(data_dir: Path) -> Path:
    dataset_path = download(IMAGENETTE_URL, data_dir)
    destination_path = dataset_path.parent / "ac_imagenette2-320/"
    if not destination_path.exists():
        destination_path.mkdir()
    convert_dataset_to_ac_format(dataset_path, destination_path)
    return dataset_path


def prepare_wider_for_test(data_dir: Path) -> Path:
    dataset_path = download(WIDER_FACE_URL, data_dir)
    _ = download(WIDER_FACE_ANNOTATION_URL, data_dir)
    return dataset_path


def get_dataset_for_test(dataset_name: str, data_dir: Path) -> Path:
    if dataset_name == "imagenette2-320":
        return prepare_imagenette_for_test(data_dir)
    if dataset_name == "wider":
        return prepare_wider_for_test(data_dir)

    raise RuntimeError(f"Unknown dataset: {dataset_name}.")


def get_nncf_dataset_from_ac_config(model_path, config_path, data_dir, framework="openvino", device="CPU"):
    args = [
        "-c",
        str(config_path),
        "-m",
        str(model_path),
        "-d",
        str(OPENVINO_DATASET_DEFINITIONS_PATH),
        "-s",
        str(data_dir),
        "-tf",
        framework,
        "-td",
        device,
    ]
    parser = build_arguments_parser()
    args = parser.parse_args(args)

    config, mode = ConfigReader.merge(args)
    model_evaluator = ModelEvaluator.from_configs(config[mode][0])

    def transform_fn(data_item):
        _, batch_annotation, batch_input, _ = data_item
        filled_inputs, _, _ = model_evaluator._get_batch_input(batch_annotation, batch_input)
        return filled_inputs[0]

    calibration_dataset = Dataset(model_evaluator.dataset, transform_fn)
    return calibration_dataset
