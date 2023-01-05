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
import requests
import shutil
from fastdownload import FastDownload
from pathlib import Path

IMAGENETTE_URL = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz'
IMAGENETTE_ANNOTATION_URL =  \
        'https://huggingface.co/datasets/frgfm/imagenette/resolve/main/metadata/imagenette2-320/val.txt'
WIDER_FACE_URL = 'https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_val.zip'
WIDER_FACE_ANNOTATION_URL = 'https://huggingface.co/datasets/wider_face/resolve/main/data/wider_face_split.zip'
DATASET_PATH = '~/.cache/nncf/datasets'


def download(url: str, path: str) -> Path:
    downloader = FastDownload(base=path, archive='downloaded', data='extracted')
    return downloader.get(url)


def preprocess_imagenette_data(dataset_path: str) -> None:
    destination = os.path.join(dataset_path, 'val')
    for filename in os.listdir(destination):
        path = os.path.join(destination, filename)
        if os.path.isdir(path):
            for img_name in os.listdir(path):
                img_path = os.path.join(path, img_name)
                shutil.move(img_path, destination)
            os.rmdir(path)


def preprocess_imagenette_labels(dataset_path: str) -> None:
    labels_map = {
        'n01440764': 0,    # tench
        'n02102040': 217,  # English springer
        'n02979186': 482,  # cassette player
        'n03000684': 491,  # chain saw
        'n03028079': 497,  # church
        'n03394916': 566,  # French horn
        'n03417042': 569,  # garbage truck
        'n03425413': 571,  # gas pump
        'n03445777': 574,  # golf ball
        'n03888257': 701   # parachute
    }

    response = requests.get(IMAGENETTE_ANNOTATION_URL, timeout=10)
    annotation_path = dataset_path / 'imagenette2-320_val.txt'
    with open(annotation_path, 'w', encoding='utf-8') as output_file:
        for line in response.iter_lines():
            image_path = line.decode("utf-8").split('/')
            class_name = image_path[2]
            new_path = os.path.join(image_path[0], image_path[1], image_path[3])
            label = labels_map[class_name]
            img_path_with_labels = f'{new_path} {label}\n'
            output_file.write(img_path_with_labels)


def prepare_imagenette_for_test() -> Path:
    dataset_path = download(IMAGENETTE_URL, DATASET_PATH)
    preprocess_imagenette_labels(dataset_path)
    preprocess_imagenette_data(dataset_path)
    return dataset_path


def prepare_wider_for_test() -> Path:
    dataset_path = download(WIDER_FACE_URL, DATASET_PATH)
    _ = download(WIDER_FACE_ANNOTATION_URL, DATASET_PATH)
    return dataset_path


def get_dataset_for_test(dataset_name: str) -> Path:
    if dataset_name == 'imagenette2-320':
        return prepare_imagenette_for_test()
    if dataset_name == 'wider':
        return prepare_wider_for_test()

    raise RuntimeError(f'Unknown dataset: {dataset_name}.')
