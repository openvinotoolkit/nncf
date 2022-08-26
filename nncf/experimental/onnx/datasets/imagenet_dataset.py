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

from typing import Optional, Tuple, List

import os

import torch

from nncf.common.utils.logger import logger as nncf_logger
from nncf.experimental.onnx.tensor import ONNXNNCFTensor

from nncf.experimental.post_training.api.dataset import Dataset, NNCFData

from onnx import ModelProto
from google.protobuf.json_format import MessageToDict
from torchvision import transforms
from torchvision.datasets import ImageFolder


class ImageNetDataset(Dataset):
    def __init__(self, dataset, batch_size, shuffle, input_key):
        super().__init__(batch_size, shuffle)
        self.dataset = dataset
        self.input_key = input_key
        nncf_logger.info(
            f"The dataset is built with the data located on {dataset.root}. "
            f"Model input key is {input_key}."
        )

    def __getitem__(self, item) -> NNCFData:
        tensor, target = self.dataset[item]
        tensor = tensor.cpu().detach().numpy()
        return {self.input_key: ONNXNNCFTensor(tensor), "targets": ONNXNNCFTensor(target)}

    def __len__(self) -> int:
        return len(self.dataset)


def infer_input_shape(model: ModelProto,
                      main_shape: Optional[List[int]] = None,
                      main_keys: Optional[str] = None) -> Tuple[List[int], List[str]]:
    assert len(model.graph.input) > 0

    def set_input_shape(node):
        dim = node.type.tensor_type.shape.dim
        return [int(MessageToDict(d).get("dimValue")) for d in dim]

    if main_shape and main_keys:
        return main_shape, [main_keys]

    if main_keys:
        nncf_logger.info(
            "input_shape is None. Infer input_shape from the model.")

        for _input in model.graph.input:
            if main_keys == _input.name:
                input_shape = set_input_shape(_input)
                input_keys = main_keys
                break

    elif main_shape:
        nncf_logger.info(
            "input_keys is None. Infer input_keys from the model.")
            
        for _input in model.graph.input:
            _input_shape = set_input_shape(_input)
            if main_shape == _input_shape:
                input_shape = main_shape
                input_keys = _input.name
                break

    else:
        raise ValueError('Either main_shape or main_keys must be set correctly.')

    assert len(input_shape) == 4 and input_keys is not None

    return input_shape, [input_keys]


def get_transform(image_size: Tuple[int, int],
                  crop_ratio: float,
                  mean: Tuple[float, float, float],
                  std: Tuple[float, float, float],
                  channel_last: bool) -> transforms.Lambda:
    size = int(image_size[0] / crop_ratio)
    normalize = transforms.Normalize(mean=mean,
                                     std=std)

    transform_list = [
        transforms.Resize(size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ]

    if channel_last:
        transform_list += [transforms.Lambda(
            lambda x: torch.permute(x, (1, 2, 0)))]

    return transforms.Compose(transform_list)


def create_imagenet_torch_dataset(dataset_dir: str,
                                  input_key: str,
                                  input_shape: Optional[Tuple[int, int, int, int]],
                                  mean=(0.485, 0.456, 0.406),
                                  std=(0.229, 0.224, 0.225),
                                  crop_ratio=0.875,
                                  batch_size: int = 1,
                                  shuffle: bool = True):
    channel_last = input_shape[1] > input_shape[3] and input_shape[2] > input_shape[3]

    if channel_last:
        image_size = [input_shape[-3], input_shape[-2]]
    else:
        image_size = [input_shape[-2], input_shape[-1]]

    transform = get_transform(image_size, crop_ratio, mean, std, channel_last)
    # The best practise is to use validation part of dataset for calibration (aligning with POT)
    initialization_dataset = ImageFolder(os.path.join(dataset_dir), transform)
    return ImageNetDataset(initialization_dataset, batch_size, shuffle, input_key)
