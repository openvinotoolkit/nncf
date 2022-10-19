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

from typing import Optional, Tuple

import os

import torch

from nncf.common.utils.logger import logger as nncf_logger
from nncf.onnx.tensor import ONNXNNCFTensor

from nncf.experimental.post_training.api.dataset import Dataset, NNCFData

from torchvision import transforms
from torchvision.datasets import ImageFolder


class ImageNetDataset(Dataset):
    def __init__(self, dataset, batch_size, shuffle, input_name):
        super().__init__(batch_size, shuffle)
        self.dataset = dataset
        self.input_name = input_name
        nncf_logger.info(
            f"The dataset is built with the data located on {dataset.root}. "
            f"Model input key is {input_name}."
        )

    def __getitem__(self, item) -> NNCFData:
        tensor, target = self.dataset[item]
        tensor = tensor.cpu().detach().numpy()
        return {self.input_name: ONNXNNCFTensor(tensor), "targets": ONNXNNCFTensor(target)}

    def __len__(self) -> int:
        return len(self.dataset)


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
                                  input_name: str,
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
    return ImageNetDataset(initialization_dataset, batch_size, shuffle, input_name)
