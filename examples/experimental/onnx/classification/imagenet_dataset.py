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
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader
from nncf import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


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


def create_dataloader(dataset_dir: str,
                      input_shape: Optional[Tuple[int, int, int, int]],
                      mean=(0.485, 0.456, 0.406),
                      std=(0.229, 0.224, 0.225),
                      crop_ratio=0.875,
                      batch_size: int = 1,
                      shuffle: bool = True) -> DataLoader:

    channel_last = input_shape[1] > input_shape[3] and input_shape[2] > input_shape[3]

    if channel_last:
        image_size = [input_shape[-3], input_shape[-2]]
    else:
        image_size = [input_shape[-2], input_shape[-1]]

    transform = get_transform(image_size, crop_ratio, mean, std, channel_last)
    # The best practise is to use validation part of dataset for calibration (aligning with POT)
    initialization_dataset = ImageFolder(os.path.join(dataset_dir), transform)
    return DataLoader(initialization_dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      generator=torch.manual_seed(0))


def create_dataset(dataloader: DataLoader, input_name: str) -> Dataset:

    def transform_fn(data_item):
        tensor, _ = data_item
        tensor = tensor.cpu().detach().numpy()
        return {input_name: tensor}

    return Dataset(dataloader, transform_fn)
