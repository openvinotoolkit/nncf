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

from typing import List

import os

import numpy as np

from nncf.common.utils.logger import logger as nncf_logger
from nncf.experimental.post_training.api.dataloader import DataLoader


class ImageNetDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle):
        super().__init__(batch_size, shuffle)
        self.dataset = dataset
        nncf_logger.info('The dataloader is built with the data located on  {}'.format(dataset.root))

    def __getitem__(self, item):
        tensor, target = self.dataset[item]
        tensor = tensor.cpu().detach().numpy()
        return tensor, target

    def __len__(self):
        return len(self.dataset)


class BinarizedImageNetDataLoader(DataLoader):
    def __init__(self, dataset_path, batch_size, shuffle):
        super().__init__(batch_size, shuffle)
        self.dataset_path = dataset_path
        self.instances = []
        for root, _, fnames in sorted(os.walk(dataset_path, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                item = path
                self.instances.append(item)

        nncf_logger.info('The dataloader is built with the data located on  {}'.format(dataset_path))

    def __getitem__(self, item):
        filename = self.instances[item]
        sample = np.load(filename, allow_pickle=True)
        tensor, target = sample
        # tensor = tensor.cpu().detach().numpy()
        return tensor, target

    def __len__(self):
        return len(self.dataset)


def create_dataloader_from_imagenet_torch_dataset(dataset_dir: str,
                                                  input_shape: List[int],
                                                  mean=(0.485, 0.456, 0.406),
                                                  std=(0.229, 0.224, 0.225),
                                                  crop_ratio=0.875,
                                                  batch_size: int = 1,
                                                  shuffle: bool = True):
    import torchvision
    from torchvision import transforms
    image_size = [input_shape[-2], input_shape[-1]]
    size = int(image_size[0] / crop_ratio)
    normalize = transforms.Normalize(mean=mean,
                                     std=std)
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])
    # The best practise is to use validation part of dataset for calibration (aligning with POT)
    initialization_dataset = torchvision.datasets.ImageFolder(os.path.join(dataset_dir), transform)
    return ImageNetDataLoader(initialization_dataset, batch_size, shuffle)


def binarize_imagenet_dataset(dataloader: ImageNetDataLoader):
    for i, sample in enumerate(dataloader):
        np.save('/home/aleksei/tmp/imagenet_binary/' + str(i), sample)


def create_binarized_imagenet_dataset(dataset_path):
    return BinarizedImageNetDataLoader(dataset_path, 1, True)
