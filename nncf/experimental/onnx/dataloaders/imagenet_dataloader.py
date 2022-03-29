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
