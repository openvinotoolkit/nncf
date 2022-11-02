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

import torch
# from nncf.quantization.dataset import PTQDataset
from examples.torch.semantic_segmentation.datasets.camvid import CamVid
from examples.torch.semantic_segmentation.datasets.mapillary import Mapillary
from nncf import Dataset
from nncf.common.utils.logger import logger as nncf_logger
from nncf.experimental.onnx.tensor import ONNXNNCFTensor


class SegmentationDataLoader():
    def __init__(self, dataset, batch_size, shuffle, input_name):
        super().__init__(batch_size, shuffle)
        self.dataset = dataset
        self.input_name = input_name
        nncf_logger.info(
            'The dataset is built with the data located on  {}'.format(dataset.root_dir))

    def __getitem__(self, item):
        tensor, target = self.dataset[item]
        tensor = tensor.cpu().detach().numpy()
        return {self.input_name: ONNXNNCFTensor(tensor), "targets": ONNXNNCFTensor(target)}

    def __len__(self):
        return len(self.dataset)


def create_dataset_from_segmentation_torch_dataset(dataset_name: str,
                                                   dataset_dir: str,
                                                   input_name: str,
                                                   input_shape: List[int]):
    from examples.torch.semantic_segmentation.utils.transforms import (
        Compose, Normalize, Resize, ToTensor)

    def transform_fn(data_item):
        tensor, target = data_item
        tensor = tensor.cpu().detach().numpy()
        return {input_name: ONNXNNCFTensor(tensor), 'targets': ONNXNNCFTensor(target)}

    if dataset_name.lower() == 'mapillary':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        dataset_class = Mapillary
    elif dataset_name.lower() == 'camvid':
        mean = (0.39068785, 0.40521392, 0.41434407)
        std = (0.29652068, 0.30514979, 0.30080369)
        dataset_class = CamVid
    else:
        raise RuntimeError('The dataset is not supported')

    image_size = [input_shape[-2], input_shape[-1]]
    transform = Compose([
        Resize(image_size),
        ToTensor(),
        Normalize(mean, std),
    ])
    initialization_dataset = dataset_class(
        dataset_dir, 'val', transforms=transform)
    dataloader = torch.utils.data.DataLoader(initialization_dataset,
                                             batch_size=1,
                                             shuffle=True)
    return Dataset(dataloader, transform_fn)
