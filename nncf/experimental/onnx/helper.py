import os

from typing import List

import torchvision
from torchvision import transforms

from nncf.experimental.post_training.api.dataloader import DataLoader


class ImageNetDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle):
        super().__init__(batch_size, shuffle)
        self.dataset = dataset
        print(f"The dataloader is built with the data located on  {dataset.root}")

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)


def create_dataloader_from_imagenet_torch_dataset(dataset_dir, input_shape: List[int], batch_size: int = 1,
                                                  shuffle: bool = True):
    image_size = [input_shape[-2], input_shape[-1]]
    size = int(image_size[0] / 0.875)
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])
    initialization_dataset = torchvision.datasets.ImageFolder(os.path.join(dataset_dir, 'train'), transform)
    return ImageNetDataLoader(initialization_dataset, batch_size, shuffle)
