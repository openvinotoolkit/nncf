import os
from typing import List
from typing import Union

import torch
import torchvision
from torchvision import transforms


def create_train_dataloader(dataset_dir: Union[str, bytes, os.PathLike], input_shape: List[int],
                            batch_size: int = 1, num_workers: int = 4) -> torch.utils.data.DataLoader:
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
    train_dataset = torchvision.datasets.ImageFolder(os.path.join(dataset_dir, 'train'), transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers)
    return train_loader
