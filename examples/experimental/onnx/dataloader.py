import os
from typing import Union

import torch
from torchvision import datasets
from torchvision import transforms


def create_train_dataloader(dataset_dir: Union[str, bytes, os.PathLike], input_shape,
                            batch_size=1, num_workers=4) -> torch.utils.data.DataLoader:
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))
    image_size = [input_shape[-2], input_shape[-1]]
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = datasets.ImageFolder(os.path.join(dataset_dir, 'train'), transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers)
    return train_loader
