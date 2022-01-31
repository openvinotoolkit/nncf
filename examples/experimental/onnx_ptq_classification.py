import os
from typing import List
from typing import Union

import torch
from torch.utils.data.dataloader import DataLoader as TorchDataLoader
import torchvision
from torchvision import transforms

from nncf.experimental.post_training.api.compression_builder import CompressionBuilder
from nncf.experimental.post_training.quantization.algorithm import PostTrainingQuantization
from nncf.experimental.post_training.quantization.algorithm import DEFAULT
from nncf.experimental.post_training.api.dataloader import DataLoader


class MyDataLoader(DataLoader, TorchDataLoader):
    def __init__(self, dataset, **attrs):
        TorchDataLoader.__init__(dataset, **attrs)


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
    train_loader = MyDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_loader


fp32_model_onnx_path = ''
int8_model_onnx_path = ''
dataset_dir = ''
input_shape = []

# Step 1: Initialize the data loader.
dataloader = create_train_dataloader(dataset_dir=dataset_dir, input_shape=input_shape)

# Step 3: Create a pipeline of compression algorithms.
builder = CompressionBuilder()
quantization = PostTrainingQuantization(DEFAULT)
builder.add_algorithm(quantization)
# Step 4: Execute the pipeline.
compressed_model = builder.apply(fp32_model_onnx_path, dataloader)
# Step 5: Export the compressed model.
compressed_model.export(int8_model_onnx_path)

# samples for many backends
