import argparse
import os

from typing import List

import torchvision
from torchvision import transforms

import onnx

from nncf.experimental.post_training.api.dataloader import DataLoader
from nncf.experimental.post_training.compression_builder import CompressionBuilder
from nncf.experimental.post_training.quantization.algorithm import PostTrainingQuantization

from nncf.experimental.post_training.quantization.algorithm import PostTrainingQuantizationParameters


def create_initialization_dataset(dataset_dir, input_shape: List[int]):
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
    return torchvision.datasets.ImageFolder(os.path.join(dataset_dir, 'train'), transform)


class ImageNetDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle):
        super().__init__(batch_size, shuffle)
        self.dataset = dataset
        print(f"The dataloader is built with the data located on  {dataset.root}")

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)


def run(onnx_model_path: str, output_model_path: str,
        dataset_path: str, batch_size: int, shuffle: bool, num_init_samples: int,
        input_shape: List[int], ignored_scopes: List[str] = None):
    print("Post-Training Quantization Parameters:")
    print("  number of samples: ", num_init_samples)
    print("  ignored_scopes: ", ignored_scopes)
    onnx.checker.check_model(onnx_model_path)
    original_model = onnx.load(onnx_model_path)
    print(f"The model is loaded from {onnx_model_path}")

    # Step 1: Initialize the data loader.
    initialization_dataset = create_initialization_dataset(dataset_path, input_shape)
    dataloader = ImageNetDataLoader(initialization_dataset, batch_size=batch_size, shuffle=shuffle)

    # Step 2: Create a pipeline of compression algorithms.
    builder = CompressionBuilder()

    # Step 3: Create the quantization algorithm and add to the builder.
    quantization_parameters = PostTrainingQuantizationParameters(
        number_samples=num_init_samples
    )
    quantization = PostTrainingQuantization(quantization_parameters)
    builder.add_algorithm(quantization)

    # Step 4: Execute the pipeline.
    print("Post-Training Quantization has just started!")
    quantized_model = builder.apply(original_model, dataloader)
    # Step 5: Save the quantized model.
    onnx.save(quantized_model, output_model_path)
    print(f"The quantized model is saved on {output_model_path}")
    onnx.checker.check_model(output_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_model_path", "-m", help="Path to ONNX model", type=str)
    parser.add_argument("--output_model_path", "-o", help="Path to output quantized ONNX model", type=str)
    parser.add_argument("--data", help="Path to ImageNet dataset", type=str)
    parser.add_argument("--batch_size", help="Batch size for initialization", default=1)
    parser.add_argument("--shuffle", help="Whether to shuffle dataset for initialization", default=True)
    parser.add_argument("--input_shape", help="Model's input shape", nargs="+", type=int, default=[1, 3, 224, 224])
    parser.add_argument("--init_samples", help="Number of initialization samples", type=int, default=100)
    parser.add_argument("--ignored_scopes", help="Ignored operations ot quantize", nargs="+", default=None)
    args = parser.parse_args()
    run(args.onnx_model_path,
        args.output_model_path,
        args.data,
        args.batch_size,
        args.shuffle,
        args.init_samples,
        args.input_shape,
        args.ignored_scopes
        )
