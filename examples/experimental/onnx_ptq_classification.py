import argparse

from typing import List

import onnx

from nncf.experimental.post_training.compression_builder import CompressionBuilder
from nncf.experimental.post_training.quantization.algorithm import PostTrainingQuantization

from nncf.experimental.post_training.quantization.algorithm import PostTrainingQuantizationParameters
from nncf.experimental.post_training.quantization.parameters import GRANULARITY
from nncf.experimental.onnx.helper import create_dataloader_from_imagenet_torch_dataset


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
    dataloader = create_dataloader_from_imagenet_torch_dataset(dataset_path, input_shape,
                                                               batch_size=batch_size, shuffle=shuffle)

    # Step 2: Create a pipeline of compression algorithms.
    builder = CompressionBuilder()

    # Step 3: Create the quantization algorithm and add to the builder.
    quantization_parameters = PostTrainingQuantizationParameters(
        weight_granularity=GRANULARITY.PERTENSOR,
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

    # Step 6:


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
