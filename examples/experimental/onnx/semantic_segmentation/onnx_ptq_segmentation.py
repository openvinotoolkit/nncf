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

import argparse

from typing import List

import onnx

from nncf.experimental.post_training.compression_builder import CompressionBuilder
from nncf.experimental.post_training.algorithms.quantization import PostTrainingQuantization
from nncf.experimental.post_training.algorithms.quantization import PostTrainingQuantizationParameters
from nncf.experimental.onnx.datasets.segmentation_dataset import create_dataset_from_segmentation_torch_dataset


def run(onnx_model_path: str, output_model_path: str, dataset_name: str,
        dataset_path: str, num_init_samples: int,
        input_shape: List[int], ignored_scopes: List[str] = None):
    print("Post-Training Quantization Parameters:")
    print("  number of samples: ", num_init_samples)
    print("  ignored_scopes: ", ignored_scopes)
    onnx.checker.check_model(onnx_model_path)
    original_model = onnx.load(onnx_model_path)
    print(f"The model is loaded from {onnx_model_path}")

    # Step 1: Initialize the data loader.
    dataloader = create_dataset_from_segmentation_torch_dataset(
        dataset_name, dataset_path, input_shape)

    # Step 2: Create a pipeline of compression algorithms.
    builder = CompressionBuilder()

    # Step 3: Create the quantization algorithm and add to the builder.
    quantization_parameters = PostTrainingQuantizationParameters(
        number_samples=num_init_samples,
        ignored_scopes=ignored_scopes
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
    parser.add_argument("--onnx_model_path", "-m",
                        help="Path to ONNX model", type=str)
    parser.add_argument("--output_model_path", "-o",
                        help="Path to output quantized ONNX model", type=str)
    parser.add_argument("--dataset_name",
                        help="CamVid or Mapillary",
                        type=str)
    parser.add_argument("--data",
                        help="Path to dataset",
                        type=str)
    parser.add_argument("--input_shape", help="Model's input shape",
                        nargs="+", type=int, default=[1, 3, 768, 960])
    parser.add_argument(
        "--init_samples", help="Number of initialization samples", type=int, default=300)
    parser.add_argument(
        "--ignored_scopes", help="Ignored operations ot quantize", nargs="+", default=None)
    args = parser.parse_args()
    run(args.onnx_model_path,
        args.output_model_path,
        args.dataset_name,
        args.data,
        args.init_samples,
        args.input_shape,
        args.ignored_scopes
        )
