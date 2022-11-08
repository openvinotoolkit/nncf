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

from typing import List, Optional

import onnx

from nncf.quantization.compression_builder import CompressionBuilder
from nncf.quantization.algorithms import DefaultQuantization
from nncf.quantization.algorithms import DefaultQuantizationParameters
from nncf.experimental.onnx.common import infer_input_shape
from examples.experimental.onnx.semantic_segmentation.segmentation_dataset import create_dataloader, create_dataset
from examples.experimental.onnx.common.argparser import get_common_argument_parser


def run(onnx_model_path: str, output_model_path: str, dataset_name: str,
        dataset_path: str, num_init_samples: int,
        input_shape: Optional[List[int]] = None, input_name: Optional[str] = None,
        ignored_scopes: Optional[List[str]] = None):
    print("Post-Training Quantization Parameters:")
    print("  number of samples: ", num_init_samples)
    print("  ignored_scopes: ", ignored_scopes)
    onnx.checker.check_model(onnx_model_path)
    original_model = onnx.load(onnx_model_path)
    print(f"The model is loaded from {onnx_model_path}")

    assert input_shape or input_name, "Either input_shape or input_name must be set."

    input_shape, input_name = infer_input_shape(original_model, input_shape, input_name)

    # Step 1: Initialize the dataloader & dataset.
    dataloader = create_dataloader(dataset_name, dataset_path, input_shape)
    dataset = create_dataset(dataloader, input_name)

    # Step 2: Create a pipeline of compression algorithms.
    builder = CompressionBuilder()

    # Step 3: Create the quantization algorithm and add to the builder.
    quantization_parameters = DefaultQuantizationParameters(
        number_samples=num_init_samples,
        ignored_scopes=ignored_scopes
    )
    quantization = DefaultQuantization(quantization_parameters)
    builder.add_algorithm(quantization)

    # Step 4: Execute the pipeline.
    print("Post-Training Quantization has just started!")
    quantized_model = builder.apply(original_model, dataset)

    # Step 5: Save the quantized model.
    onnx.save(quantized_model, output_model_path)
    print(f"The quantized model is saved on {output_model_path}")

    onnx.checker.check_model(output_model_path)


if __name__ == '__main__':
    parser = get_common_argument_parser()

    parser.add_argument("--dataset_name",
                        help="CamVid or Mapillary",
                        type=str)

    args = parser.parse_args()

    run(args.onnx_model_path,
        args.output_model_path,
        args.dataset_name,
        args.data,
        args.init_samples,
        args.input_shape,
        args.input_name,
        args.ignored_scopes
        )
