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
from typing import Optional

import numpy as np
import onnx

from nncf.experimental.post_training.compression_builder import CompressionBuilder
from nncf.experimental.post_training.algorithms.quantization import PostTrainingQuantization
from nncf.experimental.post_training.algorithms.quantization import PostTrainingQuantizationParameters
from nncf.experimental.onnx.datasets.imagenet_dataset import create_imagenet_torch_dataset, infer_input_shape
from nncf.experimental.post_training.api.metric import Accuracy
from nncf.common.utils.logger import logger as nncf_logger


def run(onnx_model_path: str, output_model_path: str,
        dataset_path: str, batch_size: int, shuffle: bool, num_init_samples: int,
        input_shape: Optional[List[int]] = None, ignored_scopes: Optional[List[str]] = None,
        evaluate: Optional[bool] = False):
    nncf_logger.info("Post-Training Quantization Parameters:")
    nncf_logger.info("  number of samples: {}".format(nncf_logger.info))
    nncf_logger.info("  ignored_scopes: {}".format(ignored_scopes))
    onnx.checker.check_model(onnx_model_path)
    original_model = onnx.load(onnx_model_path)
    nncf_logger.info("The model is loaded from {}".format(onnx_model_path))

    if input_shape is None:
        nncf_logger.info(
            "input_shape is None. Infer input_shape from the model.")
        input_shape = infer_input_shape(original_model)

    # Step 1: Initialize the data loader and metric (if it is needed).
    dataset = create_imagenet_torch_dataset(dataset_path, input_shape,
                                            batch_size=batch_size, shuffle=shuffle)
    metric = Accuracy(top_k=1)

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
    nncf_logger.info("Post-Training Quantization has just started!")
    quantized_model = builder.apply(original_model, dataset)

    # Step 5: Save the quantized model.
    onnx.save(quantized_model, output_model_path)
    nncf_logger.info(
        "The quantized model is saved on {}".format(output_model_path))

    onnx.checker.check_model(output_model_path)

    # Step 6: (Optional) Validate the quantized model.
    if evaluate:
        nncf_logger.info("Validation of the quantized model "
                         "on the validation part of the dataset.")

        metrics = builder.evaluate(
            quantized_model, metric, dataset, outputs_transforms=lambda x: np.concatenate(x, axis=0))
        for metric_name, metric_value in metrics.items():
            nncf_logger.info("{}: {}".format(metric_name, metric_value))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_model_path", "-m",
                        help="Path to ONNX model", type=str)
    parser.add_argument("--output_model_path", "-o",
                        help="Path to output quantized ONNX model", type=str)
    parser.add_argument("--data",
                        help="Path to ImageNet validation data in the ImageFolder torchvision format "
                             "(Please, take a look at torchvision.datasets.ImageFolder)",
                        type=str)
    parser.add_argument(
        "--batch_size", help="Batch size for initialization", type=int, default=1)
    parser.add_argument(
        "--shuffle", help="Whether to shuffle dataset for initialization", default=True)
    parser.add_argument("--input_shape",
        help="Model's input shape. e.g. [1, 3, 224, 224]. If it's not given, it automatically infers input shape.",
                        nargs="+", type=int, default=None)
    parser.add_argument(
        "--init_samples", help="Number of initialization samples", type=int, default=300)
    parser.add_argument(
        "--ignored_scopes", help="Ignored operations ot quantize", nargs="+", default=None)
    parser.add_argument(
        "--evaluate", help="Run an evaluation step for the final quantized model", action="store_true")
    args = parser.parse_args()
    run(args.onnx_model_path,
        args.output_model_path,
        args.data,
        args.batch_size,
        args.shuffle,
        args.init_samples,
        args.input_shape,
        args.ignored_scopes,
        args.evaluate
        )
