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
import numpy as np

from typing import List

import onnx

from nncf.experimental.post_training.compression_builder import CompressionBuilder
from nncf.experimental.post_training.algorithms.quantization import PostTrainingQuantization
from nncf.experimental.post_training.algorithms.quantization import PostTrainingQuantizationParameters
from nncf.experimental.onnx.dataloaders.imagenet_data_loader import create_dataloader_from_imagenet_torch_dataset
from nncf.experimental.post_training.api.metric import Metric


class Accuracy(Metric):

    # Required methods
    def __init__(self, top_k=1):
        super().__init__()
        self._top_k = top_k
        self._name = 'accuracy@top{}'.format(self._top_k)
        self._matches = []

    @property
    def avg_value(self):
        """ Returns accuracy metric value for all model outputs. """
        return {self._name: np.ravel(self._matches).mean()}

    def update(self, output, target):
        """ Updates prediction matches.
        :param output: model output
        :param target: annotations
        """
        if len(output) > 1:
            raise Exception('The accuracy metric cannot be calculated '
                            'for a model with multiple outputs')
        predictions = np.argsort(output[0], axis=1)[:, -self._top_k:]
        match = [float(t in predictions[i]) for i, t in enumerate(target)]

        self._matches.append(match)

    def reset(self):
        """ Resets collected matches """
        self._matches = []


def run(onnx_model_path: str, output_model_path: str,
        dataset_path: str, batch_size: int, shuffle: bool, num_init_samples: int,
        input_shape: List[int], ignored_scopes: List[str] = None, evaluate: bool = False):
    print("Post-Training Quantization Parameters:")
    print("  number of samples: ", num_init_samples)
    print("  ignored_scopes: ", ignored_scopes)
    onnx.checker.check_model(onnx_model_path)
    original_model = onnx.load(onnx_model_path)
    print(f"The model is loaded from {onnx_model_path}")

    # Step 1: Initialize the data loader and metric (if need)
    dataloader = create_dataloader_from_imagenet_torch_dataset(dataset_path, input_shape,
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
    print("Post-Training Quantization has just started!")
    quantized_model = builder.apply(original_model, dataloader)

    # Step 6: Save the quantized model.
    onnx.save(quantized_model, output_model_path)
    print(f"The quantized model is saved on {output_model_path}")

    onnx.checker.check_model(output_model_path)

    # Step 7: (Optional) Validate the quantized model.
    if evaluate:
        print("Validating of the quantized model has started")
        metrics = builder.evaluate(quantized_model, metric, dataloader)
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_model_path", "-m", help="Path to ONNX model", type=str)
    parser.add_argument("--output_model_path", "-o", help="Path to output quantized ONNX model", type=str)
    parser.add_argument("--data",
                        help="Path to ImageNet validation data in the ImageFolder torchvision format "
                             "(Please, take a look at torchvision.datasets.ImageFolder)",
                        type=str)
    parser.add_argument("--batch_size", help="Batch size for initialization", default=1)
    parser.add_argument("--shuffle", help="Whether to shuffle dataset for initialization", default=False)
    parser.add_argument("--input_shape", help="Model's input shape", nargs="+", type=int, default=[1, 3, 224, 224])
    parser.add_argument("--init_samples", help="Number of initialization samples", type=int, default=300)
    parser.add_argument("--ignored_scopes", help="Ignored operations ot quantize", nargs="+", default=None)
    parser.add_argument("--evaluate", help="Whether is a validation of the final model need", default=False)
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
