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

from typing import List
from typing import Optional

import onnx
import onnxruntime as rt

from sklearn.metrics import accuracy_score
from nncf.quantization.compression_builder import CompressionBuilder
from nncf.quantization.algorithms import DefaultQuantization
from nncf.quantization.algorithms import DefaultQuantizationParameters
from nncf.experimental.onnx.common import infer_input_shape
from nncf.experimental.onnx.datasets.imagenet_dataset import create_dataloader, create_dataset
from nncf.common.utils.logger import logger as nncf_logger
from examples.experimental.onnx.common.argparser import get_common_argument_parser
from examples.experimental.onnx.classification.accuracy import Accuracy


def run(onnx_model_path: str, output_model_path: str,
        dataset_path: str, batch_size: int, shuffle: bool, num_init_samples: int,
        input_shape: Optional[List[int]] = None, input_name: Optional[str] = None,
        ignored_scopes: Optional[List[str]] = None,
        evaluate: Optional[bool] = False):
    nncf_logger.info("Post-Training Quantization Parameters:")
    nncf_logger.info("  number of samples: {}".format(num_init_samples))
    nncf_logger.info("  ignored_scopes: {}".format(ignored_scopes))
    onnx.checker.check_model(onnx_model_path)
    original_model = onnx.load(onnx_model_path)
    nncf_logger.info("The model is loaded from {}".format(onnx_model_path))

    assert input_shape or input_name, "Either input_shape or input_name must be set."

    input_shape, input_name = infer_input_shape(original_model, input_shape, input_name)

    # Step 1: Initialize the data loader, dataset and metric (if it is needed).
    dataloader = create_dataloader(dataset_path, input_shape=input_shape,
                                   batch_size=batch_size, shuffle=shuffle)
    dataset = create_dataset(dataloader, input_name)
    metric = Accuracy(top_k=1)

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
    nncf_logger.info("Post-Training Quantization has just started!")
    quantized_model = builder.apply(original_model, dataset)

    # Step 5: Save the quantized model.
    onnx.save(quantized_model, output_model_path)
    nncf_logger.info(
        "The quantized model is saved on {}".format(output_model_path))

    # Step 6: (Optional) Validate the quantized model.
    if evaluate:
        nncf_logger.info("Validation of the quantized model "
                         "on the validation part of the dataset.")
        
        logging_step = 1000

        so = rt.SessionOptions()
        so.log_severity_level = 3
        providers = ['OpenVINOExecutionProvider']
        provider_options = [{'device_type': 'CPU_FP32'}]

        sess = rt.InferenceSession(quantized_model.SerializeToString(), so, providers, provider_options)

        input_name = sess.get_inputs()[0].name
        outputs = sess.get_outputs()
        output_names = list(map(lambda output: output.name, outputs))

        for i, (images, target) in enumerate(dataloader):
            output = sess.run(output_names, {input_name: images.numpy()})[0]
            metric.update(output, target.numpy())
            if i != 0 and i % logging_step == 0:
                print("{} samples validated".format(i))
        
        metrics = metric.avg_value
        for metric_name, metric_value in metrics.items():
            nncf_logger.info("{}: {}".format(metric_name, metric_value))

    onnx.checker.check_model(output_model_path)


if __name__ == '__main__':
    parser = get_common_argument_parser()

    parser.add_argument(
        "--batch_size", help="Batch size for initialization", type=int, default=1)
    parser.add_argument(
        "--shuffle", help="Whether to shuffle dataset for initialization", default=True)
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
        args.input_name,
        args.ignored_scopes,
        args.evaluate
        )
