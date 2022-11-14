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

import os
from typing import List
from typing import Optional

import numpy as np
import onnx
from nncf.experimental.onnx.tensor import ONNXNNCFTensor

from nncf.experimental.post_training.compression_builder import CompressionBuilder
from nncf.experimental.post_training.algorithms.quantization import PostTrainingQuantization
from nncf.experimental.post_training.algorithms.quantization import PostTrainingQuantizationParameters
from nncf.common.utils.logger import logger as nncf_logger

from openvino.tools.accuracy_checker.config import ConfigReader
from openvino.tools.accuracy_checker.argparser import build_arguments_parser
from openvino.tools.accuracy_checker.dataset import Dataset
from openvino.tools.accuracy_checker.evaluators import ModelEvaluator

# pylint: disable=unused-import
# This import need to register custom Conerter
from tests.onnx.benchmarking.accuracy_checker import MSCocoSegmentationToVOCConverter
from nncf.experimental.post_training.api import dataset as ptq_api_dataset

from tests.onnx.quantization.common import find_ignored_scopes
from tests.onnx.utils import convert_opset_version


# pylint: disable=redefined-outer-name


class OpenVINOAccuracyCheckerDataset(ptq_api_dataset.Dataset):
    def __init__(self, model_evaluator: ModelEvaluator, batch_size: int, shuffle: bool, has_batch_dim: bool = True):
        super().__init__(batch_size, shuffle)
        self.model_evaluator = model_evaluator
        self.has_batch_dim = has_batch_dim

    def __getitem__(self, item):
        _, batch_annotation, batch_input, _ = self.model_evaluator.dataset[item]
        filled_inputs, _, _ = self.model_evaluator._get_batch_input(
            batch_annotation, batch_input)

        if len(filled_inputs) == 1:
            return {
                k: ONNXNNCFTensor(np.squeeze(v, axis=0))
                if self.has_batch_dim else ONNXNNCFTensor(v)
                for k, v in filled_inputs[0].items()
            }

        raise Exception("len(filled_inputs) should be one.")

    def __len__(self):
        return len(self.model_evaluator.dataset)


def run(onnx_model_path: str, output_model_path: str, dataset: Dataset,
        ignored_scopes: Optional[List[str]] = None,
        disallowed_op_types: Optional[List[str]] = None,
        convert_model_opset: bool = True):
    num_init_samples = len(dataset)

    nncf_logger.info("Post-Training Quantization Parameters:")
    onnx.checker.check_model(onnx_model_path)
    original_model = onnx.load(onnx_model_path)
    nncf_logger.info(f"The model is loaded from {onnx_model_path}")
    if ignored_scopes is None:
        ignored_scopes = []
    if disallowed_op_types is not None:
        ignored_scopes += find_ignored_scopes(disallowed_op_types, original_model)
    nncf_logger.info(f"  number of samples: {num_init_samples}")
    nncf_logger.info(f"  ignored_scopes: {ignored_scopes}")

    # Step 0: Convert model opset
    model = convert_opset_version(original_model) if convert_model_opset else original_model

    # Step 1: Create a pipeline of compression algorithms.
    builder = CompressionBuilder()

    # Step 2: Create the quantization algorithm and add to the builder.
    quantization_parameters = PostTrainingQuantizationParameters(
        number_samples=num_init_samples,
        ignored_scopes=ignored_scopes
    )
    quantization = PostTrainingQuantization(quantization_parameters)
    builder.add_algorithm(quantization)

    # Step 4: Execute the pipeline.
    nncf_logger.info("Post-Training Quantization has just started!")
    quantized_model = builder.apply(model, dataset)

    # Step 5: Save the quantized model.
    onnx.save(quantized_model, output_model_path)
    nncf_logger.info(
        "The quantized model is saved on {}".format(output_model_path))

    onnx.checker.check_model(output_model_path)


if __name__ == '__main__':
    parser = build_arguments_parser()
    parser.add_argument("--output-model-dir", "-o", required=True,
                        help="Directory path to save output quantized ONNX model", type=str)
    args = parser.parse_args()
    config, mode = ConfigReader.merge(args)

    assert mode == "models"
    for config_entry in config[mode]:
        model_evaluator = ModelEvaluator.from_configs(config_entry)
        assert "datasets" in config_entry
        assert len(config_entry["datasets"]
                   ) == 1, "Config should have one dataset."

        if config_entry.get("no_ptq", False):
            continue

        ignored_scopes = config_entry.get("ignored_scopes", None)
        disallowed_op_types = config_entry.get("disallowed_op_types", None)
        has_batch_dim = config_entry.get("has_batch_dim", True)
        convert_model_opset = config_entry.get("convert_opset_version", True)

        dataset_config = config_entry["datasets"][0]
        dataset = OpenVINOAccuracyCheckerDataset(
            model_evaluator, batch_size=1, shuffle=True, has_batch_dim=has_batch_dim)

        assert "launchers" in config_entry
        assert len(config_entry["launchers"]) == 1

        onnx_model_path = config_entry["launchers"][0]["model"]

        fname = onnx_model_path.stem
        output_model_path = os.path.join(
            args.output_model_dir, fname + "-quantized.onnx")

        onnx_model_path = str(onnx_model_path)

        run(onnx_model_path,
            output_model_path,
            dataset,
            ignored_scopes,
            disallowed_op_types,
            convert_model_opset)
