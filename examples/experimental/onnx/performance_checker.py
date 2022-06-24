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

from tqdm import tqdm
from nncf.common.utils.logger import logger as nncf_logger

from openvino.tools.accuracy_checker.config import ConfigReader
from openvino.tools.accuracy_checker.argparser import build_arguments_parser
from openvino.tools.accuracy_checker.dataset import Dataset
from openvino.tools.accuracy_checker.evaluators import ModelEvaluator

import nncf.experimental.post_training.api.dataset as ptq_api_dataset
from nncf.experimental.onnx.engine import ONNXEngine
from nncf.experimental.onnx.samplers import create_onnx_sampler
from time import time
import pandas as pd


class OpenVINOAccuracyCheckerDataset(ptq_api_dataset.Dataset):
    def __init__(self, evaluator: ModelEvaluator, batch_size, shuffle):
        super().__init__(batch_size, shuffle)
        self.model_evaluator = evaluator

    def __getitem__(self, item):
        _, batch_annotation, batch_input, _ = self.model_evaluator.dataset[item]
        filled_inputs, _, _ = self.model_evaluator._get_batch_input(
            batch_annotation, batch_input)

        assert len(filled_inputs) == 1
        dummy_target = 0

        for _, v in filled_inputs[0].items():
            return np.squeeze(v, axis=0), dummy_target

        raise RuntimeError("filled_inputs has no value.")

    def __len__(self):
        return len(self.model_evaluator.dataset)


def run(onnx_model_path: str, output_file_path: str, dataset: Dataset,
        ignored_scopes: Optional[List[str]] = None, evaluate: Optional[bool] = False):

    num_init_samples = len(dataset)

    nncf_logger.info("Post-Training Quantization Parameters:")
    nncf_logger.info(f"  number of samples: {num_init_samples}")
    nncf_logger.info(f"  ignored_scopes: {ignored_scopes}")
    onnx.checker.check_model(onnx_model_path)
    original_model = onnx.load(onnx_model_path)
    nncf_logger.info(f"The model is loaded from {onnx_model_path}")

    onnx.checker.check_model(original_model)

    engine = ONNXEngine()
    sampler = create_onnx_sampler(dataset, range(len(dataset)))

    engine.rt_session_options['providers'] = ["OpenVINOExecutionProvider"]
    engine.set_model(original_model)
    engine.set_sampler(sampler)

    elapsed_times = []

    for input_data, _ in tqdm(sampler):
        start_time = time()
        engine.infer(input_data)
        elapsed_times += [1000.0 * (time() - start_time)]

    elapsed_times = np.array(elapsed_times)

    model_name, _ = os.path.splitext(os.path.basename(onnx_model_path))

    df = pd.DataFrame({
        "model_name": [model_name],
        "latency_mean": [np.mean(elapsed_times)],
        "latency_std": [np.std(elapsed_times)]
    })

    if os.path.exists(output_file_path):
        df.to_csv(output_file_path, header=False, mode="a", index=False)
    else:
        df.to_csv(output_file_path, header=True, mode="w", index=False)


if __name__ == '__main__':
    parser = build_arguments_parser()
    parser.add_argument("--output-file-path", "-o",
                        help="Directory path to save output quantized ONNX model", type=str)
    args = parser.parse_args()
    config, mode = ConfigReader.merge(args)

    assert mode == "models"
    for config_entry in config[mode]:
        model_evaluator = ModelEvaluator.from_configs(config_entry)
        assert "datasets" in config_entry
        assert len(config_entry["datasets"]
                   ) == 1, "Config should have one dataset."

        dataset_config = config_entry["datasets"][0]

        assert "launchers" in config_entry
        assert len(config_entry["launchers"]) == 1

        run(onnx_model_path=str(config_entry["launchers"][0]["model"]),
            output_file_path=args.output_file_path,
            dataset=OpenVINOAccuracyCheckerDataset(model_evaluator, batch_size=1, shuffle=True))
