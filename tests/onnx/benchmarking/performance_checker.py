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
from functools import partial
from time import time
from typing import List, Optional

import numpy as np
import onnx
import pandas as pd
from openvino.tools.accuracy_checker.argparser import build_arguments_parser
from openvino.tools.accuracy_checker.config import ConfigReader
from openvino.tools.accuracy_checker.evaluators import ModelEvaluator
from tqdm import tqdm

import nncf
from nncf.common.utils.logger import logger as nncf_logger
from nncf.experimental.onnx.engine import ONNXEngine
from nncf.experimental.onnx.tensor import ONNXNNCFTensor

#pylint: disable=redefined-outer-name,protected-access


def process_fn(data_item, model_evaluator: ModelEvaluator, has_batch_dim: Optional[bool] = False):
    _, batch_annotation, batch_input, _ = data_item
    filled_inputs, _, _ = model_evaluator._get_batch_input(batch_annotation, batch_input)

    if len(filled_inputs) == 1:
        return {k: ONNXNNCFTensor(np.squeeze(v, axis=0))
                if has_batch_dim else ONNXNNCFTensor(v) for k, v in filled_inputs[0].items()}

    raise Exception("len(filled_inputs) should be one.")


def run(onnx_model_path: str, output_file_path: str, dataset: nncf.Dataset,
        model_name: str, num_init_samples: int,
        ignored_scopes: Optional[List[str]] = None):

    nncf_logger.info("Post-Training Quantization Parameters:")
    nncf_logger.info(f"  number of samples: {num_init_samples}")
    nncf_logger.info(f"  ignored_scopes: {ignored_scopes}")
    onnx.checker.check_model(onnx_model_path)
    original_model = onnx.load(onnx_model_path)
    nncf_logger.info(f"The model is loaded from {onnx_model_path}")

    onnx.checker.check_model(original_model)

    engine = ONNXEngine()

    engine.rt_session_options['providers'] = ["OpenVINOExecutionProvider"]
    engine.set_model(original_model)

    elapsed_times = []

    indices_list = list(range(num_init_samples))

    for input_data in tqdm(dataset.get_inference_data(indices_list), total=num_init_samples):
        start_time = time()
        engine.infer(input_data)
        elapsed_times += [1000.0 * (time() - start_time)]

    elapsed_times = np.array(elapsed_times)

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
        has_batch_dim = config_entry.get("has_batch_dim", False)

        dataset_config = config_entry["datasets"][0]

        assert "launchers" in config_entry
        assert len(config_entry["launchers"]) == 1

        options = {
            'model_evaluator': model_evaluator,
            'has_batch_dim': has_batch_dim
        }
        transform_fn = partial(process_fn, **options)
        dataset = nncf.Dataset(model_evaluator.dataset, transform_fn)

        num_init_samples = len(model_evaluator.dataset)

        run(onnx_model_path=str(config_entry["launchers"][0]["model"]),
            output_file_path=args.csv_result,
            dataset=dataset,
            model_name=config_entry["name"],
            num_init_samples=num_init_samples,
            ignored_scopes=ignored_scopes)
