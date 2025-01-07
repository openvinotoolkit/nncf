# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from functools import partial
from typing import List, Optional

import numpy as np
import onnx

try:
    from accuracy_checker.argparser import build_arguments_parser
    from accuracy_checker.config import ConfigReader
    from accuracy_checker.evaluators import ModelEvaluator
except ImportError:
    from openvino.tools.accuracy_checker.argparser import build_arguments_parser
    from openvino.tools.accuracy_checker.config import ConfigReader
    from openvino.tools.accuracy_checker.evaluators import ModelEvaluator

import nncf
from nncf.scopes import IgnoredScope
from tests.onnx.opset_converter import convert_opset_version

# This import need to register custom Conerter
from tests.onnx.quantization.common import find_ignored_scopes


def process_fn(data_item, model_evaluator: ModelEvaluator, has_batch_dim: Optional[bool] = False):
    _, batch_annotation, batch_input, _ = data_item
    filled_inputs, _, _ = model_evaluator._get_batch_input(batch_annotation, batch_input)

    if len(filled_inputs) == 1:
        return {k: np.squeeze(v, axis=0) if has_batch_dim else v for k, v in filled_inputs[0].items()}

    raise Exception("len(filled_inputs) should be one.")


def run(
    onnx_model_path: str,
    output_model_path: str,
    dataset: nncf.Dataset,
    num_init_samples: int,
    ignored_scopes: Optional[List[str]] = None,
    disallowed_op_types: Optional[List[str]] = None,
    convert_model_opset: bool = True,
):
    print("Post-Training Quantization Parameters:")
    onnx.checker.check_model(onnx_model_path)
    original_model = onnx.load(onnx_model_path)
    print(f"The model is loaded from {onnx_model_path}")
    print(f"  number of samples: {num_init_samples}")
    # TODO(kshpv):: add support of types ignored_scopes
    if ignored_scopes is None:
        ignored_scopes = []
    if disallowed_op_types is not None:
        ignored_scopes += find_ignored_scopes(disallowed_op_types, original_model)
    print(f"  ignored_scopes: {ignored_scopes}")

    # Convert the model opset if needed.
    model = convert_opset_version(original_model) if convert_model_opset else original_model
    # Execute the pipeline.
    quantized_model = nncf.quantize(
        model, dataset, subset_size=num_init_samples, ignored_scope=IgnoredScope(names=ignored_scopes)
    )
    # Save the quantized model.
    onnx.save(quantized_model, output_model_path)
    print("The quantized model is saved to: {}".format(output_model_path))


if __name__ == "__main__":
    parser = build_arguments_parser()
    parser.add_argument(
        "--output-model-dir", "-o", required=True, help="Directory path to save output quantized ONNX model", type=str
    )
    args = parser.parse_args()
    args.target_framework = "onnx_runtime"
    config, mode = ConfigReader.merge(args)

    assert mode == "models"
    for config_entry in config[mode]:
        model_evaluator = ModelEvaluator.from_configs(config_entry)
        assert "datasets" in config_entry
        assert len(config_entry["datasets"]) == 1, "Config should have one dataset."

        if config_entry.get("no_ptq", False):
            continue

        ignored_scopes = config_entry.get("ignored_scopes", None)
        disallowed_op_types = config_entry.get("disallowed_op_types", None)
        has_batch_dim = config_entry.get("has_batch_dim", False)
        convert_model_opset = config_entry.get("convert_opset_version", True)

        dataset_config = config_entry["datasets"][0]
        options = {"model_evaluator": model_evaluator, "has_batch_dim": has_batch_dim}
        transform_fn = partial(process_fn, **options)
        dataset = nncf.Dataset(model_evaluator.dataset, transform_fn)

        assert "launchers" in config_entry
        assert len(config_entry["launchers"]) == 1

        onnx_model_path = config_entry["launchers"][0]["model"]

        fname = onnx_model_path.stem
        output_model_path = os.path.join(args.output_model_dir, fname + "-quantized.onnx")

        onnx_model_path = str(onnx_model_path)

        num_init_samples = len(model_evaluator.dataset)

        run(
            onnx_model_path,
            output_model_path,
            dataset,
            num_init_samples,
            ignored_scopes,
            disallowed_op_types,
            convert_model_opset,
        )
