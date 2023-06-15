# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import sys
from argparse import ArgumentParser
from typing import Dict, Tuple

from tests.shared.paths import PROJECT_ROOT

# pylint: disable=maybe-no-member
# pylint: disable=import-error


def post_training_quantization_mobilenet_v2(example_root_dir: str) -> Dict[str, float]:
    sys.path.append(example_root_dir)
    import main as mobilenet_v2

    metrics = {
        "fp32_top1": float(mobilenet_v2.fp32_top1),
        "int8_top1": float(mobilenet_v2.int8_top1),
        "accuracy_drop": float(mobilenet_v2.fp32_top1 - mobilenet_v2.int8_top1),
        "fp32_fps": mobilenet_v2.fp32_fps,
        "int8_fps": mobilenet_v2.int8_fps,
        "performance_speed_up": mobilenet_v2.int8_fps / mobilenet_v2.fp32_fps,
    }

    if hasattr(mobilenet_v2, "fp32_model_size") and hasattr(mobilenet_v2, "int8_model_size"):
        metrics["fp32_model_size"] = mobilenet_v2.fp32_model_size
        metrics["int8_model_size"] = mobilenet_v2.int8_model_size
        metrics["model_compression_rate"] = mobilenet_v2.fp32_model_size / mobilenet_v2.int8_model_size

    return metrics


def post_training_quantization_onnx_mobilenet_v2() -> Dict[str, float]:
    example_root = str(PROJECT_ROOT / "examples" / "post_training_quantization" / "onnx" / "mobilenet_v2")
    return post_training_quantization_mobilenet_v2(example_root)


def post_training_quantization_openvino_mobilenet_v2_quantize() -> Dict[str, float]:
    example_root = str(PROJECT_ROOT / "examples" / "post_training_quantization" / "openvino" / "mobilenet_v2")
    return post_training_quantization_mobilenet_v2(example_root)


def post_training_quantization_tensorflow_mobilenet_v2() -> Dict[str, float]:
    example_root = str(PROJECT_ROOT / "examples" / "post_training_quantization" / "tensorflow" / "mobilenet_v2")
    return post_training_quantization_mobilenet_v2(example_root)


def post_training_quantization_torch_mobilenet_v2() -> Dict[str, float]:
    example_root = str(PROJECT_ROOT / "examples" / "post_training_quantization" / "torch" / "mobilenet_v2")
    return post_training_quantization_mobilenet_v2(example_root)


def format_results(results: Tuple[float]) -> Dict[str, float]:
    return {
        "fp32_mAP": results[0],
        "int8_mAP": results[1],
        "accuracy_drop": results[0] - results[1],
        "fp32_fps": results[2],
        "int8_fps": results[3],
        "performance_speed_up": results[3] / results[2],
    }


def post_training_quantization_openvino_yolo8_quantize() -> Dict[str, float]:
    from examples.post_training_quantization.openvino.yolov8.main import main as yolo8_main

    results = yolo8_main()

    return format_results(results)


def post_training_quantization_openvino_yolo8_quantize_with_accuracy_control() -> Dict[str, float]:
    from examples.post_training_quantization.openvino.yolov8_quantize_with_accuracy_control.main import (
        main as yolo8_main,
    )

    results = yolo8_main()

    return format_results(results)


def post_training_quantization_openvino_anomaly_stfpm_quantize_with_accuracy_control() -> Dict[str, float]:
    sys.path.append(
        str(
            PROJECT_ROOT
            / "examples"
            / "post_training_quantization"
            / "openvino"
            / "anomaly_stfpm_quantize_with_accuracy_control"
        )
    )
    import main as stfpm

    return {
        "fp32_top1": float(stfpm.fp32_top1),
        "int8_top1": float(stfpm.int8_top1),
        "accuracy_drop": float(stfpm.fp32_top1 - stfpm.int8_top1),
        "fp32_fps": stfpm.fp32_fps,
        "int8_fps": stfpm.int8_fps,
        "performance_speed_up": stfpm.int8_fps / stfpm.fp32_fps,
        "fp32_model_size": stfpm.fp32_size,
        "int8_model_size": stfpm.int8_size,
        "model_compression_rate": stfpm.fp32_size / stfpm.int8_size,
    }


def post_training_quantization_torch_ssd300_vgg16() -> Dict[str, float]:
    from examples.post_training_quantization.torch.ssd300_vgg16.main import main as ssd300_vgg16_main

    results = ssd300_vgg16_main()

    return {
        "fp32_mAP": float(results[0]),
        "int8_mAP": float(results[1]),
        "accuracy_drop": float(results[0] - results[1]),
        "fp32_fps": results[2],
        "int8_fps": results[3],
        "performance_speed_up": results[3] / results[2],
        "fp32_model_size": results[4],
        "int8_model_size": results[5],
        "model_compression_rate": results[4] / results[5],
    }


def main(argv):
    parser = ArgumentParser()
    parser.add_argument("--name", help="Example name", required=True)
    parser.add_argument("-o", "--output", help="Path to the json file to save example metrics", required=True)
    args = parser.parse_args(args=argv)

    metrics = globals()[args.name]()

    with open(args.output, "w", encoding="utf8") as json_file:
        return json.dump(metrics, json_file)


if __name__ == "__main__":
    self_argv = sys.argv[1:]
    sys.argv = sys.argv[:1]
    main(self_argv)
