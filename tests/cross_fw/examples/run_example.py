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

import json
import os
import sys
import tarfile
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Tuple, Union

from tests.cross_fw.shared.paths import PROJECT_ROOT


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
    import tensorflow_datasets as tfds

    tfds.display_progress_bar(enable=False)

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


def post_training_quantization_onnx_yolo8_quantize_with_accuracy_control() -> Dict[str, float]:
    from examples.post_training_quantization.onnx.yolov8_quantize_with_accuracy_control.main import (
        run_example as yolo8_main,
    )

    onnx_fp32_box_mAP, onnx_fp32_mask_mAP, onnx_int8_box_mAP, onnx_int8_mask_mAP = yolo8_main()

    import examples.post_training_quantization.onnx.yolov8_quantize_with_accuracy_control.deploy as yolov8_deploy

    return {
        "onnx_fp32_box_mAP": onnx_fp32_box_mAP,
        "onnx_fp32_mask_mAP": onnx_fp32_mask_mAP,
        "onnx_int8_box_mAP": onnx_int8_box_mAP,
        "onnx_int8_mask_mAP": onnx_int8_mask_mAP,
        "onnx_drop_box_mAP": onnx_fp32_box_mAP - onnx_int8_box_mAP,
        "onnx_drop_mask_mAP": onnx_fp32_mask_mAP - onnx_int8_mask_mAP,
        "ov_fp32_box_mAP": yolov8_deploy.fp32_stats["metrics/mAP50-95(B)"],
        "ov_fp32_mask_mAP": yolov8_deploy.fp32_stats["metrics/mAP50-95(M)"],
        "ov_int8_box_mAP": yolov8_deploy.int8_stats["metrics/mAP50-95(B)"],
        "ov_int8_mask_mAP": yolov8_deploy.int8_stats["metrics/mAP50-95(M)"],
        "ov_drop_box_mAP": yolov8_deploy.box_metric_drop,
        "ov_drop_mask_mAP": yolov8_deploy.mask_metric_drop,
        "ov_fp32_fps": yolov8_deploy.fp32_fps,
        "ov_int8_fps": yolov8_deploy.int8_fps,
        "performance_speed_up": yolov8_deploy.int8_fps / yolov8_deploy.fp32_fps,
    }


def post_training_quantization_openvino_anomaly_stfpm_quantize_with_accuracy_control() -> Dict[str, float]:
    from examples.post_training_quantization.openvino.anomaly_stfpm_quantize_with_accuracy_control.main import (
        run_example as anomaly_stfpm_main,
    )

    fp32_top1, int8_top1, fp32_fps, int8_fps, fp32_size, int8_size = anomaly_stfpm_main()

    return {
        "fp32_top1": float(fp32_top1),
        "int8_top1": float(int8_top1),
        "accuracy_drop": float(fp32_top1 - int8_top1),
        "fp32_fps": fp32_fps,
        "int8_fps": int8_fps,
        "performance_speed_up": int8_fps / fp32_fps,
        "fp32_model_size": fp32_size,
        "int8_model_size": int8_size,
        "model_compression_rate": fp32_size / int8_size,
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


def llm_compression() -> Dict[str, float]:
    from examples.llm_compression.openvino.tiny_llama.main import main as llm_compression_main

    result = llm_compression_main()

    return {"word_count": len(result.split())}


def llm_tune_params() -> Dict[str, float]:
    from examples.llm_compression.openvino.tiny_llama_find_hyperparams.main import main as llm_tune_params_main

    awq, ratio, group_size = llm_tune_params_main()

    return {"awq": bool(awq), "ratio": ratio, "group_size": group_size}


def llm_compression_synthetic() -> Dict[str, float]:
    from examples.llm_compression.openvino.tiny_llama_synthetic_data.main import main as llm_compression_synthetic_main

    result = llm_compression_synthetic_main()

    return {"word_count": len(result.split())}


def fp8_llm_quantization() -> Dict[str, float]:
    from examples.llm_compression.openvino.smollm2_360m_fp8.main import main as fp8_llm_quantization_main

    result = fp8_llm_quantization_main()

    return {"answers": list(result.values())}


def post_training_quantization_torch_fx_resnet18():
    from examples.post_training_quantization.torch_fx.resnet18.main import main as resnet18_main

    # Set manual seed and determenistic cuda mode to make the test determenistic
    results = resnet18_main()

    return {
        "fp32_top1": float(results[0]),
        "int8_top1": float(results[1]),
        "fp32_latency": float(results[2]),
        "fp32_ov_latency": float(results[3]),
        "int8_latency": float(results[4]),
    }


def quantization_aware_training_torch_resnet18():
    from examples.quantization_aware_training.torch.resnet18.main import main as resnet18_main

    # Set manual seed and determenistic cuda mode to make the test determenistic
    set_torch_cuda_seed()
    results = resnet18_main()

    return {
        "fp32_top1": float(results[0]),
        "int8_init_top1": float(results[1]),
        "int8_top1": float(results[2]),
        "accuracy_drop": float(results[0] - results[2]),
        "fp32_fps": results[3],
        "int8_fps": results[4],
        "performance_speed_up": results[4] / results[3],
        "fp32_model_size": results[5],
        "int8_model_size": results[6],
        "model_compression_rate": results[5] / results[6],
    }


def set_torch_cuda_seed(seed: int = 42):
    """
    Sets torch, cuda and python random module to determenistic mode with
    given seed.
    :param seed: Seed to use for determenistic run.
    """
    import random

    import numpy as np
    import torch
    from torch.backends import cudnn

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def quantization_aware_training_torch_anomalib(data: Union[str, None]):
    from anomalib.data.image import mvtec

    from examples.quantization_aware_training.torch.anomalib.main import DATASET_PATH as dataset_path
    from examples.quantization_aware_training.torch.anomalib.main import main as anomalib_main

    if data is not None and not dataset_path.exists():
        dataset_path.mkdir(parents=True, exist_ok=True)
        tar_file_path = Path(data) / mvtec.DOWNLOAD_INFO.url.split("/")[-1]
        with tarfile.open(tar_file_path) as tar_file:
            tar_file.extractall(dataset_path)

    # Set manual seed and determenistic cuda mode to make the test determenistic
    set_torch_cuda_seed()
    results = anomalib_main()

    return {
        "fp32_f1score": float(results[0]),
        "int8_init_f1score": float(results[1]),
        "int8_f1score": float(results[2]),
        "accuracy_drop": float(results[0] - results[2]),
        "fp32_fps": results[3],
        "int8_fps": results[4],
        "performance_speed_up": results[4] / results[3],
        "fp32_model_size": results[5],
        "int8_model_size": results[6],
        "model_compression_rate": results[5] / results[6],
    }


def main(argv):
    parser = ArgumentParser()
    parser.add_argument("--name", help="Example name", required=True)
    parser.add_argument("--data", help="Path to datasets", default=None, required=False)
    parser.add_argument("-o", "--output", help="Path to the json file to save example metrics", required=True)
    args = parser.parse_args(args=argv)

    # Disable progress bar for fastdownload module
    try:
        import fastprogress.fastprogress

        fastprogress.fastprogress.NO_BAR = True
    except ImportError:
        pass

    if args.name == "quantization_aware_training_torch_anomalib":
        metrics = globals()[args.name](args.data)
    else:
        metrics = globals()[args.name]()

    with open(args.output, "w", encoding="utf8") as json_file:
        return json.dump(metrics, json_file)


if __name__ == "__main__":
    self_argv = sys.argv[1:]
    sys.argv = sys.argv[:1]
    main(self_argv)
