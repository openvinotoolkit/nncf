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

import re
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple

import openvino as ov
import torch
from rich.progress import track
from ultralytics.cfg import get_cfg
from ultralytics.models.yolo import YOLO
from ultralytics.models.yolo.segment.val import SegmentationValidator
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.metrics import ConfusionMatrix

from examples.post_training_quantization.onnx.yolov8_quantize_with_accuracy_control.main import prepare_validation
from examples.post_training_quantization.onnx.yolov8_quantize_with_accuracy_control.main import print_statistics

ROOT = Path(__file__).parent.resolve()
MODEL_NAME = "yolov8n-seg"
FP32_ONNX_MODEL_PATH = ROOT / f"{MODEL_NAME}.onnx"
INT8_ONNX_MODEL_PATH = ROOT / f"{MODEL_NAME}_int8.onnx"
FP32_OV_MODEL_PATH = ROOT / f"{MODEL_NAME}.xml"
INT8_OV_MODEL_PATH = ROOT / f"{MODEL_NAME}_int8.xml"


def validate_ov_model(
    ov_model: ov.Model,
    data_loader: torch.utils.data.DataLoader,
    validator: SegmentationValidator,
    num_samples: Optional[int] = None,
) -> Tuple[Dict, int, int]:
    validator.seen = 0
    validator.jdict = []
    validator.stats = dict(tp_m=[], tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])
    validator.batch_i = 1
    validator.confusion_matrix = ConfusionMatrix(nc=validator.nc)
    compiled_model = ov.compile_model(ov_model, device_name="CPU")
    num_outputs = len(compiled_model.outputs)
    for batch_i, batch in enumerate(track(data_loader, description="Validating")):
        if num_samples is not None and batch_i == num_samples:
            break
        batch = validator.preprocess(batch)
        results = compiled_model(batch["img"])
        if num_outputs == 1:
            preds = torch.from_numpy(results[compiled_model.output(0)])
        else:
            preds = [
                torch.from_numpy(results[compiled_model.output(0)]),
                torch.from_numpy(results[compiled_model.output(1)]),
            ]
        preds = validator.postprocess(preds)
        validator.update_metrics(preds, batch)
    stats = validator.get_stats()
    return stats, validator.seen, validator.nt_per_class.sum()


def run_benchmark(model_path: Path, config) -> float:
    command = [
        "benchmark_app",
        "-m", model_path.as_posix(),
        "-d", "CPU",
        "-api", "async",
        "-t", "30",
        "-shape", str([1, 3, config.imgsz, config.imgsz]),
    ]  # fmt: skip
    cmd_output = subprocess.check_output(command, text=True)  # nosec
    match = re.search(r"Throughput\: (.+?) FPS", cmd_output)
    return float(match.group(1))


args = get_cfg(cfg=DEFAULT_CFG)
args.data = "coco128-seg.yaml"

print(f"[1/7] Save FP32 OpenVINO model: {FP32_OV_MODEL_PATH}")
fp32_ov_model = ov.convert_model(FP32_ONNX_MODEL_PATH)
ov.save_model(fp32_ov_model, FP32_OV_MODEL_PATH, compress_to_fp16=False)

print(f"[2/7] Save INT8 OpenVINO model: {INT8_OV_MODEL_PATH}")
int8_ov_model = ov.convert_model(INT8_ONNX_MODEL_PATH)
ov.save_model(int8_ov_model, INT8_OV_MODEL_PATH, compress_to_fp16=False)

print("[3/7] Benchmark FP32 OpenVINO model:", end=" ")
fp32_fps = run_benchmark(FP32_OV_MODEL_PATH, args)
print(f"{fp32_fps} FPS")

print("[4/7] Benchmark INT8 OpenVINO model:", end=" ")
int8_fps = run_benchmark(INT8_OV_MODEL_PATH, args)
print(f"{int8_fps} FPS")

validator, data_loader = prepare_validation(YOLO(ROOT / f"{MODEL_NAME}.pt"), args)

print("[5/7] Validate OpenVINO FP32 model:")
fp32_stats, total_images, total_objects = validate_ov_model(fp32_ov_model, data_loader, validator)
print_statistics(fp32_stats, total_images, total_objects)

print("[6/7] Validate OpenVINO INT8 model:")
int8_stats, total_images, total_objects = validate_ov_model(int8_ov_model, data_loader, validator)
print_statistics(int8_stats, total_images, total_objects)

print("[7/7] Report:")
box_metric_drop = fp32_stats["metrics/mAP50-95(B)"] - int8_stats["metrics/mAP50-95(B)"]
mask_metric_drop = fp32_stats["metrics/mAP50-95(M)"] - int8_stats["metrics/mAP50-95(M)"]
print(f"\tMetric drop mAP50-95(B): {box_metric_drop:.6f}")
print(f"\tMetric drop mAP50-95(M): {mask_metric_drop:.6f}")
print(f"\tPerformance speed up (throughput mode): {int8_fps / fp32_fps:.3f}")
