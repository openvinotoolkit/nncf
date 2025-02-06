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
from typing import Any, Dict, Tuple

import openvino as ov
import torch
from rich.progress import track
from ultralytics.cfg import get_cfg
from ultralytics.data.converter import coco80_to_coco91_class
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import DATASETS_DIR
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.metrics import ConfusionMatrix

import nncf

MODEL_NAME = "yolov8n"

ROOT = Path(__file__).parent.resolve()


def validate(
    model: ov.Model, data_loader: torch.utils.data.DataLoader, validator: DetectionValidator, num_samples: int = None
) -> Tuple[Dict, int, int]:
    validator.seen = 0
    validator.jdict = []
    validator.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])
    validator.confusion_matrix = ConfusionMatrix(nc=validator.nc)
    model.reshape({0: [1, 3, -1, -1]})
    compiled_model = ov.compile_model(model, device_name="CPU")
    output_layer = compiled_model.output(0)
    for batch_i, batch in enumerate(track(data_loader, description="Validating")):
        if num_samples is not None and batch_i == num_samples:
            break
        batch = validator.preprocess(batch)
        preds = torch.from_numpy(compiled_model(batch["img"])[output_layer])
        preds = validator.postprocess(preds)
        validator.update_metrics(preds, batch)
    stats = validator.get_stats()
    return stats, validator.seen, validator.nt_per_class.sum()


def print_statistics(stats: Dict[str, float], total_images: int, total_objects: int) -> None:
    mp, mr, map50, mean_ap = (
        stats["metrics/precision(B)"],
        stats["metrics/recall(B)"],
        stats["metrics/mAP50(B)"],
        stats["metrics/mAP50-95(B)"],
    )
    s = ("%20s" + "%12s" * 6) % ("Class", "Images", "Labels", "Precision", "Recall", "mAP@.5", "mAP@.5:.95")
    print(s)
    pf = "%20s" + "%12i" * 2 + "%12.3g" * 4  # print format
    print(pf % ("all", total_images, total_objects, mp, mr, map50, mean_ap))


def prepare_validation(model: YOLO, args: Any) -> Tuple[DetectionValidator, torch.utils.data.DataLoader]:
    validator: DetectionValidator = model.task_map[model.task]["validator"](args=args)
    validator.data = check_det_dataset(args.data)
    validator.stride = 32
    validator.is_coco = True
    validator.class_map = coco80_to_coco91_class()
    validator.names = model.model.names
    validator.metrics.names = validator.names
    validator.nc = model.model.model[-1].nc

    coco_data_path = DATASETS_DIR / "coco128"
    data_loader = validator.get_dataloader(coco_data_path.as_posix(), 1)

    return validator, data_loader


def benchmark_performance(model_path: Path, config) -> float:
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


def prepare_openvino_model(model: YOLO, model_name: str) -> Tuple[ov.Model, Path]:
    ir_model_path = ROOT / f"{model_name}_openvino_model" / f"{model_name}.xml"
    if not ir_model_path.exists():
        onnx_model_path = ROOT / f"{model_name}.onnx"
        if not onnx_model_path.exists():
            model.export(format="onnx", dynamic=True, half=False)

        ov.save_model(ov.convert_model(onnx_model_path), ir_model_path)
    return ov.Core().read_model(ir_model_path), ir_model_path


def quantize(model: ov.Model, data_loader: torch.utils.data.DataLoader, validator: DetectionValidator) -> ov.Model:
    def transform_fn(data_item: Dict):
        """
        Quantization transform function. Extracts and preprocess input data from dataloader
        item for quantization.
        Parameters:
        data_item: Dict with data item produced by DataLoader during iteration
        Returns:
            input_tensor: Input data for quantization
        """
        input_tensor = validator.preprocess(data_item)["img"].numpy()
        return input_tensor

    quantization_dataset = nncf.Dataset(data_loader, transform_fn)

    quantized_model = nncf.quantize(
        model,
        quantization_dataset,
        subset_size=len(data_loader),
        preset=nncf.QuantizationPreset.MIXED,
        ignored_scope=nncf.IgnoredScope(
            types=["Multiply", "Subtract", "Sigmoid"],
            subgraphs=[
                nncf.Subgraph(
                    inputs=["/model.22/Concat", "/model.22/Concat_1", "/model.22/Concat_2"],
                    outputs=["output0/sink_port_0"],
                )
            ],
        ),
    )
    return quantized_model


def main():
    model = YOLO(ROOT / f"{MODEL_NAME}.pt")
    args = get_cfg(cfg=DEFAULT_CFG)
    args.data = "coco128.yaml"

    # Prepare validation dataset and helper
    validator, data_loader = prepare_validation(model, args)

    # Convert to OpenVINO model
    ov_model, ov_model_path = prepare_openvino_model(model, MODEL_NAME)

    # Quantize mode in OpenVINO representation
    quantized_model = quantize(ov_model, data_loader, validator)
    quantized_model_path = ov_model_path.with_name(ov_model_path.stem + "_quantized" + ov_model_path.suffix)
    ov.save_model(quantized_model, str(quantized_model_path))

    # Validate FP32 model
    fp_stats, total_images, total_objects = validate(ov_model, data_loader, validator)
    print("Floating-point model validation results:")
    print_statistics(fp_stats, total_images, total_objects)

    # Validate quantized model
    q_stats, total_images, total_objects = validate(quantized_model, data_loader, validator)
    print("Quantized model validation results:")
    print_statistics(q_stats, total_images, total_objects)

    # Benchmark performance of FP32 model
    fp_model_perf = benchmark_performance(ov_model_path, args)
    print(f"Floating-point model performance: {fp_model_perf} FPS")

    # Benchmark performance of quantized model
    quantized_model_perf = benchmark_performance(quantized_model_path, args)
    print(f"Quantized model performance: {quantized_model_perf} FPS")

    return fp_stats["metrics/mAP50-95(B)"], q_stats["metrics/mAP50-95(B)"], fp_model_perf, quantized_model_perf


if __name__ == "__main__":
    main()
