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
from functools import partial
from pathlib import Path
from typing import Any, Dict, Tuple

import openvino as ov
import torch
from rich.progress import track
from ultralytics.cfg import get_cfg
from ultralytics.data.converter import coco80_to_coco91_class
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo import YOLO
from ultralytics.models.yolo.segment.val import SegmentationValidator
from ultralytics.utils import DATASETS_DIR
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils import ops
from ultralytics.utils.metrics import ConfusionMatrix

import nncf

MODEL_NAME = "yolov8n-seg"

ROOT = Path(__file__).parent.resolve()


def validate(
    model: ov.Model, data_loader: torch.utils.data.DataLoader, validator: SegmentationValidator, num_samples: int = None
) -> Tuple[Dict, int, int]:
    validator.seen = 0
    validator.jdict = []
    validator.stats = dict(tp_m=[], tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])
    validator.batch_i = 1
    validator.confusion_matrix = ConfusionMatrix(nc=validator.nc)
    model.reshape({0: [1, 3, -1, -1]})
    compiled_model = ov.compile_model(model, device_name="CPU")
    num_outputs = len(model.outputs)
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


def print_statistics(stats: Dict[str, float], total_images: int, total_objects: int) -> None:
    print("Metrics(Box):")
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

    print("Metrics(Mask):")
    s_mp, s_mr, s_map50, s_mean_ap = (
        stats["metrics/precision(M)"],
        stats["metrics/recall(M)"],
        stats["metrics/mAP50(M)"],
        stats["metrics/mAP50-95(M)"],
    )
    # Print results
    s = ("%20s" + "%12s" * 6) % ("Class", "Images", "Labels", "Precision", "Recall", "mAP@.5", "mAP@.5:.95")
    print(s)
    pf = "%20s" + "%12i" * 2 + "%12.3g" * 4  # print format
    print(pf % ("all", total_images, total_objects, s_mp, s_mr, s_map50, s_mean_ap))


def prepare_validation(model: YOLO, args: Any) -> Tuple[SegmentationValidator, torch.utils.data.DataLoader]:
    validator: SegmentationValidator = model.task_map[model.task]["validator"](args=args)
    validator.data = check_det_dataset(args.data)
    validator.stride = 32
    validator.is_coco = True
    validator.class_map = coco80_to_coco91_class()
    validator.names = model.model.names
    validator.metrics.names = validator.names
    validator.nc = model.model.model[-1].nc
    validator.process = ops.process_mask
    validator.plot_masks = []

    coco_data_path = DATASETS_DIR / "coco128-seg"
    data_loader = validator.get_dataloader(coco_data_path.as_posix(), 1)

    return validator, data_loader


def benchmark_performance(model_path, config) -> float:
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


def quantize_ac(
    model: ov.Model, data_loader: torch.utils.data.DataLoader, validator_ac: SegmentationValidator
) -> ov.Model:
    def transform_fn(data_item: Dict):
        input_tensor = validator_ac.preprocess(data_item)["img"].numpy()
        return input_tensor

    def validation_ac(
        compiled_model: ov.CompiledModel,
        validation_loader: torch.utils.data.DataLoader,
        validator: SegmentationValidator,
        num_samples: int = None,
    ) -> float:
        validator.seen = 0
        validator.jdict = []
        validator.stats = dict(tp_m=[], tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])
        validator.batch_i = 1
        validator.confusion_matrix = ConfusionMatrix(nc=validator.nc)
        num_outputs = len(compiled_model.outputs)

        for batch_i, batch in enumerate(validation_loader):
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
        if num_outputs == 1:
            stats_metrics = stats["metrics/mAP50-95(B)"]
        else:
            stats_metrics = stats["metrics/mAP50-95(M)"]
        return stats_metrics

    quantization_dataset = nncf.Dataset(data_loader, transform_fn)

    validation_fn = partial(validation_ac, validator=validator_ac)

    quantized_model_ac = nncf.quantize_with_accuracy_control(
        model,
        quantization_dataset,
        quantization_dataset,
        subset_size=len(data_loader),
        validation_fn=validation_fn,
        max_drop=0.003,
        preset=nncf.QuantizationPreset.MIXED,
        ignored_scope=nncf.IgnoredScope(
            types=["Multiply", "Subtract", "Sigmoid"],  # ignore operations
            subgraphs=[
                nncf.Subgraph(
                    inputs=[
                        "/model.22/Concat_3",
                        "/model.22/Concat_6",
                        "/model.22/Concat_5",
                        "/model.22/Concat_4",
                    ],
                    outputs=["output0"],
                )
            ],
        ),
    )
    return quantized_model_ac


def main():
    model = YOLO(ROOT / f"{MODEL_NAME}.pt")
    args = get_cfg(cfg=DEFAULT_CFG)
    args.data = "coco128-seg.yaml"
    args.workers = 0

    # Prepare validation dataset and helper
    validator, data_loader = prepare_validation(model, args)

    # Convert to OpenVINO model
    ov_model, ov_model_path = prepare_openvino_model(model, MODEL_NAME)

    # Quantize mode in OpenVINO representation
    quantized_model = quantize_ac(ov_model, data_loader, validator)

    quantized_model_path = ov_model_path.with_name(ov_model_path.stem + "_quantized" + ov_model_path.suffix)
    ov.save_model(quantized_model, str(quantized_model_path))

    # Validate FP32 model
    print("Floating-point model validation results:")
    fp_stats, total_images, total_objects = validate(ov_model, data_loader, validator)
    print_statistics(fp_stats, total_images, total_objects)

    # Validate quantized model
    print("Quantized model validation results:")
    q_stats, total_images, total_objects = validate(quantized_model, data_loader, validator)
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
