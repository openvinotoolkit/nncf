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

from functools import partial
from pathlib import Path
from typing import Any, Dict, Tuple

import onnx
import onnxruntime
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
    model: onnx.ModelProto,
    data_loader: torch.utils.data.DataLoader,
    validator: SegmentationValidator,
    num_samples: int = None,
) -> Tuple[Dict, int, int]:
    validator.seen = 0
    validator.jdict = []
    validator.stats = dict(tp_m=[], tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])
    validator.batch_i = 1
    validator.confusion_matrix = ConfusionMatrix(nc=validator.nc)

    input_name = model.graph.input[0].name
    serialized_model = model.SerializeToString()

    session = onnxruntime.InferenceSession(serialized_model, providers=["CPUExecutionProvider"])
    output_names = [output.name for output in session.get_outputs()]
    num_outputs = len(output_names)

    for batch_i, batch in enumerate(track(data_loader, description="Validating")):
        if num_samples is not None and batch_i == num_samples:
            break
        batch = validator.preprocess(batch)

        input_feed = {input_name: batch["img"].numpy()}
        results = session.run(output_names, input_feed=input_feed)

        if num_outputs == 1:
            preds = torch.from_numpy(results[0])
        else:
            preds = [
                torch.from_numpy(results[0]),
                torch.from_numpy(results[1]),
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


def prepare_onnx_model(model: YOLO, model_name: str) -> Tuple[onnx.ModelProto, Path]:
    model_path = ROOT / f"{model_name}.onnx"
    if not model_path.exists():
        model.export(format="onnx", dynamic=True, half=False)

    model = onnx.load(model_path)
    return model, model_path


def quantize_ac(
    model: onnx.ModelProto, data_loader: torch.utils.data.DataLoader, validator_ac: SegmentationValidator
) -> onnx.ModelProto:
    input_name = model.graph.input[0].name

    def transform_fn(data_item: Dict):
        input_tensor = validator_ac.preprocess(data_item)["img"].numpy()
        return {input_name: input_tensor}

    def validation_ac(
        val_model: onnx.ModelProto,
        validation_loader: torch.utils.data.DataLoader,
        validator: SegmentationValidator,
        num_samples: int = None,
    ) -> float:
        validator.seen = 0
        validator.jdict = []
        validator.stats = dict(tp_m=[], tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])
        validator.batch_i = 1
        validator.confusion_matrix = ConfusionMatrix(nc=validator.nc)

        rt_session_options = {"providers": ["CPUExecutionProvider"]}
        serialized_model = val_model.SerializeToString()
        session = onnxruntime.InferenceSession(serialized_model, **rt_session_options)
        output_names = [output.name for output in session.get_outputs()]
        num_outputs = len(output_names)

        for batch_i, batch in enumerate(validation_loader):
            if num_samples is not None and batch_i == num_samples:
                break
            batch = validator.preprocess(batch)
            input_feed = {input_name: batch["img"].numpy()}
            results = session.run(output_names, input_feed=input_feed)

            if num_outputs == 1:
                preds = torch.from_numpy(results[0])
            else:
                preds = [
                    torch.from_numpy(results[0]),
                    torch.from_numpy(results[1]),
                ]
            preds = validator.postprocess(preds)
            validator.update_metrics(preds, batch)

        stats = validator.get_stats()
        if num_outputs == 1:
            stats_metrics = stats["metrics/mAP50-95(B)"]
        else:
            stats_metrics = stats["metrics/mAP50-95(M)"]
        return stats_metrics, None

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
            types=["Mul", "Sub", "Sigmoid"],  # ignore operations
            subgraphs=[
                nncf.Subgraph(
                    inputs=[
                        "/model.22/Concat_3",
                        "/model.22/Concat_6",
                        "/model.22/Concat_24",
                        "/model.22/Concat_5",
                        "/model.22/Concat_4",
                    ],
                    outputs=["/model.22/Concat_29"],
                )
            ],
        ),
    )
    return quantized_model_ac


def run_example():
    model = YOLO(ROOT / f"{MODEL_NAME}.pt")
    args = get_cfg(cfg=DEFAULT_CFG)
    args.data = "coco128-seg.yaml"

    validator, data_loader = prepare_validation(model, args)

    fp32_model, fp32_model_path = prepare_onnx_model(model, MODEL_NAME)
    print(f"[1/5] Save FP32 model: {fp32_model_path}")

    int8_model = quantize_ac(fp32_model, data_loader, validator)

    int8_model_path = ROOT / f"{MODEL_NAME}_int8.onnx"
    onnx.save(int8_model, int8_model_path)
    print(f"[2/5] Save INT8 model: {int8_model_path}")

    print("[3/5] Validate ONNX FP32 model:")
    fp_stats, total_images, total_objects = validate(fp32_model, data_loader, validator)
    print_statistics(fp_stats, total_images, total_objects)

    print("[4/5] Validate ONNX INT8 model:")
    q_stats, total_images, total_objects = validate(int8_model, data_loader, validator)
    print_statistics(q_stats, total_images, total_objects)

    print("[5/5] Report:")
    metric_drop = fp_stats["metrics/mAP50-95(B)"] - q_stats["metrics/mAP50-95(B)"]
    print(f"Metric drop: {metric_drop}")

    # fp32_box_mAP, fp32_mask_mAP, int8_box_mAP, int8_mask_mAP
    return (
        fp_stats["metrics/mAP50-95(B)"],
        fp_stats["metrics/mAP50-95(M)"],
        q_stats["metrics/mAP50-95(B)"],
        q_stats["metrics/mAP50-95(M)"],
    )


if __name__ == "__main__":
    run_example()
