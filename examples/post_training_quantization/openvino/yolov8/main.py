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
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import openvino.runtime as ov
import torch
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.yolo.configs import get_config
from ultralytics.yolo.data.utils import check_dataset_yaml
from ultralytics.yolo.engine.validator import BaseValidator as Validator
from ultralytics.yolo.utils import DEFAULT_CONFIG
from ultralytics.yolo.utils import ops
from ultralytics.yolo.utils.metrics import ConfusionMatrix

import nncf


def validate(
    model: ov.Model, data_loader: torch.utils.data.DataLoader, validator: Validator, num_samples: int = None
) -> Tuple[Dict, int, int]:
    validator.seen = 0
    validator.jdict = []
    validator.stats = []
    validator.confusion_matrix = ConfusionMatrix(nc=validator.nc)
    model.reshape({0: [1, 3, -1, -1]})
    compiled_model = ov.compile_model(model)
    output_layer = compiled_model.output(0)
    for batch_i, batch in enumerate(data_loader):
        if num_samples is not None and batch_i == num_samples:
            break
        batch = validator.preprocess(batch)
        preds = torch.from_numpy(compiled_model(batch["img"])[output_layer])
        preds = validator.postprocess(preds)
        validator.update_metrics(preds, batch)
    stats = validator.get_stats()
    return stats, validator.seen, validator.nt_per_class.sum()


def print_statistics(stats: np.ndarray, total_images: int, total_objects: int) -> None:
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


def prepare_validation(model: YOLO, args: Any) -> Tuple[Validator, torch.utils.data.DataLoader]:
    data = check_dataset_yaml(args.data)
    dataset = data["val"]
    print(f"{dataset}")

    validator = model.ValidatorClass(args)
    data_loader = validator.get_dataloader("../datasets/coco128", 1)

    validator = model.ValidatorClass(args)

    validator.is_coco = True
    validator.class_map = ops.coco80_to_coco91_class()
    validator.names = model.model.names
    validator.metrics.names = validator.names
    validator.nc = model.model.model[-1].nc

    return validator, data_loader


def benchmark_performance(model_path, config) -> float:
    command = f"benchmark_app -m {model_path} -d CPU -api async -t 30"
    command += f' -shape "[1,3,{config.imgsz},{config.imgsz}]"'
    cmd_output = subprocess.check_output(command, shell=True)  # nosec

    match = re.search(r"Throughput\: (.+?) FPS", str(cmd_output))
    return float(match.group(1))


def prepare_openvino_model(model: YOLO, model_name: str) -> Tuple[ov.Model, Path]:
    model_path = Path(f"{model_name}_openvino_model/{model_name}.xml")
    if not model_path.exists():
        model.export(format="openvino", dynamic=True, half=False)

    model = ov.Core().read_model(model_path)
    return model, model_path


def quantize(model: ov.Model, data_loader: torch.utils.data.DataLoader, validator: Validator) -> ov.Model:
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
        preset=nncf.QuantizationPreset.MIXED,
        ignored_scope=nncf.IgnoredScope(
            types=["Multiply", "Subtract", "Sigmoid"],  # ignore operations
            names=[
                "/model.22/dfl/conv/Conv",  # in the post-processing subgraph
                "/model.22/Add",
                "/model.22/Add_1",
                "/model.22/Add_2",
                "/model.22/Add_3",
                "/model.22/Add_4",
                "/model.22/Add_5",
                "/model.22/Add_6",
                "/model.22/Add_7",
                "/model.22/Add_8",
                "/model.22/Add_9",
                "/model.22/Add_10",
            ],
        ),
    )
    return quantized_model


def main():
    MODEL_NAME = "yolov8n"

    model = YOLO(f"{MODEL_NAME}.pt")
    args = get_config(config=DEFAULT_CONFIG)
    args.data = "coco128.yaml"

    # Prepare validation dataset and helper
    validator, data_loader = prepare_validation(model, args)

    # Convert to OpenVINO model
    ov_model, ov_model_path = prepare_openvino_model(model, MODEL_NAME)

    # Quantize mode in OpenVINO representation
    quantized_model = quantize(ov_model, data_loader, validator)
    quantized_model_path = Path(f"{MODEL_NAME}_openvino_model/{MODEL_NAME}_quantized.xml")
    ov.serialize(quantized_model, str(quantized_model_path))

    # Validate FP32 model
    fp_stats, total_images, total_objects = validate(ov_model, tqdm(data_loader), validator)
    print("Floating-point model validation results:")
    print_statistics(fp_stats, total_images, total_objects)

    # Validate quantized model
    q_stats, total_images, total_objects = validate(quantized_model, tqdm(data_loader), validator)
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
