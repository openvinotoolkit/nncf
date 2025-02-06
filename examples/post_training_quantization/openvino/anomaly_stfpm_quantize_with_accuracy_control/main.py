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
import re
import subprocess
import sys
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import openvino as ov
import torch
from anomalib.data.mvtec import MVTec
from anomalib.data.utils import download
from anomalib.post_processing.normalization.min_max import normalize
from anomalib.utils.metrics import create_metric_collection

import nncf

ROOT = Path(__file__).parent.resolve()
HOME_PATH = Path.home()
MODEL_INFO = download.DownloadInfo(
    name="stfpm_mvtec_capsule",
    url="https://huggingface.co/alexsu52/stfpm_mvtec_capsule/resolve/main/openvino_model.tar",
    hash="2005ef44eb701ad35e51417d196d8632",
)
MODEL_PATH = HOME_PATH / ".cache/nncf/models/stfpm_mvtec_capsule"

DATASET_INFO = download.DownloadInfo(
    name="mvtec_capsule",
    url="https://huggingface.co/datasets/alexsu52/mvtec_capsule/resolve/main/capsule.tar.xz",
    hash="380afc46701c99cb7b9a928edbe16eb5",
)
DATASET_PATH = HOME_PATH / ".cache/nncf/datasets/mvtec_capsule"

max_accuracy_drop = 0.005 if len(sys.argv) < 2 else float(sys.argv[1])


def download_and_extract(root: Path, info: download.DownloadInfo) -> None:
    if not root.is_dir():
        download.download_and_extract(root, info)


def get_anomaly_images(data_loader: Iterable[Any]) -> List[Dict[str, torch.Tensor]]:
    anomaly_images_ = []
    for data_item in data_loader:
        if data_item["label"].int() == 1:
            anomaly_images_.append({"image": data_item["image"]})
    return anomaly_images_


def validate(
    model: ov.CompiledModel, val_loader: Iterable[Any], val_params: Dict[str, float]
) -> Tuple[float, List[float]]:
    metric = create_metric_collection(["F1Score"], prefix="image_")["F1Score"]
    metric.threshold = 0.5
    per_sample_metric_values = []

    output = model.outputs[0]

    for batch in val_loader:
        anomaly_maps = model(batch["image"])[output]
        pred_scores = np.max(anomaly_maps, axis=(1, 2, 3))
        pred_scores = normalize(pred_scores, val_params["image_threshold"], val_params["min"], val_params["max"])
        pred_label = 1 if pred_scores > metric.threshold else 0
        groundtruth_label = batch["label"].int()
        per_sample_metric = 1.0 if pred_label == groundtruth_label else 0.0
        per_sample_metric_values.append(per_sample_metric)
        metric.update(torch.from_numpy(pred_scores), groundtruth_label)

    metric_value = metric.compute()
    return metric_value, per_sample_metric_values


def run_benchmark(model_path: Path, shape: List[int]) -> float:
    command = [
        "benchmark_app",
        "-m", model_path.as_posix(),
        "-d", "CPU",
        "-api", "async",
        "-t", "15",
        "-shape", str(shape),
    ]  # fmt: skip
    cmd_output = subprocess.check_output(command, text=True)  # nosec
    print(*cmd_output.splitlines()[-8:], sep="\n")
    match = re.search(r"Throughput\: (.+?) FPS", cmd_output)
    return float(match.group(1))


def get_model_size(ir_path: Path, m_type: str = "Mb") -> float:
    xml_size = ir_path.stat().st_size
    bin_size = ir_path.with_suffix(".bin").stat().st_size
    for t in ["bytes", "Kb", "Mb"]:
        if m_type == t:
            break
        xml_size /= 1024
        bin_size /= 1024
    model_size = xml_size + bin_size
    print(f"Model graph (xml):   {xml_size:.3f} Mb")
    print(f"Model weights (bin): {bin_size:.3f} Mb")
    print(f"Model size:          {model_size:.3f} Mb")
    return model_size


def run_example():
    ###############################################################################
    # Create an OpenVINO model and dataset

    download_and_extract(DATASET_PATH, DATASET_INFO)

    datamodule = MVTec(
        root=DATASET_PATH,
        category="capsule",
        image_size=(256, 256),
        train_batch_size=1,
        eval_batch_size=1,
        num_workers=0,
    )
    datamodule.setup()
    test_loader = datamodule.test_dataloader()

    download_and_extract(MODEL_PATH, MODEL_INFO)
    ov_model = ov.Core().read_model(MODEL_PATH / "stfpm_capsule.xml")

    with open(MODEL_PATH / "meta_data_stfpm_capsule.json", "r", encoding="utf-8") as f:
        validation_params = json.load(f)

    ###############################################################################
    # Quantize an OpenVINO model with accuracy control
    #
    # The transformation function transforms a data item into model input data.
    #
    # To validate the transform function use the following code:
    # >> for data_item in val_loader:
    # >>    model(transform_fn(data_item))

    def transform_fn(data_item):
        return data_item["image"]

    # Uses only anomaly images for calibration process
    anomaly_images = get_anomaly_images(test_loader)
    calibration_dataset = nncf.Dataset(anomaly_images, transform_fn)

    # Whole test dataset is used for validation
    validation_fn = partial(validate, val_params=validation_params)
    validation_dataset = nncf.Dataset(test_loader, transform_fn)

    ov_quantized_model = nncf.quantize_with_accuracy_control(
        model=ov_model,
        calibration_dataset=calibration_dataset,
        subset_size=len(anomaly_images),
        validation_dataset=validation_dataset,
        validation_fn=validation_fn,
        max_drop=max_accuracy_drop,
    )

    ###############################################################################
    # Benchmark performance, calculate compression rate and validate accuracy

    fp32_ir_path = ROOT / "stfpm_fp32.xml"
    ov.save_model(ov_model, fp32_ir_path, compress_to_fp16=False)
    print(f"[1/7] Save FP32 model: {fp32_ir_path}")
    fp32_size = get_model_size(fp32_ir_path)

    # To avoid an accuracy drop when saving a model due to compression of unquantized
    # weights to FP16, compress_to_fp16=False should be used. This is necessary because
    # nncf.quantize_with_accuracy_control(...) keeps the most impactful operations within
    # the model in the original precision to achieve the specified model accuracy.
    int8_ir_path = ROOT / "stfpm_int8.xml"
    ov.save_model(ov_quantized_model, int8_ir_path)
    print(f"[2/7] Save INT8 model: {int8_ir_path}")
    int8_size = get_model_size(int8_ir_path)

    print("[3/7] Benchmark FP32 model:")
    fp32_fps = run_benchmark(fp32_ir_path, shape=[1, 3, 256, 256])
    print("[4/7] Benchmark INT8 model:")
    int8_fps = run_benchmark(int8_ir_path, shape=[1, 3, 256, 256])

    print("[5/7] Validate OpenVINO FP32 model:")
    compiled_model = ov.compile_model(ov_model, device_name="CPU")
    fp32_top1, _ = validate(compiled_model, test_loader, validation_params)
    print(f"Accuracy @ top1: {fp32_top1:.3f}")

    print("[6/7] Validate OpenVINO INT8 model:")
    quantized_compiled_model = ov.compile_model(ov_quantized_model, device_name="CPU")
    int8_top1, _ = validate(quantized_compiled_model, test_loader, validation_params)
    print(f"Accuracy @ top1: {int8_top1:.3f}")

    print("[7/7] Report:")
    print(f"Maximum accuracy drop:                  {max_accuracy_drop}")
    print(f"Accuracy drop:                          {fp32_top1 - int8_top1:.3f}")
    print(f"Model compression rate:                 {fp32_size / int8_size:.3f}")
    # https://docs.openvino.ai/latest/openvino_docs_optimization_guide_dldt_optimization_guide.html
    print(f"Performance speed up (throughput mode): {int8_fps / fp32_fps:.3f}")

    return fp32_top1, int8_top1, fp32_fps, int8_fps, fp32_size, int8_size


if __name__ == "__main__":
    run_example()
