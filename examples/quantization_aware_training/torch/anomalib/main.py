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
import re
import subprocess
import tarfile
from copy import deepcopy
from pathlib import Path
from typing import List
from urllib.request import urlretrieve

import torch
from anomalib import TaskType
from anomalib.data import MVTec
from anomalib.data.image import mvtec
from anomalib.data.utils import download
from anomalib.deploy import ExportType
from anomalib.engine import Engine
from anomalib.models import Stfpm

import nncf

HOME_PATH = Path.home()
DATASET_PATH = HOME_PATH / ".cache" / "nncf" / "datasets" / "mvtec"
CHECKPOINT_PATH = HOME_PATH / ".cache" / "nncf" / "models" / "anomalib"
ROOT = Path(__file__).parent.resolve()
FP32_RESULTS_ROOT = ROOT / "results" / "fp32"
INT8_RESULTS_ROOT = ROOT / "results" / "int8"
CHECKPOINT_URL = "https://storage.openvinotoolkit.org/repositories/nncf/examples/torch/anomalib/stfpm_mvtec.ckpt"
USE_PRETRAINED = True


def download_and_extract(root: Path, info: download.DownloadInfo) -> None:
    root.mkdir(parents=True, exist_ok=True)
    downloaded_file_path = root / info.url.split("/")[-1]
    print(f"Downloading the {info.name} dataset.")
    with download.DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=info.name) as progress_bar:
        urlretrieve(url=f"{info.url}", filename=downloaded_file_path, reporthook=progress_bar.update_to)  # nosec
    print("Checking the hash of the downloaded file.")
    download.check_hash(downloaded_file_path, info.hashsum)
    print(f"Extracting the {info.name} dataset.")
    with tarfile.open(downloaded_file_path) as tar_file:
        tar_file.extractall(root)
    print("Cleaning up files.")
    downloaded_file_path.unlink()


def create_dataset(root: Path) -> MVTec:
    if not root.exists():
        download_and_extract(root, mvtec.DOWNLOAD_INFO)
    return MVTec(root)


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


def main():
    ###############################################################################
    # Step 1: Prepare the model and dataset
    print(os.linesep + "[Step 1] Prepare the model and dataset")

    model = Stfpm()
    datamodule = create_dataset(root=DATASET_PATH)

    # Create an engine for the original model
    engine = Engine(task=TaskType.SEGMENTATION, default_root_dir=FP32_RESULTS_ROOT, devices=1)
    if USE_PRETRAINED:
        # Load the pretrained checkpoint
        CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)
        ckpt_path = CHECKPOINT_PATH / "stfpm_mvtec.ckpt"
        torch.hub.download_url_to_file(CHECKPOINT_URL, ckpt_path)
    else:
        # (Optional) Train the model from scratch
        engine.fit(model=model, datamodule=datamodule)
        ckpt_path = engine.trainer.checkpoint_callback.best_model_path

    print("Test results for original FP32 model:")
    fp32_test_results = engine.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    ###############################################################################
    # Step 2: Quantize the model
    print(os.linesep + "[Step 2] Quantize the model")

    # Create calibration dataset
    def transform_fn(data_item):
        return data_item["image"]

    test_loader = datamodule.test_dataloader()
    calibration_dataset = nncf.Dataset(test_loader, transform_fn)

    # Quantize the inference model using Post-Training Quantization
    inference_model = model.model
    quantized_inference_model = nncf.quantize(model=inference_model, calibration_dataset=calibration_dataset)

    # Deepcopy the original model and set the quantized inference model
    quantized_model = deepcopy(model)
    quantized_model.model = quantized_inference_model

    # Create engine for the quantized model
    engine = Engine(task=TaskType.SEGMENTATION, default_root_dir=INT8_RESULTS_ROOT, max_epochs=1, devices=1)

    # Validate the quantized model
    print("Test results for INT8 model after PTQ:")
    int8_init_test_results = engine.test(model=quantized_model, datamodule=datamodule)

    ###############################################################################
    # Step 3: Fine tune the quantized model
    print(os.linesep + "[Step 3] Fine tune the quantized model")

    engine.fit(model=quantized_model, datamodule=datamodule)
    print("Test results for INT8 model after QAT:")
    int8_test_results = engine.test(model=quantized_model, datamodule=datamodule)

    ###############################################################################
    # Step 4: Export models
    print(os.linesep + "[Step 4] Export models")

    # Export FP32 model to OpenVINO™ IR
    fp32_ir_path = engine.export(model=model, export_type=ExportType.OPENVINO, export_root=FP32_RESULTS_ROOT)
    print(f"Original model path: {fp32_ir_path}")
    fp32_size = get_model_size(fp32_ir_path)

    # Export INT8 model to OpenVINO™ IR
    int8_ir_path = engine.export(model=quantized_model, export_type=ExportType.OPENVINO, export_root=INT8_RESULTS_ROOT)
    print(f"Quantized model path: {int8_ir_path}")
    int8_size = get_model_size(int8_ir_path)

    ###############################################################################
    # Step 5: Run benchmarks
    print(os.linesep + "[Step 5] Run benchmarks")

    print("Run benchmark for FP32 model (IR)...")
    fp32_fps = run_benchmark(fp32_ir_path, shape=[1, 3, 256, 256])

    print("Run benchmark for INT8 model (IR)...")
    int8_fps = run_benchmark(int8_ir_path, shape=[1, 3, 256, 256])

    ###############################################################################
    # Step 6: Summary
    print(os.linesep + "[Step 6] Summary")

    fp32_f1score = fp32_test_results[0]["image_F1Score"]
    int8_init_f1score = int8_init_test_results[0]["image_F1Score"]
    int8_f1score = int8_test_results[0]["image_F1Score"]

    print(f"Accuracy drop after PTQ:                {fp32_f1score - int8_init_f1score:.3f}")
    print(f"Accuracy drop after QAT:                {fp32_f1score - int8_f1score:.3f}")
    print(f"Model compression rate:                 {fp32_size / int8_size:.3f}")
    # https://docs.openvino.ai/latest/openvino_docs_optimization_guide_dldt_optimization_guide.html
    print(f"Performance speed up (throughput mode): {int8_fps / fp32_fps:.3f}")

    return fp32_f1score, int8_init_f1score, int8_f1score, fp32_fps, int8_fps, fp32_size, int8_size


if __name__ == "__main__":
    main()
