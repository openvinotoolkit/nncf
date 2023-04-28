"""
 Copyright (c) 2023 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import os
import re
import subprocess
from pathlib import Path

import openvino.runtime as ov
import torch
import torchvision
from fastdownload import FastDownload
from openvino.tools import mo
from PIL import Image
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection.ssd import SSD
from torchvision.models.detection.ssd import GeneralizedRCNNTransform
from tqdm import tqdm

import nncf
from nncf.torch import disable_tracing

ROOT = Path(__file__).parent.resolve()
DATASET_URL = "https://ultralytics.com/assets/coco128.zip"
DATASET_PATH = "~/.cache/nncf/datasets"


def download_dataset() -> Path:
    downloader = FastDownload(base=DATASET_PATH, archive="downloaded", data="extracted")
    return downloader.get(DATASET_URL)


def get_model_size(ir_path: str, m_type: str = "Mb", verbose: bool = True) -> float:
    xml_size = os.path.getsize(ir_path)
    bin_size = os.path.getsize(os.path.splitext(ir_path)[0] + ".bin")
    for t in ["bytes", "Kb", "Mb"]:
        if m_type == t:
            break
        xml_size /= 1024
        bin_size /= 1024
    model_size = xml_size + bin_size
    if verbose:
        print(f"Model graph (xml):   {xml_size:.3f} Mb")
        print(f"Model weights (bin): {bin_size:.3f} Mb")
        print(f"Model size:          {model_size:.3f} Mb")
    return model_size


def run_benchmark(model_path: str, shape=None, verbose: bool = True) -> float:
    command = f"benchmark_app -m {model_path} -d CPU -api async -t 15"
    if shape is not None:
        command += f' -shape [{",".join(str(x) for x in shape)}]'
    cmd_output = subprocess.check_output(command, shell=True)  # nosec
    if verbose:
        print(*str(cmd_output).split("\\n")[-9:-1], sep="\n")
    match = re.search(r"Throughput\: (.+?) FPS", str(cmd_output))
    return float(match.group(1))


class COCO128Dataset(torch.utils.data.Dataset):
    category_mapping = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        27,
        28,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        67,
        70,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
    ]

    def __init__(self, data_path, transform):
        super().__init__()
        self.transform = transform
        self.data_path = Path(data_path)
        self.images_path = self.data_path / "images" / "train2017"
        self.labels_path = self.data_path / "labels" / "train2017"
        self.image_ids = sorted(map(lambda p: int(p.stem), self.images_path.glob("*.jpg")))

    def __getitem__(self, item):
        image_id = self.image_ids[item]

        img = Image.open(self.images_path / f"{image_id:012d}.jpg")
        img_w, img_h = img.size
        target = dict(image_id=[image_id], boxes=[], labels=[])
        label_filepath = self.labels_path / f"{image_id:012d}.txt"
        if label_filepath.exists():
            with open(label_filepath, "r") as f:
                for box_descr in f.readlines():
                    category_id, rel_x, rel_y, rel_w, rel_h = tuple(map(float, box_descr.split(" ")))
                    box_x1, box_y1 = img_w * (rel_x - rel_w / 2), img_h * (rel_y - rel_h / 2)
                    box_x2, box_y2 = img_w * (rel_x + rel_w / 2), img_h * (rel_y + rel_h / 2)
                    target["boxes"].append((box_x1, box_y1, box_x2, box_y2))
                    target["labels"].append(self.category_mapping[int(category_id)])

        for k in target.keys():
            target[k] = torch.as_tensor(target[k], dtype=torch.float32 if k == "boxes" else torch.int64)
        img, target = self.transform(img, target)
        return img, target

    def __len__(self):
        return len(self.image_ids)


def validate(model, dataset, device):
    model.to(device)
    model.eval()
    metric = MeanAveragePrecision()
    with torch.no_grad():
        for img, target in tqdm(dataset, desc="Validating"):
            prediction = model(img.to(device)[None])[0]
            for k in prediction.keys():
                prediction[k] = prediction[k].to(torch.device("cpu"))
            metric.update([prediction], [target])
    computed_metrics = metric.compute()
    return computed_metrics["map_50"]


def transform_fn(data_item):
    # Skip label and add a batch dimension to an image tensor
    images, _ = data_item
    return images[None]


def main():
    # Download and prepare the COCO128 dataset
    dataset_path = download_dataset()
    weights_name = "SSD300_VGG16_Weights.DEFAULT"
    transform = torchvision.models.get_weight(weights_name).transforms()
    dataset = COCO128Dataset(dataset_path, lambda img, target: (transform(img), target))

    # Get the pretrained ssd300_vgg16 model from torchvision.models
    model = torchvision.models.get_model("ssd300_vgg16", weights=weights_name)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.cpu
    model.to(device)
    model.eval()

    # Disable NNCF tracing for some methods in order for the model to be properly traced by NNCF
    disable_tracing(GeneralizedRCNNTransform.normalize)
    disable_tracing(SSD.postprocess_detections)

    # Quantize model
    calibration_dataset = nncf.Dataset(dataset, transform_fn)
    quantized_model = nncf.quantize(model, calibration_dataset)

    # Convert to OpenVINO
    input_shape = [3, 480, 480]
    ov_model = mo.convert_model(model.cpu(), input_shape=[-1] + input_shape)
    ov_quantized_model = mo.convert_model(quantized_model.cpu(), input_shape=[-1] + input_shape)

    fp32_ir_path = f"{ROOT}/ssd300_vgg16_fp32.xml"
    ov.serialize(ov_model, fp32_ir_path)
    print(f"[1/7] Save FP32 model: {fp32_ir_path}")
    fp32_model_size = get_model_size(fp32_ir_path, verbose=True)

    int8_ir_path = f"{ROOT}/ssd300_vgg16_int8.xml"
    ov.serialize(ov_quantized_model, int8_ir_path)
    print(f"[2/7] Save INT8 model: {int8_ir_path}")
    int8_model_size = get_model_size(int8_ir_path, verbose=True)

    print("[3/7] Benchmark FP32 model:")
    fp32_fps = run_benchmark(fp32_ir_path, shape=[1] + input_shape, verbose=True)
    print("[4/7] Benchmark INT8 model:")
    int8_fps = run_benchmark(int8_ir_path, shape=[1] + input_shape, verbose=True)

    print("[5/7] Validate FP32 model:")
    torch.backends.cudnn.deterministic = True
    fp32_map = validate(model, dataset, device)
    print(f"mAP @ 0.5: {fp32_map:.3f}")

    print("[6/7] Validate INT8 model:")
    int8_map = validate(quantized_model, dataset, device)
    print(f"mAP @ 0.5: {int8_map:.3f}")

    print("[7/7] Report:")
    print(f"mAP drop: {fp32_map - int8_map:.3f}")
    print(f"Model compression rate: {fp32_model_size / int8_model_size:.3f}")
    # https://docs.openvino.ai/latest/openvino_docs_optimization_guide_dldt_optimization_guide.html
    print(f"Performance speed up (throughput mode): {int8_fps / fp32_fps:.3f}")


if __name__ == "__main__":
    main()
