# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Tuple

import numpy as np
import torch
from tqdm import tqdm
from ultralytics.data.utils import check_det_dataset
from ultralytics.engine.validator import BaseValidator as Validator
from ultralytics.models.yolo import YOLO
from ultralytics.utils.torch_utils import de_parallel


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


def prepare_validation(model: YOLO, data: str) -> Tuple[Validator, torch.utils.data.DataLoader]:
    # custom = {"rect": True, "batch": 1}  # method defaults
    # rect: false forces to resize all input pictures to one size
    custom = {"rect": False, "batch": 1}  # method defaults
    args = {**model.overrides, **custom, "mode": "val"}  # highest priority args on the right

    validator = model._smart_load("validator")(args=args, _callbacks=model.callbacks)
    stride = 32  # default stride
    validator.stride = stride  # used in get_dataloader() for padding
    validator.data = check_det_dataset(data)
    validator.init_metrics(de_parallel(model))

    data_loader = validator.get_dataloader(validator.data.get(validator.args.split), validator.args.batch)
    return validator, data_loader


def validate(model, data_loader: torch.utils.data.DataLoader, validator: Validator) -> Tuple[Dict, int, int]:
    with torch.no_grad():
        for batch in data_loader:
            batch = validator.preprocess(batch)
            preds = model(batch["img"])
            preds = validator.postprocess(preds)
            validator.update_metrics(preds, batch)
        stats = validator.get_stats()
    return stats, validator.seen, validator.nt_per_class.sum()


def main(torch_fx):
    # ultralytics @ git+https://github.com/THU-MIG/yolov10.git@2c36ab0f108efdd17c7e290564bb845ccb6844d8
    # pip install git+https://github.com/THU-MIG/yolov10.git
    # pip install huggingface-hub
    # yolo_model = YOLO("yolov10n.pt")

    yolo_model = YOLO("yolov8n")

    model_type = "torch"
    model = yolo_model.model
    if torch_fx:
        model = torch.compile(model)
        model_type = "FX"
    print(f"FP32 {model_type} model validation results:")
    validator, data_loader = prepare_validation(yolo_model, "coco128.yaml")
    stats, total_images, total_objects = validate(model, tqdm(data_loader), validator)
    print_statistics(stats, total_images, total_objects)


if __name__ == "__main__":
    print("Torch model:")
    main(torch_fx=False)
    print("Torch FX model:")
    main(torch_fx=True)
