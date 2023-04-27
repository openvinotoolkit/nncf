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

from functools import partial
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import openvino.runtime as ov
from openvino.tools.pot.api.samples.object_detection.data_loader import COCOLoader
from openvino.tools.pot.api.samples.object_detection.metric import MAP

import nncf

FILE = Path(__file__).resolve()
# Relative path to the `ssd_mobilenet_v1_fpn` directory.
ROOT = FILE.parent.relative_to(Path.cwd())
# Path to the directory where the original and quantized IR will be saved.
MODEL_DIR = ROOT / "ssd_mobilenet_v1_fpn_quantization"
# Path to COCO validation dataset.
DATASET_DIR = ROOT / "coco2017" / "val2017"
ANNOTATION_FILE = ROOT / "coco2017" / "instances_val2017.json"


ie = ov.Core()


def run_example():
    """
    Runs the SSD MobileNetV1 FPN quantize with accuracy control example.
    """
    # Step 1: Load the OpenVINO model.
    ir_model_xml = ROOT / "public" / "ssd_mobilenet_v1_fpn_coco" / "FP32" / "ssd_mobilenet_v1_fpn_coco.xml"
    ov_model = ie.read_model(ir_model_xml)

    # Step 2: Create data source.
    dataset_config = {
        "images_path": f"{DATASET_DIR}/",
        "annotation_path": ANNOTATION_FILE,
    }
    data_source = COCOLoader(dataset_config)

    # Step 4: Apply quantization algorithm.

    # Define the transformation method. This method should take a data item returned
    # per iteration through the `data_source` object and transform it into the model's
    # expected input that can be used for the model inference.
    def transform_fn(data):
        images, _ = data
        return np.expand_dims(images, 0)

    metric = MAP(91, data_source.labels)
    # Wrap framework-specific data source into the `nncf.Dataset` object.
    validation_dataset = nncf.Dataset(data_source, transform_fn)
    quantized_model = nncf.quantize_with_accuracy_control(
        ov_model,
        validation_dataset,
        validation_dataset,
        validation_fn=partial(validation_fn, metric=metric),
        max_drop=0.004,
        preset=nncf.QuantizationPreset.MIXED,
    )

    # Step 4: Save the quantized model.
    if not MODEL_DIR.exists():
        MODEL_DIR.mkdir()

    ir_qmodel_xml = MODEL_DIR / "ssd_mobilenet_v1_fpn_quantized.xml"
    ir_qmodel_bin = MODEL_DIR / "ssd_mobilenet_v1_fpn_quantized.bin"
    ov.serialize(quantized_model, str(ir_qmodel_xml), str(ir_qmodel_bin))

    # Step 6: Compare the accuracy of the original and quantized models.
    print("Checking the accuracy of the original model:")
    compiled_model = ie.compile_model(ov_model, device_name="CPU")
    metric = validation_fn(compiled_model, data_source, data_source.labels)
    print(f"mAP: {metric}")

    print("Checking the accuracy of the quantized model:")
    compiled_model = ie.compile_model(quantized_model, device_name="CPU")
    metric = validation_fn(compiled_model, data_source, data_source.labels)
    print(f"mAP: {metric}")

    # Step 7: Compare Performance of the original and quantized models.
    # benchmark_app -m ssd_mobilenet_v1_fpn_quantization/ssd_mobilenet_v1_fpn.xml -d CPU -api async
    # benchmark_app -m ssd_mobilenet_v1_fpn_quantization/ssd_mobilenet_v1_fpn_quantized.xml -d CPU -api async


def validation_fn(compiled_model: ov.CompiledModel, validation_dataset: Iterable[Any], metric) -> float:
    output_layer = compiled_model.output(0)

    for images, labels in validation_dataset:
        input_data = np.expand_dims(images, 0)
        output = compiled_model([input_data])[output_layer]
        metric.update(output, [labels])

    return metric.avg_value["MAP"].item()


if __name__ == "__main__":
    run_example()
