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
from typing import List

import openvino as ov
import tensorflow as tf
import tensorflow_datasets as tfds
from rich.progress import track

import nncf

ROOT = Path(__file__).parent.resolve()
WEIGHTS_URL = "https://huggingface.co/alexsu52/mobilenet_v2_imagenette/resolve/main/tf_model.h5"
DATASET_CLASSES = 10


def validate(model: ov.Model, val_loader: tf.data.Dataset) -> tf.Tensor:
    compiled_model = ov.compile_model(model, device_name="CPU")
    output = compiled_model.outputs[0]

    metric = tf.keras.metrics.CategoricalAccuracy(name="acc@1")
    for images, labels in track(val_loader):
        pred = compiled_model(images.numpy())[output]
        metric.update_state(labels, pred)

    return metric.result()


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


###############################################################################
# Create a Tensorflow model and dataset


def center_crop(image: tf.Tensor, image_size: int, crop_padding: int) -> tf.Tensor:
    shape = tf.shape(image)
    image_height = shape[0]
    image_width = shape[1]

    padded_center_crop_size = tf.cast(
        ((image_size / (image_size + crop_padding)) * tf.cast(tf.minimum(image_height, image_width), tf.float32)),
        tf.int32,
    )

    offset_height = ((image_height - padded_center_crop_size) + 1) // 2
    offset_width = ((image_width - padded_center_crop_size) + 1) // 2

    image = tf.image.crop_to_bounding_box(
        image,
        offset_height=offset_height,
        offset_width=offset_width,
        target_height=padded_center_crop_size,
        target_width=padded_center_crop_size,
    )
    image = tf.image.resize(image, [image_size, image_size], method=tf.image.ResizeMethod.BILINEAR)
    return image


def preprocess_for_eval(image, label):
    image = center_crop(image, 224, 32)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    label = tf.one_hot(label, DATASET_CLASSES)
    return image, label


def preprocess_for_train(image, label):
    image = tf.image.resize_with_crop_or_pad(image, 256, 256)
    image = tf.image.random_crop(image, [224, 224, 3])
    image = tf.image.convert_image_dtype(image, tf.float32)
    label = tf.one_hot(label, DATASET_CLASSES)
    return image, label


train_dataset = tfds.load("imagenette/320px-v2", split="train", shuffle_files=True, as_supervised=True)
train_dataset = train_dataset.map(preprocess_for_train).shuffle(1024).batch(128)

val_dataset = tfds.load("imagenette/320px-v2", split="validation", shuffle_files=False, as_supervised=True)
val_dataset = val_dataset.map(preprocess_for_eval).batch(128)

weights_path = tf.keras.utils.get_file("mobilenet_v2_imagenette_weights.h5", WEIGHTS_URL, cache_subdir="models")
tf_model = tf.keras.applications.MobileNetV2(weights=weights_path, classes=DATASET_CLASSES)

###############################################################################
# Quantize a Tensorflow model
#
# The transformation function transforms a data item into model input data.
#
# To validate the transform function use the following code:
# >> for data_item in val_loader:
# >>    model(transform_fn(data_item))


def transform_fn(data_item):
    images, _ = data_item
    return images


# The calibration dataset is a small, no label, representative dataset
# (~100-500 samples) that is used to estimate the range, i.e. (min, max) of all
# floating point activation tensors in the model, to initialize the quantization
# parameters.
#
# The easiest way to define a calibration dataset is to use a training or
# validation dataset and a transformation function to remove labels from the data
# item and prepare model input data. The quantize method uses a small subset
# (default: 300 samples) of the calibration dataset.

calibration_dataset = nncf.Dataset(val_dataset, transform_fn)
tf_quantized_model = nncf.quantize(tf_model, calibration_dataset)

tf_quantized_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=[tf.keras.metrics.CategoricalAccuracy()],
)

tf_quantized_model.fit(train_dataset, epochs=3, verbose=1)

# Removes auxiliary layers and operations added during the quantization process,
# resulting in a clean, fully quantized model ready for deployment.
stripped_model = nncf.strip(tf_quantized_model)

###############################################################################
# Benchmark performance, calculate compression rate and validate accuracy

ov_model = ov.convert_model(tf_model, share_weights=False)
ov_quantized_model = ov.convert_model(stripped_model, share_weights=False)

fp32_ir_path = ROOT / "mobilenet_v2_fp32.xml"
ov.save_model(ov_model, fp32_ir_path, compress_to_fp16=False)
print(f"[1/7] Save FP32 model: {fp32_ir_path}")
fp32_model_size = get_model_size(fp32_ir_path)

int8_ir_path = ROOT / "mobilenet_v2_int8.xml"
ov.save_model(ov_quantized_model, int8_ir_path)
print(f"[2/7] Save INT8 model: {int8_ir_path}")
int8_model_size = get_model_size(int8_ir_path)

print("[3/7] Benchmark FP32 model:")
fp32_fps = run_benchmark(fp32_ir_path, shape=[1, 224, 224, 3])
print("[4/7] Benchmark INT8 model:")
int8_fps = run_benchmark(int8_ir_path, shape=[1, 224, 224, 3])

print("[5/7] Validate OpenVINO FP32 model:")
fp32_top1 = validate(ov_model, val_dataset)
print(f"Accuracy @ top1: {fp32_top1:.3f}")

print("[6/7] Validate OpenVINO INT8 model:")
int8_top1 = validate(ov_quantized_model, val_dataset)
print(f"Accuracy @ top1: {int8_top1:.3f}")

print("[7/7] Report:")
print(f"Accuracy drop: {fp32_top1 - int8_top1:.3f}")
print(f"Model compression rate: {fp32_model_size / int8_model_size:.3f}")
# https://docs.openvino.ai/latest/openvino_docs_optimization_guide_dldt_optimization_guide.html
print(f"Performance speed up (throughput mode): {int8_fps / fp32_fps:.3f}")
